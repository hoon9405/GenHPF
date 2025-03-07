import logging
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn

from genhpf.models import register_model
from genhpf.models.genhpf import GenHPF, GenHPFConfig
from genhpf.modules import GradMultiply, GumbelVectorQuantizer, LayerNorm
from genhpf.utils import utils
from genhpf.utils.data_utils import compute_mask_indices

logger = logging.getLogger(__name__)


@dataclass
class GenHPFWav2Vec2Config(GenHPFConfig):
    logit_temp: float = field(default=0.1, metadata={"help": "temperature to divide logits by"})
    latent_vars: int = field(
        default=320, metadata={"help": "number of latent variables V in each group of the codebook"}
    )
    latent_groups: int = field(
        default=2, metadata={"help": "number of groups G of latent variables in the codebook"}
    )
    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )
    final_dim: int = field(
        default=128, metadata={"help": "project final representations and targets to this many dimensions."}
    )

    num_negatives: int = field(
        default=25, metadata={"help": "number of negative examples from the same sample"}
    )
    codebook_negatives: int = field(default=0, metadata={"help": "number of negative examples codebook"})

    # mask
    mask_prob: float = field(default=0.65, metadata={"help": "probability of replacing a token with mask"})
    mask_length: int = field(default=1, metadata={"help": "mask length"})
    no_mask_overlap: bool = field(default=False, metadata={"help": "whether to allow masks to overlap"})
    mask_min_space: int = field(
        default=0, metadata={"help": "min space between spans (if no overlap is enabled)"}
    )

    feature_grad_mult: float = field(
        default=0.1, metadata={"help": "multiply event encoder gradients by this"}
    )
    dropout_input: float = field(default=0.1, metadata={"help": "dropout to apply to the input"})
    dropout_features: float = field(default=0.1, metadata={"help": "dropout to apply to the features"})


@register_model("genhpf_wav2vec2", dataclass=GenHPFWav2Vec2Config)
class GenHPFWav2Vec2(GenHPF):
    def __init__(self, cfg: GenHPFWav2Vec2Config):
        super().__init__(cfg)

        self.logit_temp = cfg.logit_temp

        self.mask_prob = cfg.mask_prob
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.feature_grad_mult = cfg.feature_grad_mult

        self.codebook_negatives = cfg.codebook_negatives

        self.quantizer = GumbelVectorQuantizer(
            dim=cfg.agg_embed_dim,
            num_vars=cfg.latent_vars,
            temp=cfg.latent_temp,
            groups=cfg.latent_groups,
            combine_groups=False,
            vq_dim=cfg.agg_embed_dim,
            time_first=True,
        )

        self.layer_norm = LayerNorm(cfg.agg_embed_dim)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.project_q = nn.Linear(cfg.agg_embed_dim, cfg.final_dim)
        self.final_proj = nn.Linear(cfg.agg_embed_dim, cfg.final_dim)
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.agg_embed_dim).uniform_())
        self.cross_sample_negatives = 0
        self.n_negatives = cfg.num_negatives

    @classmethod
    def build_model(cls, cfg):
        """Build a new model instance."""
        return cls(cfg)

    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)

        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def apply_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_indices=None,
    ):
        bsz, tsz, csz = x.shape

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    shape=(bsz, tsz),
                    padding_mask=padding_mask,
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_type="static",
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        return x, mask_indices

    def sample_negatives(self, y, num):
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        batch_size, time_size, feature_size = y.shape
        y = y.view(-1, feature_size)  # B x T x C -> (B x T) x C

        cross_high = time_size * batch_size
        high = time_size
        with torch.no_grad():
            assert high > 1, f"{batch_size, time_size, feature_size}"

            if self.n_negatives > 0:
                time_sizes = utils.buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()
                neg_idxs = torch.randint(low=0, high=high - 1, size=(batch_size, self.n_negatives * num))
                neg_idxs[neg_idxs >= time_sizes] += 1

            if self.cross_sample_negatives > 0:
                time_sizes = (
                    utils.buffered_arange(num).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0, high=cross_high - 1, size=(batch_size, self.cross_sample_negatives * num)
                )
                cross_neg_idxs[cross_neg_idxs >= time_sizes] += 1

        if self.n_negatives > 0:
            for i in range(1, batch_size):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            batch_size, num, self.n_negatives + self.cross_sample_negatives, feature_size
        ).permute(
            2, 0, 1, 3
        )  # to N x B x T x C

        return negs, neg_idxs

    def get_logits(self, sample, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))

        return logits

    def get_targets(self, sample, net_output):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append((net_output["num_vars"] - net_output["prob_perplexity"]) / net_output["num_vars"])

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor = None,
        dpe_ids: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.feature_grad_mult > 0:
            features_ret = super().forward(
                input_ids=input_ids,
                type_ids=type_ids,
                dpe_ids=dpe_ids,
                padding_mask=padding_mask,
                encoder_only=True,
                **kwargs,
            )
            features = features_ret["x"]
            padding_mask = features_ret["padding_mask"]
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = super().forward(
                    input_ids=input_ids,
                    type_ids=type_ids,
                    dpe_ids=dpe_ids,
                    padding_mask=padding_mask,
                    encoder_only=True,
                    **kwargs,
                )
                features = features_ret["x"]
                padding_mask = features_ret["padding_mask"]

        features_pen = features.float().pow(2).mean()

        features = self.layer_norm(features)
        unmasked_features = features.clone()

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if padding_mask is None:
            padding_mask = input_ids[:, :, 1].eq(0).to(features.device)

        x, mask_indices = self.apply_mask(features, padding_mask, mask_indices=None)
        if mask_indices is not None:
            y = unmasked_features[mask_indices].view(
                unmasked_features.size(0), -1, unmasked_features.size(-1)
            )
        else:
            y = unmasked_features

        x = self.event_aggregator(x, src_key_padding_mask=padding_mask)

        q = self.quantizer(y, produce_targets=False)
        y = q["x"]
        num_vars = q["num_vars"]
        code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]

        y = self.project_q(y)

        negs, _ = self.sample_negatives(y, y.size(1))

        if self.codebook_negatives > 0:
            cb_negs = self.quantizer.sample_from_codebook(y.size(0) * y.size(1), self.codebook_negatives)
            cb_negs = cb_negs.view(self.codebook_negatives, y.size(0), y.size(1), -1)
            cb_negs = self.project_q(cb_negs)
            negs = torch.cat([negs, cb_negs], dim=0)

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        output = {"x": x, "features_pen": features_pen, "mask_indices": mask_indices}
        if prob_ppl is not None:
            output["prob_perplexity"] = prob_ppl
            output["code_perplexity"] = code_ppl
            output["num_vars"] = num_vars
            output["temp"] = curr_temp

        return output

    def get_pretraining_parameter_names(self):
        ret = []
        ret.append("mask_emb")
        ret.extend(["quantizer" + "." + x[0] for x in self.quantizer.named_parameters()])
        ret.extend(["layer_norm" + "." + x[0] for x in self.layer_norm.named_parameters()])
        ret.extend(["project_q" + "." + x[0] for x in self.project_q.named_parameters()])
        ret.extend(["final_proj" + "." + x[0] for x in self.final_proj.named_parameters()])
        return ret
