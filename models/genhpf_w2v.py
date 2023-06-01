import logging

import torch
import torch.nn as nn

from models import register_model, MODEL_REGISTRY
from modules import GradMultiply, LayerNorm, GumbelVectorQuantizer

from utils import utils
from utils.data_utils import compute_mask_indices

logger = logging.getLogger(__name__)

@register_model("GenHPF_w2v")
class GenHPF_w2v(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb_type_model = self._emb_type_model.build_model(args)

        self.quantizer = GumbelVectorQuantizer(
            dim=args.embed_dim,
            num_vars=args.latent_vars,
            temp=args.latent_temp,
            groups=args.latent_groups,
            combine_groups=False,
            vq_dim=args.embed_dim,
            time_first=True,
        )

        self.pred_model = self._pred_model.build_model(args)
        
        self.layer_norm = LayerNorm(args.embed_dim)
        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)
        self.project_q = nn.Linear(args.embed_dim, args.final_dim)
        self.final_proj = nn.Linear(args.embed_dim, args.final_dim)
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.embed_dim).uniform_()
        )
        self.cross_sample_negatives = 0
        self.n_negatives = self.args.num_negatives
        
    @property
    def _emb_type_model(self):
        if self.args.structure=='hi':
            return MODEL_REGISTRY['eventencoder']
        else:
            return MODEL_REGISTRY['FeatEmb']
    
    @property
    def _pred_model(self):
        return MODEL_REGISTRY[self.args.pred_model]
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    @classmethod
    def from_pretrained(cls, args, state_dict=0):
        model = cls(args)
        model.load_state_dict(state_dict)
    
        return model 

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim= 0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)

        logits = logits / self.args.logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def apply_mask( 
        self,
        x,
        padding_mask,
        mask_indices = None,
        ):
        B, T, C = x.shape
        
        if self.args.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.args.mask_prob,
                    self.args.mask_length,
                    self.args.mask_selection,
                    self.args.mask_other,
                    min_masks = 2,
                    no_overlap = self.args.no_mask_overlap,
                    min_space = self.args.mask_min_space,
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
        y = y.view(-1, feature_size) # B x T x C -> (B x T) x C

        cross_high = time_size * batch_size
        high = time_size
        with torch.no_grad():
            assert high > 1, f"{batch_size, time_size, feature_size}"

            if self.n_negatives > 0:
                time_sizes = (
                    utils.buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )
                neg_idxs = torch.randint(
                    low = 0, high = high - 1, size = (batch_size, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= time_sizes] += 1

            if self.cross_sample_negatives > 0:
                time_sizes = (
                    utils.buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low = 0,
                    high = cross_high - 1,
                    size = (batch_size, self.cross_sample_negatives * num)
                )
                cross_neg_idxs[cross_neg_idxs >= time_sizes] += 1
        
        if self.n_negatives > 0:
            for i in range(1, batch_size):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim =1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            batch_size, num, self.n_negatives + self.cross_sample_negatives, feature_size
        ).permute(
            2, 0, 1, 3
        ) # to N x B x T x C

        return negs, neg_idxs


    def forward(self, **kwargs):

        all_codes_embs = self.emb_type_model(**kwargs)  # (B, S, E)

        if self.args.feature_grad_mult != 0:
            all_codes_embs = GradMultiply.apply(all_codes_embs, self.args.feature_grad_mult)
        
        features_pen = all_codes_embs.float().pow(2).mean()

        features = self.layer_norm(all_codes_embs)
        unmasked_features = features.clone()

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        input_ids = kwargs.get('input_ids')

        padding_mask = input_ids[:, :, 1].eq(0).to(features.device) 

        all_codes_embs, mask_indices = self.apply_mask( #TODO: need modify
            features,
            padding_mask,
            mask_indices=None
        )
        if mask_indices is not None:
            y = unmasked_features[mask_indices].view(
                unmasked_features.size(0), -1, unmasked_features.size(-1)
            )
        else:
            y = unmasked_features

        x = self.pred_model(all_codes_embs, **kwargs)

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']

            y = self.project_q(y)

            negs, _ = self.sample_negatives(y, y.size(1))

            if self.args.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.args.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.args.codebook_negatives, y.size(0), y.size(1), -1
                )
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        output = {'x': x, 'features_pen': features_pen, 'mask_indices': mask_indices}
        if prob_ppl is not None:
            output["prob_perplexity"] = prob_ppl
            output["code_perplexity"] = code_ppl
            output["num_vars"] = num_vars
            output["temp"] = curr_temp

        return output


    def get_outputs(self, net_output, task):

        logits = net_output["x"]
        logits = logits.transpose(0,2)
        logits = logits.reshape(-1, logits.size(-1))
        
        return logits
    

    def get_targets(self, sample, net_output, task):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype = torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen
