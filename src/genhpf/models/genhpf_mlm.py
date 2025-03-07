import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import II

from genhpf.models import register_model
from genhpf.models.genhpf import GenHPF, GenHPFConfig

logger = logging.getLogger(__name__)


@dataclass
class GenHPFMLMConfig(GenHPFConfig):
    ignore_index: int = II("dataset.ignore_index")


@register_model("genhpf_mlm", dataclass=GenHPFMLMConfig)
class GenHPFMLM(GenHPF):
    def __init__(self, cfg: GenHPFMLMConfig):
        super().__init__(cfg)

        self.ignore_index = cfg.ignore_index

        self.input_ids_proj = nn.Linear(cfg.agg_embed_dim, cfg.vocab_size)

    @classmethod
    def build_model(cls, cfg):
        """Build a new model instance."""
        return cls(cfg)

    def get_logits(self, sample, net_output):
        masked_indices = torch.where(
            (sample["input_label"] > 0) & (sample["input_label"] != self.ignore_index)
        )
        return net_output["input_ids"][masked_indices]

    def get_targets(self, sample, net_output):
        masked_indices = torch.where(
            (sample["input_label"] > 0) & (sample["input_label"] != self.ignore_index)
        )
        return sample["input_label"][masked_indices]

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor = None,
        dpe_ids: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        x, padding_mask = super().forward(
            input_ids=input_ids, type_ids=type_ids, dpe_ids=dpe_ids, padding_mask=padding_mask, **kwargs
        )

        input_ids = self.input_ids_proj(x)

        return {"input_ids": input_ids, "padding_mask": padding_mask}

    def get_pretraining_parameter_names(self):
        ret = []
        ret.extend(["input_ids_proj" + "." + x[0] for x in self.input_ids_proj.named_parameters()])
        return ret
