from dataclasses import dataclass, field
from typing import List
from omegaconf import II
import logging

import torch
import torch.nn as nn

from genhpf.models import register_model
from genhpf.models.genhpf import GenHPFConfig, GenHPF

logger = logging.getLogger(__name__)

@dataclass
class GenHPFPredictorConfig(GenHPFConfig):
    tasks: List[str] = II("criterion.task_names")
    num_labels: List[int] = II("criterion.num_labels")

@register_model("genhpf_predictor", dataclass=GenHPFPredictorConfig)
class GenHPFPredictor(GenHPF):
    def __init__(self, cfg: GenHPFPredictorConfig):
        super().__init__(cfg)
        
        self.num_labels = cfg.num_labels
        assert len(self.num_labels) == len(cfg.tasks), (
            "The number of num_labels must be equal to the number of tasks"
        )
        
        self.final_proj = nn.ModuleDict()
        for i, task in enumerate(cfg.tasks):
            self.final_proj[task] = nn.Linear(cfg.agg_embed_dim, self.num_labels[i])

    @classmethod
    def build_model(cls, cfg):
        """Build a new model instance."""
        return cls(cfg)

    def get_logits(self, net_output):
        return net_output

    def get_targets(self, sample, net_output):
        return sample["label"]

    def forward(
        self,
        input_ids: torch.Tensor,
        times: torch.Tensor = None,
        type_ids: torch.Tensor = None,
        dpe_ids: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        **kwargs
    ):
        x, padding_mask = super().forward(
            input_ids=input_ids,
            times=times,
            type_ids=type_ids,
            dpe_ids=dpe_ids,
            padding_mask=padding_mask,
            **kwargs
        )

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        ret = {}
        for task, proj in self.final_proj.items():
            ret[task] = proj(x)

        return ret