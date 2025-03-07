import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from omegaconf import II

from genhpf.models import register_model
from genhpf.models.genhpf import GenHPF, GenHPFConfig

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
        self.tasks = cfg.tasks
        assert len(self.num_labels) == len(
            cfg.tasks
        ), "The number of num_labels must be equal to the number of tasks"

        self.final_proj = nn.ModuleDict()
        for i, task in enumerate(cfg.tasks):
            self.final_proj[task] = nn.Linear(cfg.agg_embed_dim, self.num_labels[i])

    def get_logits(self, sample, net_output):
        if len(self.tasks) == 1:
            return net_output[self.tasks[0]]
        else:
            return net_output

    def get_targets(self, sample, net_output):
        if len(self.tasks) == 1:
            return sample["label"][self.tasks[0]]
        else:
            return sample["label"]

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

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        ret = {}
        for task, proj in self.final_proj.items():
            ret[task] = proj(x)

        return ret

    def get_finetuning_parameter_names(self):
        ret = []
        ret.extend(["final_proj" + "." + x[0] for x in self.final_proj.named_parameters()])
        return ret
