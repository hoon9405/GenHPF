import logging
from dataclasses import dataclass, field

import torch

from genhpf.models import register_model
from genhpf.models.genhpf import GenHPF, GenHPFConfig
from genhpf.modules import GatherLayer
from genhpf.utils import distributed_utils as dist_utils

logger = logging.getLogger(__name__)


@dataclass
class GenHPFSimCLRConfig(GenHPFConfig):
    all_gather: bool = field(
        default=True, metadata={"help": "whether or not to apply all gather across different gpus"}
    )


@register_model("genhpf_simclr", dataclass=GenHPFSimCLRConfig)
class GenHPFSimCLR(GenHPF):
    def __init__(self, cfg: GenHPFSimCLRConfig):
        super().__init__(cfg)

        self.all_gather = cfg.all_gather

    @classmethod
    def build_model(cls, cfg):
        """Build a new model instance."""
        return cls(cfg)

    def get_logits(self, sample, net_output):
        return net_output

    def get_targets(self, sample, net_output):
        return None

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

        if self.all_gather and dist_utils.get_data_parallel_world_size() > 1:
            x = torch.cat(GatherLayer.apply(x), dim=0)

        return x
