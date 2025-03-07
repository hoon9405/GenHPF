import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F

import genhpf.utils.utils as utils
from genhpf.criterions import BaseCriterion, register_criterion
from genhpf.criterions.criterion import CriterionConfig
from genhpf.loggings import metrics


@dataclass
class SimCLRCriterionConfig(CriterionConfig):
    temp: float = field(default=0.1, metadata={"help": "temperature to divide logits by"})


@register_criterion("simclr_criterion", dataclass=SimCLRCriterionConfig)
class SimCLRCriterion(BaseCriterion):
    def __init__(self, cfg: SimCLRCriterionConfig):
        super().__init__(cfg)

        self.temp = cfg.temp

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor = None, sample=None, net_output=None, model=None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute the loss given the logits and targets from the model."""
        logits = F.normalize(logits, dim=1)  # normalize logits

        bsz = int(logits.shape[0] / 2)

        mask = 1 - torch.eye(bsz * 2, dtype=torch.uint8).to(logits.device)
        pos_ind = (
            torch.arange(bsz * 2).to(logits.device),
            2
            * torch.arange(bsz, dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, 2)
            .view(-1, 1)
            .squeeze()
            .to(logits.device),
        )
        neg_mask = torch.ones((bsz * 2, bsz * 2 - 1), dtype=torch.uint8).to(logits.device)
        neg_mask[pos_ind] = 0

        # Cosine similarity computation
        sim_matrix = torch.matmul(logits, logits.T)  # cosine similarity computation

        # Eliminate similarity between same view
        sim_matrix = torch.masked_select(sim_matrix, mask.bool()).view(sim_matrix.size(0), -1)

        positives = sim_matrix[pos_ind].unsqueeze(1)
        negatives = torch.masked_select(sim_matrix, neg_mask.bool()).view(sim_matrix.size(0), -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temp

        target = torch.zeros((logits.size(0),), dtype=torch.long).to(logits.device)

        loss = F.cross_entropy(logits, target, reduction="sum")

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, targets: torch.Tensor = None) -> int:
        return sample["net_input"]["input_ids"].size(0)

    def get_logging_outputs(self, logging_output, logits, target, sample=None, net_output=None):
        return logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        if prefix is None:
            prefix = ""
        elif prefix is not None and not prefix.endswith("_"):
            prefix = prefix + "_"

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))

        metrics.log_scalar(f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3)
