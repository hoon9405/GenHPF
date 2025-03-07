import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import genhpf.utils.utils as utils
from genhpf.criterions import BaseCriterion, register_criterion
from genhpf.criterions.criterion import CriterionConfig
from genhpf.loggings import meters, metrics
from genhpf.loggings.meters import safe_round


@dataclass
class BinaryCrossEntropyWithLogitsConfig(CriterionConfig):
    threshold: float = field(default=0.5, metadata={"help": "threshold value for binary classification"})


@register_criterion("binary_cross_entropy_with_logits", dataclass=BinaryCrossEntropyWithLogitsConfig)
class BinaryCrossEntropyWithLogits(BaseCriterion):
    def __init__(self, cfg: BinaryCrossEntropyWithLogitsConfig):
        super().__init__(cfg)

        if self.task_names is not None and len(self.task_names) > 1:
            raise ValueError(
                "binary_cross_entropy_with_logits only supports single task training."
                " if you want to train multiple tasks, use multi_task_criterion instead."
            )

        self.threshold = cfg.threshold

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, sample=None, net_output=None, model=None
    ) -> Tuple[torch.Tensor, List[float]]:
        assert (
            logits.size() == targets.size()
        ), f"logits and targets must have the same size: {logits.size()} vs {targets.size()}"
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(input=logits, target=targets, reduction="sum")
        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, targets: torch.Tensor) -> int:
        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        else:
            sample_size = targets.numel()
        return sample_size

    def get_logging_outputs(
        self, logging_output, logits: torch.Tensor, targets: torch.Tensor, sample=None
    ) -> List[Dict[str, Any]]:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            outputs = probs > self.threshold

            if probs.numel() == 0:
                corr = 0
                count = 0
            else:
                count = float(probs.numel())
                corr = (outputs == targets).sum().item()

            logging_output["correct"] = corr
            logging_output["count"] = count

            # report aucs only in eval mode
            if not self.training:
                logging_output["_y_true"] = targets.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

        return logging_output

    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]], prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        if prefix is None:
            prefix = ""
        elif prefix is not None and not prefix.endswith("_"):
            prefix = prefix + "_"

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))

        metrics.log_scalar(f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3)

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", []) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", []) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, f"_{prefix}auc", y_score, y_true)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}accuracy",
                lambda meters: safe_round(meters[f"_{prefix}correct"].sum / meters[f"_{prefix}total"].sum, 5)
                if meters[f"_{prefix}total"].sum > 0
                else float("nan"),
            )

    def post_validate(self, stats, agg, **kwargs):
        for key in agg.keys():
            if key.startswith("_") and key.endswith("auc"):
                stats[key[1:-3] + "auroc"] = agg[key].auroc
                stats[key[1:-3] + "auprc"] = agg[key].auprc

        return stats
