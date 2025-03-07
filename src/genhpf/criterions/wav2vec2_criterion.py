import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import genhpf.utils.utils as utils
from genhpf.criterions import BaseCriterion, register_criterion
from genhpf.criterions.criterion import CriterionConfig
from genhpf.loggings import metrics
from genhpf.loggings.meters import safe_round


@dataclass
class Wav2Vec2CriterionConfig(CriterionConfig):
    loss_weights: Optional[List[float]] = field(
        default=None, metadata={"help": "weights for additional loss terms (not first one)"}
    )


@register_criterion("wav2vec2_criterion", dataclass=Wav2Vec2CriterionConfig)
class Wav2Vec2Criterion(BaseCriterion):
    def __init__(self, cfg: Wav2Vec2CriterionConfig):
        super().__init__(cfg)

        self.loss_weights = cfg.loss_weights

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, sample=None, net_output=None, model=None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute the loss given the logits and targets from the model."""

        losses = []

        loss = F.cross_entropy(logits, targets, reduction="sum")
        losses.append(loss.detach().item())

        sample_size = self.get_sample_size(sample, targets)

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p.detach().item())

        return loss, losses

    def get_sample_size(self, sample, targets):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        else:
            sample_size = targets.numel()
        return sample_size

    def get_logging_outputs(
        self, logging_output, logits: torch.Tensor, targets: torch.Tensor, sample=None
    ) -> List[Dict[str, Any]]:
        """
        Get the logging output to display while training
        """
        with torch.no_grad():
            if logits.numel() == 0:
                corr = 0
                count = 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0

                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = float(max.numel())

            logging_output["correct"] = corr
            logging_output["count"] = count
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

        builtin_keys = {"loss", "sample_size", "correct", "count"}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        prefix + k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(prefix + k, val / len(logging_outputs), round=3)
