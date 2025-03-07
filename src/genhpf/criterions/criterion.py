from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.modules.loss import _Loss

from genhpf.configs import BaseConfig
from genhpf.models.genhpf import GenHPF


@dataclass
class CriterionConfig(BaseConfig):
    task_names: Optional[List[str]] = field(
        default=None, metadata={"help": "a list of task names for multi-task learning"}
    )
    num_labels: Optional[List[int]] = field(
        default=None, metadata={"help": "a list of number of labels for each task"}
    )


class BaseCriterion(_Loss):
    def __init__(self, cfg: CriterionConfig):
        super().__init__()
        self.cfg = cfg

        self.task_names = cfg.task_names
        self.num_labels = cfg.num_labels

    @classmethod
    def build_criterion(cls, cfg: CriterionConfig):
        """Construct a new criterion instance."""
        return cls(cfg)

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, sample=None, net_output=None, model=None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute the loss given the logits and targets from the model."""
        raise NotImplementedError("Criterion must implement the `compute_loss` method")

    def get_sample_size(self, sample, targets: torch.Tensor) -> int:
        """Get the sample size, which is used as the denominator for the gradient."""
        raise NotImplementedError("Criterion must implement the `get_sample_size` method")

    def get_logging_outputs(
        self, logging_output, logits: torch.Tensor, targets: torch.Tensor, sample=None
    ) -> List[Dict[str, Any]]:
        """
        Get the logging output to display while training
        """
        raise NotImplementedError("Criterion must implement the `get_logging_outputs` method")

    def forward(self, model: GenHPF, sample, return_net_output=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1. the loss
        2. the sample size, which is used as the denominator for the gradient
        3. logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(sample, net_output)
        targets = model.get_targets(sample, net_output)

        loss, losses_to_log = self.compute_loss(
            logits, targets, sample=sample, net_output=net_output, model=model
        )
        sample_size = self.get_sample_size(sample, targets)

        logging_output = {}
        if len(losses_to_log) > 1:
            logging_output["loss"] = loss.item()
            for i, l in enumerate(losses_to_log):
                logging_output[f"loss_{i}"] = l
        else:
            logging_output["loss"] = losses_to_log[0]
        logging_output["sample_size"] = sample_size
        logging_output = self.get_logging_outputs(logging_output, logits, targets, sample)

        if return_net_output:
            return loss, sample_size, logging_output, net_output
        else:
            return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(stats: Dict[str, Any], prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError
