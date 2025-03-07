import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import genhpf.utils.utils as utils
from genhpf.criterions import BaseCriterion, register_criterion
from genhpf.criterions.criterion import CriterionConfig
from genhpf.loggings import metrics
from genhpf.models.genhpf import GenHPF

from . import build_criterion


@dataclass
class MultiTaskCriterionConfig(CriterionConfig):
    task_loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "weights for each loss term. if given, has to be a float list of size " "n_criterions"
        },
    )
    args: Any = field(
        default=None,
        metadata={
            "help": "configurations for each criterion where the name of each argument should "
            "match with the corresponding task name."
        },
    )


@register_criterion("multi_task_criterion", dataclass=MultiTaskCriterionConfig)
class MultiTaskCriterion(BaseCriterion):
    def __init__(self, cfg: MultiTaskCriterionConfig):
        super().__init__(cfg)

        criterions = {}
        for task_name in self.task_names:
            criterion_cfg = getattr(cfg.args, task_name)
            criterions[task_name] = build_criterion(criterion_cfg)
        self.criterions = criterions

        if cfg.task_loss_weights is None:
            self.task_loss_weights = [1.0] * len(criterions)
        else:
            self.task_loss_weights = cfg.task_loss_weights

    def forward(self, model: GenHPF, sample, return_net_output=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(sample, net_output)
        targets = model.get_targets(sample, net_output)

        if not isinstance(logits, dict):
            logits = {self.task_names[0]: logits}
        if not isinstance(targets, dict):
            targets = {self.task_names[0]: targets}

        if len(logits) != len(self.task_names) or len(targets) != len(self.task_names):
            raise ValueError(
                "number of logits and targets should be equal to the number of tasks. "
                f"got {len(logits)} logits and {len(targets)} targets for "
                f"{len(self.task_names)} tasks"
            )

        loss = 0.0
        logging_outputs = dict()
        for i, task_name in enumerate(self.task_names):
            criterion = self.criterions[task_name]
            assert (
                task_name in logits and task_name in targets
            ), f"task name {task_name} not found in logits or targets"
            task_logits = logits[task_name]
            task_targets = targets[task_name]
            task_loss, task_losses_to_log = criterion.compute_loss(
                logits=task_logits, targets=task_targets, sample=sample, net_output=net_output, model=model
            )
            task_loss *= self.task_loss_weights[i]
            sample_size = criterion.get_sample_size(sample, task_targets)

            logging_outputs[f"<{task_name}>_criterion_cls"] = criterion.__class__
            if len(task_losses_to_log) > 1:
                logging_outputs[f"{task_name}_loss"] = task_loss.item()
                for j, l in enumerate(task_losses_to_log):
                    logging_outputs[f"<{task_name}>_loss_{j}"] = l
            else:
                logging_outputs[f"<{task_name}>_loss"] = task_losses_to_log[0]
            logging_outputs[f"<{task_name}>_sample_size"] = sample_size

            task_logging_output = criterion.get_logging_outputs({}, task_logits, task_targets, sample)
            for log, value in task_logging_output.items():
                if log.startswith("_"):
                    log = log[1:]
                    logging_outputs[f"_<{task_name}>_{log}"] = value
                else:
                    logging_outputs[f"<{task_name}>_{log}"] = value

            # divide task loss by the sample size beforehand to handle different sample
            # sizes for multiple criterions
            loss += task_loss / logging_outputs[f"<{task_name}>_sample_size"]

        # manipulate sample_size to be 1 to avoid double-dividing gradients in optimizer later
        sample_size = 1

        if return_net_output:
            return loss, sample_size, logging_outputs, net_output
        else:
            return loss, sample_size, logging_outputs

    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]]) -> None:
        log_keys = logging_outputs[0].keys()

        grouped_log_keys = defaultdict(list)
        for lk in log_keys:
            group = re.search(r"\<.*\>", lk)
            offset = group.end() + 1
            group = group.group()[1:-1]
            key = lk[offset:]
            if lk.startswith("_"):
                key = "_" + key
            grouped_log_keys[group].append(key)

        total_loss = 0
        for group, log_keys in grouped_log_keys.items():
            criterion_cls = logging_outputs[0][f"<{group}>_criterion_cls"]
            logging_output = []
            for log in logging_outputs:
                log_dict = {}
                for log_key in set(log_keys) - {"criterion_cls"}:
                    if log_key.startswith("_") and f"_<{group}>{log_key}" in log:
                        log_dict[log_key] = log[f"_<{group}>{log_key}"]
                    elif f"<{group}>_{log_key}" in log:
                        log_dict[log_key] = log[f"<{group}>_{log_key}"]
                logging_output.append(log_dict)
            criterion_cls.reduce_metrics(logging_output, prefix=group)

            loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_output))
            sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_output))

            total_loss += loss_sum / (sample_size or 1) / math.log(2)

        metrics.log_scalar("loss", total_loss, 1, round=3)

    def post_validate(self, stats, agg, **kwargs):
        task_agg = {}
        for key in agg:
            for task_name in self.task_names:
                if key.startswith(task_name) or key[1:].startswith(task_name):
                    if task_name not in task_agg:
                        task_agg[task_name] = {}
                    task_agg[task_name][key] = agg[key]
                    break

        for task_name, task_agg in task_agg.items():
            if hasattr(self.criterions[task_name], "post_validate"):
                stats = self.criterions[task_name].post_validate(stats, task_agg, **kwargs)

        for key in list(stats.keys()):
            for task_name in self.task_names:
                if key.startswith(task_name):
                    stat_key = key[len(task_name) + 1 :]
                    if f"avg_{stat_key}" not in stats:
                        stats[f"avg_{stat_key}"] = []
                    stats[f"avg_{stat_key}"].append(stats[key])
                    break

        for key in list(stats.keys()):
            if key.startswith("avg_"):
                stats[key] = sum(stats[key]) / len(stats[key])
        return stats

    def eval(self):
        super().eval()
        for criterion in self.criterions.values():
            criterion.eval()
        return self
