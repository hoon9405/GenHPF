import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import II

import genhpf.utils.utils as utils
from genhpf.criterions import BaseCriterion, register_criterion
from genhpf.criterions.criterion import CriterionConfig
from genhpf.loggings import meters, metrics
from genhpf.loggings.meters import safe_round


@dataclass
class CrossEntropyConfig(CriterionConfig):
    report_auc: bool = field(
        default=False,
        metadata={
            "help": "whether to report auc. note that this is only available in eval mode and "
            "can cause memory and performance issues if enabled."
        },
    )
    ignore_index: int = II("dataset.ignore_index")


@register_criterion("cross_entropy", dataclass=CrossEntropyConfig)
class CrossEntropy(BaseCriterion):
    def __init__(self, cfg: CrossEntropyConfig):
        super().__init__(cfg)

        if self.task_names is not None and len(self.task_names) > 1:
            raise ValueError(
                "cross_entropy only supports single task training. if you want to train multiple"
                " tasks, use multi_task_criterion instead."
            )

        self.report_auc = cfg.report_auc
        self.ignore_index = cfg.ignore_index

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, sample=None, net_output=None, model=None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute the loss given the logits and targets from the model."""
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1).long()

        if torch.all(targets == self.ignore_index):
            return logits.new_tensor(0.0), [0.0]

        loss = F.cross_entropy(logits, targets, reduction="sum", ignore_index=self.ignore_index)

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
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1).long()

            valid_indices = torch.where(targets != self.ignore_index)
            if len(valid_indices[0]) == 0:
                return {}

            logits = logits[valid_indices]
            targets = targets[valid_indices]

            preds = logits.argmax(dim=-1)
            count = targets.numel()
            corr = (preds == targets).sum().item()

            logging_output["correct"] = corr
            logging_output["count"] = count

            # report aucs only in eval mode
            if self.report_auc and not self.training:
                probs = torch.sigmoid(logits).view(-1)
                targets = F.one_hot(targets, logits.size(-1)).float().view(-1)

                logging_output["_y_true"] = targets.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

        return logging_output

    # def forward(self, model, sample):
    #     net_output = model(**sample['net_input'])
    #     if isinstance(model, DistributedDataParallel):
    #         logits = model.module.get_outputs(
    #                 net_output,
    #                 task=self.args.train_task,
    #                 normalize=False
    #             )
    #         targets = model.module.get_targets(sample, net_output, self.args.train_task)
    #     else:
    #         logits = model.get_outputs(
    #                 net_output,
    #                 task=self.args.train_task,
    #                 normalize=False
    #             )
    #         targets = model.get_targets(sample, net_output, self.args.train_task)

    #     loss_dict = {}
    #     logging_output = {}

    #     if self.args.train_task == 'pretrain' and self.args.pretrain_task in ['mlm', 'spanmlm']:
    #         B, S= targets['input_label'].shape
    #         for victim in self.args.mask_list:
    #             loss = F.cross_entropy(
    #                 logits[victim+'_ids'].view(B*S, -1),
    #                 targets[victim+'_label'].view(-1)
    #             )
    #             loss_dict[victim+'_loss'] = loss

    #             with torch.no_grad():
    #                 preds = torch.argmax(logits[victim+'_ids'], dim=-1).view(-1).detach().cpu()
    #                 target_label = targets[victim+'_label'].view(-1).detach().cpu()
    #                 mask_idcs = (target_label != -100) & (target_label != 0)
    #                 total = mask_idcs.sum()
    #                 correct = (preds[mask_idcs] == target_label[mask_idcs]).sum().float()

    #                 logging_output[victim+'_correct'] = correct
    #                 logging_output[victim+'_total'] = total

    #         loss = sum(loss_dict.values())
    #         sample_size = len(sample)
    #         logging_output['loss'] = loss.item()
    #         logging_output['sample_size'] = sample_size

    #     elif self.args.train_task in ['finetune', 'scratch']:

    #         sample_size = len(targets)
    #         loss = F.cross_entropy(
    #             logits, F.one_hot(
    #                 targets.long(),
    #                 self.multi_label_dict[self.args.pred_src][self.args.pred_target]
    #             ).float().to(logits.device),
    #             reduction=self.ce_reduction_mode
    #         )

    #         logging_output['loss'] = loss.item()
    #         logging_output['sample_size'] = sample_size

    #         with torch.no_grad():
    #             probs = torch.sigmoid(logits).view(-1).detach()
    #             targets = self.mlb.transform(np.expand_dims(targets.view(-1).cpu(), axis=1)).flatten()

    #             logging_output["_y_true"] = targets
    #             logging_output["_y_score"] = probs.cpu().numpy()

    #     return loss, sample_size, logging_output

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
