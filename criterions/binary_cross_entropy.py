import math

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.loss import _Loss

from loggings import metrics, meters
from loggings.meters import safe_round
from criterions import register_criterion
import utils.utils as utils
import numpy as np

@register_criterion('binary_cross_entropy')
class BinaryCrossEntropy(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    @classmethod
    def build_criterion(cls, args):
        return cls(args)
    
    def forward(self, model, sample):
        net_output = model(**sample['net_input'])
        if isinstance(model, DistributedDataParallel):
            logits = model.module.get_outputs(
                net_output, 
                task=self.args.train_task,
                normalize=True
            )
            targets = model.module.get_targets(sample, net_output, self.args.train_task)
        else:
            logits = model.get_outputs(
                net_output, 
                task=self.args.train_task,
                normalize=True
            )
            targets = model.get_targets(sample, net_output, self.args.train_task)

        if self.args.train_task in ['finetune', 'scratch', 'sampled_finetune'] and self.args.pred_target == 'dx':
            loss = F.binary_cross_entropy(input=logits.view(-1), target=targets.view(-1), reduction='sum')

            sample_size = len(targets)
            logging_output = {
                'loss': loss.item(),
                'sample_size': sample_size
            }

            with torch.no_grad():
                probs = logits.view(-1).detach()
                targets = targets.view(-1).detach()

                logging_output["_y_true"] = targets.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

        else:
            loss = F.binary_cross_entropy(input=logits, target=targets, reduction='sum')

            sample_size = len(targets)
            logging_output = {
                'loss': loss.item(),
                'sample_size': sample_size
            }

            with torch.no_grad():
                probs = logits
                outputs = (probs > 0.5)

                if probs.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    count = float(probs.numel())
                    corr = (outputs == targets).sum().item()

                logging_output['correct'] = corr
                logging_output['count'] = count

                logging_output["_y_true"] = targets.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(args, dataname, logging_outputs) -> None: 
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs)) # loss of all gpus

        sample_size = utils.item( # gpu x batch
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar( 
            f"{dataname}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true)

            if len(y_true) > 0:
                metrics.log_derived(
                    f"{dataname}auroc",
                    lambda meters: safe_round(
                        meters["_auc"].auroc, 3
                    )
                )
                metrics.log_derived(
                    f"{dataname}auprc",
                    lambda meters: safe_round(
                        meters["_auc"].auprc, 3
                    )
                )

        if not (args.train_task in ['finetune', 'scratch'] and args.pred_target == 'dx'):
            correct = sum(log.get("correct", 0) for log in logging_outputs)
            metrics.log_scalar("_correct", correct)

            total = sum(log.get("count", 0) for log in logging_outputs)
            metrics.log_scalar("_total", total)

            if total > 0:
                metrics.log_derived(
                    f"{dataname}accuracy",
                    lambda meters: safe_round(
                        meters["_correct"].sum / meters["_total"].sum, 5
                    )
                    if meters["_total"].sum > 0
                    else float("nan")
                )