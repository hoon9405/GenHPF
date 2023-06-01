import math

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.loss import _Loss

from loggings import metrics
from loggings.meters import safe_round
from criterions import register_criterion
import utils.utils as utils

@register_criterion('w2v')
class Wave2Vec2(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_weights = [self.args.perp_weight, self.args.reg_weight]
        self.max_loss_weights = [0.1, 5]
        self.min_loss_weights = [0.005, 0.25]
        self.log_interval = 25
        self.epoch_save_interval = 50
        self.epoch_validate_from = 10
        self.no_validation = True
    
    @classmethod
    def build_criterion(cls, args):
        return cls(args)
    
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        if isinstance(model, DistributedDataParallel):
            logits = model.module.get_outputs(
                net_output, 
                task=self.args.train_task
            )
            targets = model.module.get_targets(sample, net_output, self.args.train_task)
            extra_losses = model.module.get_extra_losses(net_output)
        else:
            logits = model.get_outputs(
                net_output, 
                task=self.args.train_task
            )
            targets = model.get_targets(sample, net_output, self.args.train_task)
            extra_losses = model.get_extra_losses(net_output)
        
        losses = []

        reduction = "none" if not reduce else "sum"

        loss = F.cross_entropy(logits, targets, reduction=reduction)

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = targets.numel() 
        losses.append(loss.detach().clone())

        if torch.is_tensor(extra_losses):
            extra_losses = [extra_losses]
        for p, coef in zip(extra_losses, self.loss_weights):
            if coef == 0:
                losses.append(torch.tensor(0))
            elif p is not None:
                p = coef * p.float() * sample_size
                loss += p
                losses.append(p)

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "ntokens": sample_size,
            "sample_size": sample_size
        }

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

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

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(args, dataname, logging_outputs) -> None:
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            f"{dataname}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"{dataname}correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f"{dataname}total", total)

        if total > 0:
            metrics.log_derived(
                f"{dataname}accuracy",
                lambda meters: safe_round(
                    meters[f"{dataname}correct"].sum / meters[f"{dataname}total"].sum, 5
                )
                if meters[f"{dataname}total"].sum > 0
                else float("nan")
            )
        
        builtin_keys = {
            "loss",
            "ntokens",
            "sample_size",
            "correct",
            "count"
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k,0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        k, val / (sample_size or 1) / math.log(2), sample_size, round = 3
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round = 3)
    

