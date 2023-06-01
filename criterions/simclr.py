import math

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.loss import _Loss

from loggings import metrics
from loggings.meters import safe_round
from criterions import register_criterion
import utils.utils as utils

@register_criterion('simclr')
class SimClr(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temp = 0.1
    
    @classmethod
    def build_criterion(cls, args):
        return cls(args)
    
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        if isinstance(model, DistributedDataParallel):
            logits = model.module.get_outputs(
                net_output
            )
        else:
            logits = model.get_outputs(
                net_output
            )

        logits = F.normalize(logits, dim=1) # normalize logits

        bsz = int(logits.shape[0] / 2)

        metrics.log_scalar("batch_size", bsz)
        metrics.log_scalar("emb_size", logits.shape[1])

        mask = 1 - torch.eye(bsz * 2, dtype=torch.uint8).to(logits.device)
        pos_ind = (
            torch.arange(bsz * 2).to(logits.device),
            2 * torch.arange(bsz, dtype=torch.long).unsqueeze(1).repeat(
                1, 2).view(-1, 1).squeeze().to(logits.device)
        )
        neg_mask = torch.ones((bsz * 2, bsz * 2 - 1), dtype=torch.uint8).to(logits.device)
        neg_mask[pos_ind] = 0
        
        # Cosine similarity computation
        sim_matrix = torch.matmul(logits, logits.T) # cosine similarity computation

        # Eliminate similarity between same view
        sim_matrix = torch.masked_select(sim_matrix, mask.bool()).view(sim_matrix.size(0), -1)

        positives = sim_matrix[pos_ind].unsqueeze(1)
        negatives = torch.masked_select(sim_matrix, neg_mask.bool()).view(sim_matrix.size(0), -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temp # divide by softmax temporature

        target = torch.zeros((logits.size(0), ), dtype=torch.long).to(logits.device)

        reduction = "none" if not reduce else "sum"

        loss = F.cross_entropy(logits, target, reduction=reduction)

        sample_size = logits.shape[0]

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "sample_size": sample_size
        }

        with torch.no_grad():
            if logits.numel() == 0: # in the case of no logits returned
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
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            f'{dataname}_loss', loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f'{dataname}correct', correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f'{dataname}total', total)

        if total > 0:
            metrics.log_derived( 
                f'{dataname}acc',
                lambda meters: safe_round(
                    meters[f'{dataname}correct'].sum / meters[f'{dataname}total'].sum, 5
                )
                if meters[f'{dataname}total'].sum > 0
                else float("nan")
            )