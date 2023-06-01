import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.loss import _Loss

from loggings import metrics, meters
from loggings.meters import safe_round
from criterions import register_criterion
import utils.utils as utils
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

@register_criterion('cross_entropy')
class CrossEntropy(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.multi_label_dict = {
            'mimic3':{
                'dx':18,
                'im_disch':17,
                'fi_ac':18},
            'eicu':{
                'dx':18,
                'im_disch':8,
                'fi_ac':9},
             'mimic4':{
                'dx':18,
                'im_disch':17,
                'fi_ac':18},
        }
        self.mlb = None
        if self.args.train_task in ['finetune', 'scratch'] and self.args.pred_target in ['fi_ac', 'im_disch']:
            self.mlb = MultiLabelBinarizer()
            class_n = self.multi_label_dict[args.pred_src][args.pred_target]
            self.mlb.fit([[i] for i in range(class_n)])

        self.ce_reduction_mode = 'mean' if args.train_task in ['finetune', 'scratch'] else 'sum'
    
    @classmethod
    def build_criterion(cls, args):
        return cls(args)
    
    def forward(self, model, sample):
        net_output = model(**sample['net_input'])
        if isinstance(model, DistributedDataParallel):
            logits = model.module.get_outputs(
                    net_output, 
                    task=self.args.train_task, 
                    normalize=False
                )
            targets = model.module.get_targets(sample, net_output, self.args.train_task)
        else:
            logits = model.get_outputs(
                    net_output, 
                    task=self.args.train_task, 
                    normalize=False
                )
            targets = model.get_targets(sample, net_output, self.args.train_task)

        loss_dict = {}
        logging_output = {}
      
        if self.args.train_task == 'pretrain' and self.args.pretrain_task in ['mlm', 'spanmlm']:
            B, S= targets['input_label'].shape
            for victim in self.args.mask_list:  
                loss = F.cross_entropy(
                    logits[victim+'_ids'].view(B*S, -1), 
                    targets[victim+'_label'].view(-1)
                )
                loss_dict[victim+'_loss'] = loss
            
                with torch.no_grad():
                    preds = torch.argmax(logits[victim+'_ids'], dim=-1).view(-1).detach().cpu()
                    target_label = targets[victim+'_label'].view(-1).detach().cpu()
                    mask_idcs = (target_label != -100) & (target_label != 0)
                    total = mask_idcs.sum()
                    correct = (preds[mask_idcs] == target_label[mask_idcs]).sum().float()

                    logging_output[victim+'_correct'] = correct
                    logging_output[victim+'_total'] = total

            loss = sum(loss_dict.values()) 
            sample_size = len(sample)
            logging_output['loss'] = loss.item()
            logging_output['sample_size'] = sample_size
        
        elif self.args.train_task in ['finetune', 'scratch']:
            
            sample_size = len(targets)
            loss = F.cross_entropy(
                logits, F.one_hot(
                    targets.long(), 
                    self.multi_label_dict[self.args.pred_src][self.args.pred_target]
                ).float().to(logits.device),
                reduction=self.ce_reduction_mode 
            )

            logging_output['loss'] = loss.item()
            logging_output['sample_size'] = sample_size

            with torch.no_grad():
                probs = torch.sigmoid(logits).view(-1).detach()
                targets = self.mlb.transform(np.expand_dims(targets.view(-1).cpu(), axis=1)).flatten()
            
                logging_output["_y_true"] = targets
                logging_output["_y_score"] = probs.cpu().numpy()

        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(args, dataname, logging_outputs) -> None:
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs)) # loss of all gpus
        
        sample_size = utils.item( # gpu x batch
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar( # icustay 당 loss
            f"{dataname}_loss", loss_sum / (1) / math.log(2), sample_size, round = 3
        ) # prediction -> loss reduction mean (sample_size or 1)

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true)

            if len(y_true) > 0:
                metrics.log_derived(
                    f"{dataname}_auroc",
                    lambda meters: safe_round(
                        meters["_auc"].auroc, 3
                    )
                )
                metrics.log_derived(
                    f"{dataname}_auprc",
                    lambda meters: safe_round(
                        meters["_auc"].auprc, 3
                    )
                )

        if "correct" in logging_outputs[0]:
            correct = sum(log.get("correct", 0) for log in logging_outputs)
            metrics.log_scalar(f'{dataname}_correct', correct)

            total = sum(log.get("total", 0) for log in logging_outputs)
            metrics.log_scalar(f'{dataname}_total', total)

            metrics.log_derived(
                f'{dataname}_acc',
                lambda meters: safe_round(
                    meters[f'{dataname}_correct'].sum / meters[f'{dataname}_total'].sum, 5
                )
                if meters[f'{dataname}_total'].sum > 0
                else float("nan")
            )