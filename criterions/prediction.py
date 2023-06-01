import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.loss import _Loss

from loggings import metrics, meters
from loggings.meters import safe_round
from criterions import register_criterion
import utils.utils as utils
import numpy as np

@register_criterion('prediction')
class PredCriterion(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
   
    @classmethod
    def build_criterion(cls, args):
        return cls(args)
    
    def forward(self, model, sample):
        net_output = model(**sample['net_input'])

        if isinstance(model, DistributedDataParallel):
            logits = model.module.get_outputs(
                net_output
            )
            targets = model.module.get_targets(sample)
        else:
            logits = model.get_outputs(
                net_output
            )
            targets = model.get_targets(sample)

        losses = {}
        sample_size = sample['net_input']['input_ids'].size(dim=0)
        logging_output = {
            'sample_size': sample_size
        }

        # Calculate losses, log metrics
        for task in self.args.pred_tasks: # TODO: weighted loss 적용하기
            pred = logits[task.name]
            target = targets[task.name]

            if task.property == 'binary':  
                pred = pred.view(-1)
                target = target.view(-1)
                loss = self.bce(input=pred, target=target) # input: (B, ), target: (B, )
                
            elif task.property == 'multi-label':
                loss = self.bce(input=pred, target=target) # input: (B, C), target: (B, C)
                
            elif task.property == 'multi-class':
                loss = self.ce(input=pred, target=target) # input: (B, C), target: (B, C)
                
            losses[task.name] = loss
            
            with torch.no_grad(): 
                logging_output[task.name] = {
                    "_y_score": torch.sigmoid(pred).detach().cpu().numpy(),
                    "_y_true": target.detach().cpu().numpy(),
                }
               
        total_loss = sum(list(losses.values()))

        logging_output['loss'] = total_loss.item()

        return total_loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(args, dataname, logging_outputs) -> None: 
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        sample_size = utils.item( # total batch size (batch x gpu count)
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar( # loss per gpu (loss of total batch / total batch size)
            f"{dataname}_loss", loss_sum / (sample_size) / math.log(2), sample_size, round = 3
        ) # prediction -> loss reduction mean (sample_size or 1)

        def _mortality_auroc(meters):
            return safe_round(meters["_mortality_auc"].auroc, 3)
        def _mortality_auprc(meters):
            return safe_round(meters["_mortality_auc"].auprc, 3)
        def _long_term_mortality_auroc(meters):
            return safe_round(meters["_long_term_mortality_auc"].auroc, 3)
        def _long_term_mortality_auprc(meters):
            return safe_round(meters["_long_term_mortality_auc"].auprc, 3)
        def _los_3day_auroc(meters):
            return safe_round(meters["_los_3day_auc"].auroc, 3)
        def _los_3day_auprc(meters):
            return safe_round(meters["_los_3day_auc"].auprc, 3)
        def _los_7day_auroc(meters):
            return safe_round(meters["_los_7day_auc"].auroc, 3)
        def _los_7day_auprc(meters):
            return safe_round(meters["_los_7day_auc"].auprc, 3)
        def _readmission_auroc(meters):
            return safe_round(meters["_readmission_auc"].auroc, 3)
        def _readmission_auprc(meters):
            return safe_round(meters["_readmission_auc"].auprc, 3)
        def _final_acuity_auroc(meters):
            return safe_round(meters["_final_acuity_auc"].auroc, 3)
        def _final_acuity_auprc(meters):
            return safe_round(meters["_final_acuity_auc"].auprc, 3)
        def _imminent_discharge_auroc(meters):
            return safe_round(meters["_imminent_discharge_auc"].auroc, 3)
        def _imminent_discharge_auprc(meters):
            return safe_round(meters["_imminent_discharge_auc"].auprc, 3)
        def _creatinine_auroc(meters):
            return safe_round(meters["_creatinine_auc"].auroc, 3)
        def _creatinine_auprc(meters):
            return safe_round(meters["_creatinine_auc"].auprc, 3)
        def _bilirubin_auroc(meters):
            return safe_round(meters["_bilirubin_auc"].auroc, 3)
        def _bilirubin_auprc(meters):
            return safe_round(meters["_bilirubin_auc"].auprc, 3)
        def _platelets_auroc(meters):
            return safe_round(meters["_platelets_auc"].auroc, 3)
        def _platelets_auprc(meters):
            return safe_round(meters["_platelets_auc"].auprc, 3)
        def _diagnosis_auroc(meters):
            return safe_round(meters["_diagnosis_auc"].auroc, 3)
        def _diagnosis_auprc(meters):
            return safe_round(meters["_diagnosis_auc"].auprc, 3)
        def _wbc_auroc(meters):
            return safe_round(meters["_wbc_auc"].auroc, 3)
        def _wbc_auprc(meters):
            return safe_round(meters["_wbc_auc"].auprc, 3)
         
        
        for task in args.pred_tasks:
            if task.name not in logging_outputs[0].keys():
                continue
            
            if "_y_score" in logging_outputs[0][task.name] and "_y_true" in logging_outputs[0][task.name]:
                y_score = np.concatenate([log[task.name].get("_y_score", 0) for log in logging_outputs])
                y_true = np.concatenate([log[task.name].get("_y_true", 0) for log in logging_outputs])
                
                if task.property in ['multi-class' ,'multi-label']:
                    mask_idcs = (y_true.sum(axis=1)!=0)
                    y_score = y_score[mask_idcs]
                    y_true = y_true[mask_idcs]
                    
                metrics.log_custom(meters.AUCMeter, f"_{task.name}_auc", y_score, y_true) # 저장 정확히 됨
                auroc_func = _mortality_auroc if task.name == "mortality" else _long_term_mortality_auroc if task.name == "long_term_mortality" else _los_3day_auroc if task.name == "los_3day" \
                    else _los_7day_auroc if task.name == "los_7day" else _readmission_auroc if task.name == "readmission" else _final_acuity_auroc if task.name == "final_acuity" else _imminent_discharge_auroc if task.name == "imminent_discharge" \
                    else _creatinine_auroc if task.name == "creatinine" else _bilirubin_auroc if task.name == "bilirubin" else _platelets_auroc if task.name == "platelets" else _wbc_auroc if task.name == "wbc" else _diagnosis_auroc
                
                auprc_func = _mortality_auprc if task.name == "mortality" else _long_term_mortality_auprc if task.name == "long_term_mortality" else _los_3day_auprc if task.name == "los_3day" \
                    else _los_7day_auprc if task.name == "los_7day" else _readmission_auprc if task.name == "readmission" else _final_acuity_auprc if task.name == "final_acuity" else _imminent_discharge_auprc if task.name == "imminent_discharge" \
                    else _creatinine_auprc if task.name == "creatinine" else _bilirubin_auprc if task.name == "bilirubin" else _platelets_auprc if task.name == "platelets" else  _wbc_auprc if task.name == "wbc" else _diagnosis_auprc

                if len(y_true) > 0:
                    metrics.log_derived( 
                        f"{dataname}_{task.name}_auroc", auroc_func
                    )
                    metrics.log_derived(
                        f"{dataname}_{task.name}_auprc", auprc_func
                    )
                    