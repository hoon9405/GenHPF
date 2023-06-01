from builtins import NotImplemented
import os
import logging

import numpy as np
import torch
from collections import OrderedDict
from contextlib import contextmanager
import wandb
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Task:
    name: str
    num_classes: int
    property: str


def get_task(pred_task, src_data):
    final_acuity = 6
    imminent_discharge = 6
    diagnosis = 17

    return {
        'mortality': Task('mortality', 1, 'binary'),
        'long_term_mortality': Task('long_term_mortality', 1, 'binary'), 
        'los_3day': Task('los_3day', 1, 'binary'), 
        'los_7day': Task('los_7day', 1, 'binary'),
        'readmission': Task('readmission', 1, 'binary'),
        'final_acuity': Task('final_acuity', final_acuity, 'multi-class'), 
        'imminent_discharge': Task('imminent_discharge', imminent_discharge, 'multi-class'), 
        'diagnosis': Task('diagnosis', diagnosis, 'multi-label'), 
        'creatinine': Task('creatinine', 5, 'multi-class'), 
        'bilirubin': Task('bilirubin', 5, 'multi-class'), 
        'platelets': Task('platelets', 5, 'multi-class'),
        'wbc': Task('wbc', 3, 'multi-class'),
    }[pred_task]

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.
        
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

# legacy
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, 
                patience=7, 
                verbose=True, 
                delta=0, 
                compare='increase',
                metric='auprc'

                ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metric_min = 0
        self.delta = delta
        self.compare_score = self.increase if compare=='increase' else self.decrease
        self.metric = metric

    def __call__(self, target_metric):
        update_token=False
        score = target_metric

        if self.best_score is None:
            self.best_score = score

        if self.compare_score(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation {self.metric} {self.compare_score.__name__}d {self.target_metric_min:.6f} --> {target_metric:.6f})')
            self.target_metric_min = target_metric
            self.counter = 0
            update_token = True
        
        return update_token

    def increase(self, score):
        if score < self.best_score + self.delta:
           return True
        else:
           return False

    def decrease(self, score):
        if score > self.best_score + self.delta:
            return True
        else:
           return False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_save(model, path, n_epoch, optimizer):
    torch.save(
                {'model_state_dict': model.state_dict()  if (
                    isinstance(model, DataParallel) or
                    isinstance(model, DistributedDataParallel)
                    ) else model.state_dict(),
                'n_epoch': n_epoch,
                 'optimizer_state_dict': optimizer.state_dict()},
                path
    )
    print(f'model save at : {path}')


def log_from_dict(metric_dict, data_type, data_name, n_epoch):
    log_dict = {'epoch':n_epoch}
    for metric, values in metric_dict.items():
        log_dict[data_type+'/'+data_name+'_'+metric] = values
        print(data_type+'/'+data_name+'_'+metric +  ' : {:.3f}'.format(values))
    return log_dict
    
def pretraiend_load(model, args, load_path):
    load_path = load_path.replace('lr_0.0003', 'lr_0.0001')
    load_path = load_path.replace(f'seed_{args.seed}', 'seed_2020')
    target_path = load_path + '.pkl'
    
    if os.path.exists(target_path):
        print('finetuning on pretrained model training start ! load checkpoint from : ', target_path)
        load_dict = torch.load(target_path, map_location='cpu')
        model_state_dict = load_dict['model_state_dict']
        
        state_dict = {
                k: v for k,v in model_state_dict.items() if (
                    'emb2out' not in k
                )
            }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected or len(missing) > 1:
            logger.warn(
                'pretrained model has unexpected or missing keys.'
            )
        print('pretrained mlm paramters loaded !')
        print('pretrained load target path : ', target_path)
    else:
        print('pretrained load fail !')
        print('pretrained load target path : ', target_path)
    return model