import argparse
import os
import sys
import logging
import logging.config
import random
import pprint
from typing import OrderedDict, Tuple
from  numbers import Number

import torch.distributed as dist
import utils.distributed_utils as dist_utils
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch
from loggings.meters import safe_round

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger("train")

from utils import utils
import utils.trainer_utils as trainer_utils
from loggings import metrics, meters
from loggings.meters import AverageMeter, StopwatchMeter, TimeMeter
from datasets.base_dataset import HierarchicalEHRDataset, FlattenEHRDataset, EHRDataset
import models
import criterions


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/nfs_edlab/ghhur/GenHPF/input12hr/')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_prefix', type=str, default='checkpoint')

    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=1)

    # dataset
    parser.add_argument('--pt_src', choices=[
        'mimiciii', 'eicu', 'mimiciv', 
        'mimiciii_eicu', 'mimiciii_mimiciv', 'eicu_mimiciv', 
        'mimiciii_eicu_mimiciv'], type=str, default='mimiciii'
    )
    
    parser.add_argument('--train_src', choices=[
        'mimiciii', 'eicu', 'mimiciv', 
        'mimiciii_eicu', 'mimiciii_mimiciv', 'eicu_mimiciv' ,
        'mimiciii_eicu_mimiciv'
        ], type=str, default='mimiciii'
        )
    
    parser.add_argument('--eval_src', choices=[
        'mimiciii', 'eicu', 'mimiciv', 
        'mimiciii_eicu', 'mimiciii_mimiciv', 'eicu_mimiciv', 
        'mimiciii_eicu_mimiciv'
        ], type=str, default=None
        )
    
    parser.add_argument('--pooled_eval', action='store_true', default=False)
    parser.add_argument('--pretrain_task',
        choices=['mlm', 'spanmlm', 'text_encoder_mlm', 'w2v', 'simclr', 'scratch', None], 
        type=str, 
        default=None, 
        help="the pre-training method applied"
        )
    
    parser.add_argument('--ratio', choices=['0', '10', '30', '50', '70', '100'], type=str, default='100')

    parser.add_argument(
        '--pred_tasks',
        default='mortality, long_term_mortality, los_3day, los_7day, readmission, final_acuity, imminent_discharge, diagnosis, creatinine, bilirubin, platelets, wbc',
        type=str,
        help=""
    )

    
    parser.add_argument('--emb_type', choices=['codebase', 'textbase'], type=str, default=None)
    parser.add_argument('--feature', choices=['select', 'all_features'], type=str, default=None)
    parser.add_argument('--structure', choices=['hi', 'fl'], type=str, default=None)
        
    # trainer
    parser.add_argument('--train_task', choices=['pretrain', 'finetune', 'scratch'], type=str, default=None)
    parser.add_argument('--seed', type=str, default='42')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--valid_subset', type=str, default="valid,test")
    parser.add_argument('--patience', type=int, default=10) 
    parser.add_argument('--criterion', type=str, default=None) 
    parser.add_argument(
        '--best_checkpoint_metric', 
        type=str, 
        default='avg_auroc', 
        choices=['loss', 'avg_auprc', 'acc', 'avg_auroc']
    )

    # For Self-supervised pretraining, this should be set to False
    parser.add_argument(
        '--maximize_best_checkpoint_metric', action='store_true', default=True
    )

    # pretrain
    parser.add_argument('--mlm_prob', type=float, default=0.15)
    parser.add_argument('--mask_list', type=str, default='input, type, dpe')
    
    # model
    parser.add_argument(
        '--model_run', choices=['GenHPF', 'DescEmb','Rajikomar', 'SAnD'], type=str, default='GenHPF',
        help='name of the model to be trained'
    )

    parser.add_argument(
        '--model', choices=['ehr_model', 'GenHPF_simclr', 'GenHPF_w2v', 'GenHPF_mlm'], type=str, default='ehr_model',
        help='name of the model to be trained'
    )
   
    parser.add_argument(
        '--pred_model', type=str, required=False, default=None,
        help='name of the encoder model in the --pred_model'
    )

    # model hyper-parameter configs
    parser.add_argument('--pred_dim', type=int, default=128)  
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--type_token', action='store_true', default=True)
    parser.add_argument('--dpe', action='store_true', default=True)
    parser.add_argument('--pos_enc', action='store_true', default=True)
    parser.add_argument('--time_embed', choices=['sinusoidal', 'alibi_time'], default=None)
    parser.add_argument("--alibi_const", type=int, default=3)
    parser.add_argument('--pred_pooling', choices=['cls', 'mean'], default='mean')
    parser.add_argument('--text_post_proj', action='store_true')
    parser.add_argument('--map_layers', type=int, default=1)
    parser.add_argument('--max_word_len', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=256)
    

    # for w2v
    parser.add_argument('--feature_grad_mult', type=float, default=0.1)
    parser.add_argument('--num_negatives', type=int, default=25)
    parser.add_argument('--codebook_negatives', type=int, default=0)
    parser.add_argument('--logit_temp', type=float, default=0.1)
    parser.add_argument('--latent_vars', type=int, default=320)
    parser.add_argument('--latent_groups', type=int, default=2)
    parser.add_argument('--latent_temp', type=Tuple[float, float, float], default=(2, 0.5, 0.999995))
    parser.add_argument('--final_dim', type=int, default=128)
    parser.add_argument('--dropout_input', type=float, default=0.1)
    parser.add_argument('--dropout_features', type=float, default=0.1)

    parser.add_argument('--mask_prob', type=float, default=0.65)
    parser.add_argument('--mask_length', type=int, default=1)
    parser.add_argument('--mask_selection', type=str, default='static')
    parser.add_argument('--mask_other', type=float, default=0)
    parser.add_argument('--no_mask_overlap', type=bool, default=False)
    parser.add_argument('--mask_min_space', type=int, default=0)
    parser.add_argument('--perp_weight', type=float, default=0.1)
    parser.add_argument('--reg_weight', type=int, default=5)
    parser.add_argument(
        '--log_interval', type=int, default=100,
    )

    # for ddp setting
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--port', type=str, default = '12355')
    
   
    # resume
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--resume', action='store_true')

    parser.add_argument(
        '--wandb', action='store_true', default=False,
    )
    parser.add_argument(
        '--wandb_project_name', type=str, default='Test'
    )
    parser.add_argument(
        '--wandb_entity', type=str, default=None
    )
    parser.add_argument(
        '--wandb_run_name', type=str, default=None
    )

    return parser

def pre_main(args):
    args.pred_tasks = [item.strip() for item in args.pred_tasks.split(',')]
    
    if args.valid_subset and len(args.valid_subset) > 0:
        args.valid_subset = args.valid_subset.replace(' ','').split(',')

    if args.mask_list and len(args.mask_list) > 0:
        args.mask_list = args.mask_list.replace(' ','').split(',')

    if args.seed and len(args.seed) > 0:
        args.seed = [int(s) for s in args.seed.replace(' ','').split(',')]

    if args.eval_src is None:
        args.eval_src = (
            args.train_src.split('_')
        )

    #set model config
    model_configs = {
        'GenHPF': {
            'emb_type':'textbase', 
            'structure':'hi', 
            'feature':'all_features', 
            'pred_model':'eventaggregator',
            'n_layers': 2,
            'time_embed': 'alibi_time', 
            'max_seq_len':256
        },
        'DescEmb': {
            'emb_type':'textbase', 
            'structure':'hi', 
            'feature':'select', 
            'pred_model':'eventaggregator',
            'n_layers': 2,
            'time_embed': 'sinusoidal',
            'max_seq_len':256
        },
        'Rajikomar': {
            'emb_type':'codebase', 
            'structure':'hi', 
            'feature':'all_features', 
            'pred_model':'eventaggregator',
            'n_layers': 2,
            'time_embed': 'sinusoidal', 
            'max_seq_len':256
        },
        'SAnD': {
            'emb_type':'codebase', 
            'structure':'fl', 
            'feature':'select', 
            'pred_model':'performer_pred',
            'n_layers': 4,
            'time_embed': 'sinusoidal',
            'max_seq_len':8192
        },
    }

    print(f'model_run ={args.model_run} and cofnig {model_configs[args.model_run]}')
    for k, v in model_configs[args.model_run].items():
        if args.__dict__[k] is None:
            args.__dict__[k]=v       

    if args.train_task in ['scratch', 'finetune']: 
        assert len(args.seed) == 1, "Scratch / Finetune should run on one seed"
    elif 'pretrain' in args.train_task:
        assert len(args.seed) == 5, "Pretrain should run on 5 seeds"
        assert len(args.valid_subset) == 0, "Pretrain should not have valid subset"

    #wandb setting
    if args.wandb_project_name is None:
        args.wandb_project_name = args.save_dir
    if args.wandb_run_name is None:
        args.wandb_run_name = args.save_prefix

    dist_utils.call_main(args, main)


def load_dataset(args, split, dataname, seed) -> None:
    if args.structure =='hi':
        ds = HierarchicalEHRDataset(
            args=args,
            data=dataname,
            emb_type=args.emb_type,
            feature=args.feature,
            input_path=args.input_path,
            split=split,
            structure=args.structure,
            train_task=args.train_task,
            ratio=args.ratio,
            pred_tasks=args.pred_tasks,
            seed=args.seed,
            mask_list=args.mask_list,
            pretrain_task=args.pretrain_task,
            mlm_prob=args.mlm_prob,
            max_word_len=args.max_word_len,
            max_seq_len=args.max_seq_len,
        )
    elif args.structure=='fl':
        ds = FlattenEHRDataset(
            args=args,
            data=dataname,
            emb_type=args.emb_type,
            feature=args.feature,
            input_path=args.input_path,
            split=split,
            structure=args.structure,
            train_task=args.train_task,
            ratio=args.ratio,
            pred_tasks=args.pred_tasks,
            seed=args.seed,
            mask_list=args.mask_list,
            pretrain_task=args.pretrain_task,
            mlm_prob=args.mlm_prob,
            max_word_len=args.max_word_len,
            max_seq_len=args.max_seq_len,
        )
    
    return ds

def load_dataloader(dataset, batch_size, seed, collator) -> None:
    sampler = None if not dist.is_initialized() else (
    DistributedSampler(dataset, seed=seed)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if not dist.is_initialized() else False,
        num_workers=0,
        collate_fn=collator,
        sampler=sampler,
    )  
 
    return dataloader, sampler


def model_load(path):
    state_dict = torch.load(path, map_location='cpu')
    model = state_dict['model']
    epoch = state_dict['epoch']
    optimizer = state_dict['optimizer']
    valid_losses = state_dict['valid_losses']
    try:
        num_runs = state_dict['patience']
    except KeyError:
        num_runs = 0

    return model, epoch, optimizer, num_runs, valid_losses



def main(args) -> None:
    np.random.seed(args.seed[0])
    random.seed(args.seed[0])
    utils.set_torch_seed(args.seed[0])
  
    set_struct(vars(args))

    args.pred_tasks = [trainer_utils.get_task(task, args.train_src) for task in args.pred_tasks]
    
    logger.info(pprint.pformat(args))

    # model build
    model = models.build_model(args)
    
    # ckpt load
    optimizer_load=None
    if args.load_checkpoint is not None:    
        optimizer_load = False if args.train_task=='finetune' else True

        model_loaded, args.start_epoch, optimizer, num_runs, valid_losses = model_load(args.load_checkpoint)
        model = model.from_pretrained(args, checkpoint=args.load_checkpoint)
        
        print("Model training from args checkpoint "+ args.load_checkpoint + " ...")
        utils.should_stop_early.num_runs = num_runs
        utils.should_stop_early.best = valid_losses[0]
    
    # criterion build
    criterion = criterions.build_criterion(args)
    
    logger.info(model)
    logger.info('model: {}'.format(model.__class__.__name__))
    logger.info(
        'num. shared model params: {:,} (num. trained: {:,})'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    
    #dataloader build
    if args.valid_subset and len(args.valid_subset) > 0:
        data_split = ['train'] + args.valid_subset
    else:
        print("Only in train mode (no evaluation)")
        data_split = ['train']
        
    datanames = [args.train_src] + args.eval_src*len(args.valid_subset)
        
    dataloaders = {k:{} for k in data_split}
    samplers = {k:{} for k in data_split}
        
    for split in data_split:
        if split=='train':
            datanames = [args.train_src]
        else:
            datanames = args.eval_src
                        
        for dataname in datanames:
            concat_list = [] 
            for data in dataname.split('_'):
                dataset= load_dataset(args, split, data, args.seed)
                concat_list.append(dataset)
                logger.info(f'{split}, {len(dataset)}')

            dataloader, sampler = load_dataloader(
                torch.utils.data.ConcatDataset(concat_list), 
                args.batch_size, 
                args.seed[0], 
                concat_list[0].collator
                )
            print(f'split : {split}, dataname : {dataname}, len of dataloder, {len(dataloader)}')
            dataloaders[split].update({dataname: dataloader})
            samplers[split].update({dataname: sampler})
            print(f'split : {split}, dataname : {dataname}, len : {len(dataloader)}')
    logger.info(f'{args.train_task}, {data_split}, {dataloaders}')
  
    # trainer build
    from trainers.base_trainer import BaseTrainer as Trainer

    trainer = Trainer(args, model, criterion)
    if optimizer_load:
        trainer.optimizer.load_state_dict(optimizer)
        
    logger.info(
        'training on {} devices (GPUs)'.format(
            args.world_size
        )
    )

    max_epoch = args.max_epoch
    lr = args.lr
    
    if args.wandb and dist_utils.is_master(args):
        if args.resume:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=args,
                name=args.wandb_run_name,
                resume="must"
            )
            print("Start patience: ", utils.should_stop_early.num_runs)
        else:
            args.sub_id = wandb.util.generate_id()
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=args,
                name=args.wandb_run_name,
                reinit=True
            )

    cum_data_count = 0

    train_meter = meters.StopwatchMeter()
    train_meter.start()

    #train
    if args.ratio !='0':
        for i in range(args.start_epoch, max_epoch + 1):
            cum_data_count += len(dataloaders['train'][args.train_src])
            validation = True if 'valid' in data_split else False
            valid_losses, should_stop = train(args, trainer, cum_data_count, epoch_itr=dataloaders, epoch=i, sampler=samplers, validation=validation)
            if should_stop:
                break
    
    if 'test' in args.valid_subset:
        for dataname in dataloaders['test'].keys():
            if args.ratio !='0':
                best_state_dict = torch.load(os.path.join(
                    args.save_dir, f'{args.save_prefix}_{dataname}_best.pt'), map_location='cpu'
                    )['model']
                if not isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                    trainer.model.load_state_dict(best_state_dict, strict=True)
                else:
                    trainer.model.module.load_state_dict(best_state_dict, strict=True)
                print(f"{dataname} loaded best checkpoint")
            
            valid_losses = validate(args, trainer, dataloaders, 'test', dataname)
            logger.info(f'test_losses = {valid_losses}')

        train_meter.stop()

    if args.wandb and dist_utils.is_master(args):
        wandb.finish(0)
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


@metrics.aggregate('train')
def train(args, trainer, cum_data_count, epoch_itr, epoch, sampler, validation):
    logger.info('begin training epoch {}'.format(epoch))

    if dist.is_initialized():
        sampler['train'][args.train_src].set_epoch(epoch)

    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info('Start iterating over samples')
    logger.info(f'len(epoch_itr[train]) ={len(epoch_itr["train"][args.train_src])}')
    
    for i, sample in enumerate(epoch_itr['train'][args.train_src]):
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(sample)
        
        if log_output is not None:
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
                progress_log(
                    args,
                    stats, 
                    tag='train_inner', 
                    step=num_updates, 
                    size=cum_data_count, 
                    log_wandb=args.wandb
                    )
                metrics.reset_meters('train_inner')
    logger.info('finish get training stats inner')
    
    stats = multi_task_avg(args, 
                           get_training_stats(metrics.get_smoothed_values('train')), 
                           args.train_src, 
                           metrics=['auroc', 'auprc']
                           )
    progress_print(args, stats, tag='train', log_wandb=args.wandb)


    if validation:
        logger.info('Evaluation start') 
        stop_list = []   
        for dataname in epoch_itr['valid'].keys():
            valid_losses, should_stop = validate_and_save(args, trainer, epoch_itr, epoch, 'valid', dataname)
            if should_stop:
                stop_list.append(dataname)
        
        if len(stop_list)>0:
            for stop_data in stop_list:
                del epoch_itr['valid'][stop_data]
                
        if len(epoch_itr['valid'])==0:
            should_stop = True
        else:
            should_stop = False

    else:
        valid_losses, should_stop = train_save(args, trainer, epoch, stats, args.train_src)
    logger.info('end of epoch {} (average epoch stats below)'.format(epoch))
        
    metrics.reset_meters('train')
    return valid_losses, should_stop


def train_save(args, trainer, epoch, stats, dataname):
    should_stop = False
    
    valid_losses = []
    valid_losses.append(stats[f'{dataname}_{args.best_checkpoint_metric}'])
    logger.info(f'train_losses = {valid_losses}')

    state_dict = {
        'model': trainer.model.state_dict() if not (
            isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
        ) else trainer.model.module.state_dict(),
        'epoch': epoch,
        'optimizer': trainer.optimizer.state_dict(),
        'valid_losses': valid_losses,
        'patience': 0,
    }

    if 'pretrain' in args.train_task:
        torch.save(
            state_dict,
            os.path.join(args.save_dir, f'{args.save_prefix}_last.pt')
        )
        logger.info(f'pretrain checkpoint save {args.save_dir}, {args.save_prefix}_last.pt')
    return valid_losses, should_stop

def validate_and_save(args, trainer, epoch_itr, epoch, valid_subset, dataname):
    num_updates = trainer.get_num_updates()
    should_stop = False
    training_time_hours = trainer.cumulative_training_time() / (60 * 60)

    valid_losses = validate(args, trainer, epoch_itr, valid_subset, dataname)
    logger.info(f'valid_losses = {valid_losses}')
    should_stop |= utils.should_stop_early(
        args.patience,
        dataname,
        valid_losses[0],
        descending=(not args.maximize_best_checkpoint_metric)
    )

    state_dict = {
        'model': trainer.model.state_dict() if not (
            isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
        ) else trainer.model.module.state_dict(),
        'epoch': epoch,
        'optimizer': trainer.optimizer.state_dict(),
        'valid_losses': valid_losses,
        'patience': utils.should_stop_early.num_runs[dataname] ,
    }

    if 'pretrain' in args.train_task:
        torch.save(
            state_dict,
            os.path.join(args.save_dir, f'{args.save_prefix}_{dataname}_last.pt')
        )
        
    else:
        torch.save(
            state_dict,
            os.path.join(args.save_dir, f'{args.save_prefix}_{dataname}_last.pt')
        )

        if utils.should_stop_early.best[dataname] == valid_losses[0]:
            torch.save(
                state_dict,
                os.path.join(args.save_dir, f'{args.save_prefix}_{dataname}_best.pt')
            )
    
    return valid_losses, should_stop

def get_training_stats(stats):
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats

def validate(args, trainer, epoch_itr, valid_subset, dataname):
    valid_losses = []
    
    logger.info('begin validation on "{}" subset, data {}'.format(valid_subset, dataname))
    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(epoch_itr[valid_subset][dataname]):
            trainer.valid_step(sample, subset=valid_subset, dataname=dataname)
            

    stats = multi_task_avg(args, agg.get_smoothed_values(), dataname, metrics=['auroc', 'auprc'])
    if valid_subset !='test':
        stats = get_valid_stats(args, trainer, valid_subset, dataname, stats)
    
    progress_print(args, 
                   stats, 
                   tag=valid_subset, 
                   prefix=f'valid on {valid_subset} subset, data {dataname}', 
                   log_wandb=args.wandb
                   )
    valid_losses.append(stats[f'{dataname}_{args.best_checkpoint_metric}'])
   
    return valid_losses

def get_valid_stats(args, trainer, subset, dataname, stats):
    stats['num_updates'] = trainer.get_num_updates()
    
    if not hasattr(get_valid_stats, f'best_{dataname}'):
        setattr(get_valid_stats, f'best_{dataname}', dict())
    
    prev_best = getattr(get_valid_stats, f'best_{dataname}').get(
        subset, stats[f'{dataname}_{args.best_checkpoint_metric}']
    )
    best_function = max if args.maximize_best_checkpoint_metric else min
    getattr(get_valid_stats, f'best_{dataname}')[subset] = best_function(
        stats[f'{dataname}_{args.best_checkpoint_metric}'], prev_best
    )

    key = f'best_{dataname}_{args.best_checkpoint_metric}'
    stats[key] = getattr(get_valid_stats, f'best_{dataname}')[subset]

    return stats



def format_stat(stat):
    if isinstance(stat, Number):
        stat = "{:g}".format(stat)
    elif isinstance(stat, AverageMeter):
        stat = "{:.3f}".format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = "{:g}".format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = "{:g}".format(round(stat.sum))
    elif torch.is_tensor(stat):
        stat = stat.tolist()
    return stat


def _format_stats(stats):
    postfix = OrderedDict(stats)
    # Preprocess stats according to datatype
    for key in postfix.keys():
        postfix[key] = str(format_stat(postfix[key]))
    return postfix

def _str_pipes(stats):
    return ' | '.join(key + ' ' + stats[key].strip() for key in stats.keys())

def _str_commas(stats):
    return ', '.join(key + '=' + stats[key].strip() for key in stats.keys())

def progress_log(args, stats, tag=None, step=0, size=1, prefix='', log_wandb=False):
    stats = _format_stats(stats)
    postfix = _str_commas(stats)
    with utils.rename_logger(logger, tag):
        logger.info(
            '{}: {:5d} / {:d} {}'.format(
                prefix, step, size, postfix
            )
        )
        if log_wandb and dist_utils.is_master(args):
            _stats = {}
            for key in stats:
                _stats[tag + '/' + key] = float(stats[key])
            wandb.log(_stats)

def progress_print(args, stats, tag=None, prefix='', log_wandb=False):
    postfix = _str_pipes(_format_stats(stats))
    with utils.rename_logger(logger, tag):
        logger.info('{} | {}'.format(prefix, postfix))
        if log_wandb and dist_utils.is_master(args):
            _stats = {}
            for key in stats:
                _stats[tag + '/' + key] = float(stats[key])
            wandb.log(_stats)


def multi_task_avg(args, stats, dataname, metrics):
    for metric in metrics:
        stats[f'{dataname}_avg_{metric}']=safe_round(np.mean(
                [stats[key] for key in stats if metric in key]), 3)
    return stats

def set_struct(cfg: dict):
    root = os.path.abspath(
        os.path.dirname(__file__)
    )
    from datetime import datetime
    now = datetime.now()
    from pytz import timezone
    
    now = now.astimezone(timezone('Asia/Seoul'))

    output_dir = os.path.join(
        root,
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )
    print('output_dir : ', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)

    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train.log'
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = "config"
    os.mkdir(cfg_dir)
    os.mkdir(cfg['save_dir'])

    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in cfg.items():
            print("{}: {}".format(k, v), file=f)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pre_main(args)
