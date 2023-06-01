import logging
import math
import time
from itertools import chain
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel

import utils.utils as utils
import utils.distributed_utils as distributed_utils
from loggings import metrics


logger = logging.getLogger(__name__)

class BaseTrainer(object):
    def __init__(
        self,
        args,
        model,
        criterion,
    ):
        self.args = args
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self._criterion = criterion
        self._model = model

        self._criterion = self._criterion.to(device=self.device)
        self._model = self._model.to(device=self.device)

        self._num_updates = 0

        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None

        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=self.data_parallel_process_group
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time('wall', priority=790, round=0)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    @property
    def data_parallel_world_size(self):
        if self.args.world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()
    
    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()
    
    @property
    def data_parallel_rank(self):
        if self.args.world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()
    
    @property
    def is_data_parallel_master(self):
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return self.data_parallel_world_size > 1
    
    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if utils.has_parameters(self._criterion) and self.use_distributed_wrapper:
                self._wrapped_criterion = DistributedDataParallel(
                    module=self._criterion.to(self.device),
                    device_ids=[self.args.device_id],
                    output_device=self.args.device_id,
                    broadcast_buffers=False,
                    bucket_cap_mb=25,
                    process_group=self.data_parallel_process_group,
                    find_unused_parameters=False,
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion
    
    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = DistributedDataParallel(
                    module=self._model.to(self.device),
                    device_ids=[self.args.device_id],
                    output_device=self.args.device_id,
                    broadcast_buffers=False,
                    bucket_cap_mb=25,
                    process_group=self.data_parallel_process_group,
                    find_unused_parameters=False,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
            if self.args.resume:
                print('loading optimizer from ', self.args.ckpt_load_path)
                state_dict = torch.load(self.args.ckpt_load_path, map_location='cpu')['optimizer']
                self._optimizer.load_state_dict(state_dict)
        return self._optimizer
    
    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters())
            )
        )

        if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
            self._optimizer = Adam(params, lr=self.args.lr, weight_decay=0.001)

    @metrics.aggregate('train')
    def train_step(self, sample):
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time('train_wall', priority=800, round=0)

        logging_outputs = []
        sample = utils.prepare_sample(sample)

        loss, sample_size, logging_output = self.criterion(self.model, sample)
        loss.backward() # backwward
        self.optimizer.step()
        
        logging_outputs.append(logging_output)
        
        if self.cuda and self.get_num_updates() == 0:
            torch.cuda.empty_cache()
    
        sample_size = float(sample_size)
        
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            
            logging_outputs, ( 
                sample_size, total_train_time
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, train_time
            )
            
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        logging_output = None
        self.set_num_updates(self.get_num_updates() + 1)

        if self.cuda and self.cuda_env is not None:
            gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
            gb_free = self.cuda_env.total_memory_in_GB - gb_used
            metrics.log_scalar(
                'gb_free', gb_free, priority=1500, round=1, weight=0
            )
        
        
        logging_outputs = list(map(
            lambda x: {key: x[key] for key in x}, logging_outputs)
        ) # -> 이거 왜있는거?
        
        
        logging_output = self._reduce_and_log_stats(
            self.args.train_src, logging_outputs, sample_size
        )
        
        metrics.log_stop_time('train_wall')
        return logging_output

    @metrics.aggregate('valid')
    def valid_step(self, sample, subset=None, dataname=None):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = utils.prepare_sample(sample)
            
            _loss, sample_size, logging_output = self.criterion(self.model, sample)

            logging_outputs = [logging_output]
        
        if self.data_parallel_world_size > 1 :
            logging_outputs, (sample_size, ) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ignore=False,
            )
        
        logging_output = self._reduce_and_log_stats(dataname, logging_outputs, sample_size)

        return _loss, sample_size, logging_outputs 

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_num_updates(self):
        return self._num_updates

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates
        metrics.log_scalar("num_updates", self._num_updates, weight = 0, priority = 200)


    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time
        
    def _local_cumulative_training_time(self):
        return time.time() - self._start_time + self._previous_training_time


    def _set_seed(self):
        seed = self.args.seed[0] + self.get_num_updates()
        utils.set_torch_seed(seed)
    
    def _sync_stats(self):
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False
    ):
        
        return self._all_gather_list_sync(
            logging_outputs, *extra_stats_to_sum, ignore=ignore
        )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore = False
    ):
       
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size = getattr(self, "all_gather_list_size", 32768),
                    group = self.data_parallel_process_group
                )
            )
        )
        
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _reduce_and_log_stats(self, dataname, logging_outputs, sample_size):
        metrics.log_speed('ups', 1.0, priority=100, round=2)
        
        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.criterion.__class__.reduce_metrics(self.args, dataname, logging_outputs)
                del logging_outputs
                
            logging_output = agg.get_smoothed_values()
            
            logging_output['sample_size'] = sample_size

            
            return logging_output
    
