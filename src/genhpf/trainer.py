import logging
import time
from itertools import chain
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
from omegaconf import OmegaConf
from torch.optim import Adam

from genhpf.configs import Config
from genhpf.datasets import BaseDataset
from genhpf.loggings import metrics
from genhpf.utils import checkpoint_utils, distributed_utils, utils
from genhpf.utils.file_io import PathManager

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, cfg: Config, model, criterion):
        self.cfg = cfg
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

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
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
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
                self._wrapped_criterion = distributed_utils.DistributedModel(
                    self.cfg.distributed_training,
                    self._criterion,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = distributed_utils.DistributedModel(
                    self.cfg.distributed_training,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def _build_optimizer(self):
        params = list(
            filter(lambda p: p.requires_grad, chain(self.model.parameters(), self.criterion.parameters()))
        )
        self._optimizer = Adam(
            params,
            lr=self.cfg.optimization.lr,
            betas=self.cfg.optimization.adam_betas,
            eps=self.cfg.optimization.adam_eps,
            weight_decay=self.cfg.optimization.weight_decay,
        )

    def state_dict(self):
        pretraining_parameter_names = []
        if hasattr(self.model, "get_pretraining_parameter_names"):
            pretraining_parameter_names = self.model.get_pretraining_parameter_names()
        model_state_dict = {
            k: v for k, v in self.model.state_dict().items() if k not in pretraining_parameter_names
        }

        state_dict = {
            "cfg": (
                OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True)
                if OmegaConf.is_config(self.cfg)
                else self.cfg
            ),
            "model": model_state_dict,
        }

        return state_dict

    def save_checkpoint(self, filename):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")

        state_dict = utils.move_to_cpu(self.state_dict())
        if self.is_data_parallel_master:
            checkpoint_utils.torch_persistent_save(state_dict, filename, async_write=False)
        logger.info(f"Finished saving checkpoint to {filename}")

    def load_checkpoint(self, filename) -> None:
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint and broadcast it to other ranks.
        """
        logger.info(f"Loading checkpoint from {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(filename)
        if bexists:
            if self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(filename)
            else:
                state = None

            if is_distributed:
                state = distributed_utils.broadcast_object(
                    state, src_rank=0, group=self.data_parallel_process_group, dist_device=self.device
                )

            # load model parameters
            try:
                self.model.load_state_dict(state["model"], strict=True)
                # save memory for later steps
                del state["model"]
            except Exception:
                raise Exception(
                    f"Cannot load model parameters from checkpoint {filename}; "
                    "please ensure that the architectures match."
                )
        else:
            logger.info(f"No existing checkpoint found {filename}")

    def get_train_iterator(
        self,
        dataset: BaseDataset,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
        """Return an DataLoader instance for training."""
        batch_sampler = (
            torch.utils.data.DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
        )
        batch_iterator = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True if not dist.is_initialized() else False,
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=dataset.collator,
            sampler=batch_sampler,
        )
        return batch_iterator, batch_sampler

    def get_valid_iterator(
        self,
        dataset: BaseDataset,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
        """Return an DataLoader instance for validation."""
        batch_sampler = (
            torch.utils.data.DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        )
        batch_iterator = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=dataset.collator,
            sampler=batch_sampler,
        )
        return batch_iterator, batch_sampler

    @metrics.aggregate("train")
    def train_step(self, sample):
        """Do forward, backward and optimization steps."""

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        logging_outputs = []
        sample = utils.prepare_sample(sample)

        loss, sample_size, logging_output = self.criterion(self.model, sample)
        if loss.item() > 0:
            loss.backward()  # backward
            self.optimizer.step()

        logging_outputs.append(logging_output)

        # emptying the CUDA cache after the first step can
        # reduce the chance of OOM
        if self.cuda and self.get_num_updates() == 0:
            torch.cuda.empty_cache()

        sample_size = float(sample_size)

        if self._sync_stats():
            train_time = self._local_cumulative_training_time()

            logging_outputs, (sample_size, total_train_time) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, train_time
            )

            self._cumulative_training_time = total_train_time / self.data_parallel_world_size

        logging_output = None
        self.set_num_updates(self.get_num_updates() + 1)

        if self.cuda and self.cuda_env is not None:
            gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
            gb_free = self.cuda_env.total_memory_in_GB - gb_used
            metrics.log_scalar("gb_free", gb_free, priority=1500, round=1, weight=0)

        # extract private logs (usually only for valid steps) before logging
        logging_outputs = list(
            map(lambda x: {key: x[key] for key in x if not key.startswith("_")}, logging_outputs)
        )
        # log stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)

        metrics.log_stop_time("train_wall")
        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample, subset=None):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = utils.prepare_sample(sample)
            _loss, sample_size, logging_output = self.criterion(self.model, sample)
            logging_outputs = [logging_output]

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ignore=False,
            )

        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)

        return _loss, sample_size, logging_outputs

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_num_updates(self):
        return self._num_updates

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        return time.time() - self._start_time + self._previous_training_time

    def _set_seed(self):
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def _aggregate_logging_outputs(
        self, logging_outputs: List[Dict[str, Any]], *extra_stats_to_sum, ignore=False
    ):
        return self._all_gather_list_sync(logging_outputs, *extra_stats_to_sum, ignore=ignore)

    def _all_gather_list_sync(self, logging_outputs: List[Dict[str, Any]], *extra_stats_to_sum, ignore=False):
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 32768),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _reduce_and_log_stats(self, logging_outputs, sample_size):
        metrics.log_speed("ups", 1.0, priority=100, round=2)

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.criterion.__class__.reduce_metrics(logging_outputs)
                del logging_outputs

            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            return logging_output
