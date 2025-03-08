import argparse
import logging
import os
import pprint
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("genhpf.train")

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from genhpf import criterions, models
from genhpf.configs import Config
from genhpf.configs.initialize import add_defaults, hydra_init
from genhpf.datasets import load_dataset
from genhpf.loggings import meters, metrics, progress_bar
from genhpf.trainer import Trainer
from genhpf.utils import checkpoint_utils, distributed_utils, utils


def main(cfg: Config) -> None:
    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert cfg.dataset.batch_size is not None, "batch_size must be specified"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # print args
    logger.info(pprint.pformat(dict(cfg)))

    model = models.build_model(cfg.model)
    if cfg.checkpoint.load_checkpoint is not None:
        state_dict = torch.load(cfg.checkpoint.load_checkpoint, map_location="cpu")["model"]
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"loaded model from {cfg.checkpoint.load_checkpoint}")
    criterion = criterions.build_criterion(cfg.criterion)

    logger.info(model)
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"criterion: {criterion.__class__.__name__}")
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    datasets = {}
    train_subsets = cfg.dataset.train_subset.split(",")
    if len(train_subsets) > 1:
        assert (
            cfg.dataset.combine_train_subsets
        ), "train_subset contains multiple datasets, but combine_train_subsets is not set"
        datasets["train"] = [("combined-train", load_dataset(cfg.dataset.data, train_subsets, cfg))]
    else:
        datasets["train"] = [(train_subsets[0].strip(), load_dataset(cfg.dataset.data, train_subsets, cfg))]

    if not cfg.dataset.disable_validation and cfg.dataset.valid_subset is not None:
        valid_subsets = cfg.dataset.valid_subset.split(",")
        if cfg.dataset.combine_valid_subsets:
            datasets["valid"] = [("combined-valid", load_dataset(cfg.dataset.data, valid_subsets, cfg))]
        else:
            datasets["valid"] = [
                (subset.strip(), load_dataset(cfg.dataset.data, [subset], cfg)) for subset in valid_subsets
            ]
    if cfg.dataset.test_subset is not None:
        test_subsets = cfg.dataset.test_subset.split(",")
        if cfg.dataset.combine_test_subsets:
            datasets["test"] = [("combined-test", load_dataset(cfg.dataset.data, test_subsets, cfg))]
        else:
            datasets["test"] = [
                (subset.strip(), load_dataset(cfg.dataset.data, [subset], cfg)) for subset in test_subsets
            ]

    trainer = Trainer(cfg, model, criterion)

    logger.info(f"training on {cfg.distributed_training.distributed_world_size} devices (GPUs)")
    logger.info(f"batch size per device = {cfg.dataset.batch_size}")

    max_epoch = cfg.optimization.max_epoch

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    for i in range(1, max_epoch + 1):
        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, datasets, i)
        if should_stop:
            break
    train_meter.stop()
    logger.info(f"done training in {train_meter.sum:.1f} seconds")


def should_stop_early(cfg: Config, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                f"early stop since valid performance hasn't improved for " f"{cfg.checkpoint.patience} runs"
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: Config,
    trainer: Trainer,
    datasets,
    epoch: int,
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # initialize data iterator
    data_loader, batch_sampler = trainer.get_train_iterator(datasets["train"][0][1])
    if batch_sampler is not None:
        batch_sampler.set_epoch(epoch)

    itr = iter(data_loader)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch,
        default_log_format=("tqdm" if cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project if distributed_utils.is_master(cfg.distributed_training) else None
        ),
        wandb_entity=(
            cfg.common.wandb_entity if distributed_utils.is_master(cfg.distributed_training) else None
        ),
        wandb_run_name=os.environ.get("WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)),
    )
    progress.update_config(_flatten_config(cfg))

    logger.info(f"begin training epoch {epoch}")

    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, sample in enumerate(progress):
        with metrics.aggregate("train_inner"):
            log_output = trainer.train_step(sample)

        if log_output is not None:
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

    valid_losses, should_stop = validate_and_save(cfg, trainer, datasets, epoch)

    # log end-of-epoch stats
    logger.info(f"end of epoch {epoch} (average epoch stats below)")
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(
    cfg: Config,
    trainer: Trainer,
    datasets,
    epoch: int,
) -> Tuple[List[Optional[float]], bool]:
    should_stop = False
    if epoch >= cfg.optimization.max_epoch:
        should_stop = True
        logger.info(
            "Stopping training due to " f"num_epochs: {epoch} >= max_epochs: {cfg.optimization.max_epoch}"
        )

    do_validate = "valid" in datasets or "test" in datasets

    # validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, datasets, epoch)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch, valid_losses[0])
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return valid_losses, should_stop


def validate(
    cfg: Config,
    trainer: Trainer,
    datasets,
    epoch: int,
):
    """Evaluate the model on the validation set(s) and return the losses."""

    valid_subsets = datasets.get("valid", [])
    test_subsets = datasets.get("test", [])

    valid_losses = []
    for subset, dataset in valid_subsets + test_subsets:
        logger.info(f"begin validation on '{subset}' subset")

        # initialize data iterator
        data_loader, _ = trainer.get_valid_iterator(dataset)
        progress = progress_bar.progress_bar(
            data_loader,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            log_file=cfg.common.log_file,
            epoch=epoch,
            default_log_format=("tqdm" if cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            wandb_entity=(
                cfg.common.wandb_entity if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            wandb_run_name=os.environ.get("WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                trainer.valid_step(sample, subset=subset)

        stats = agg.get_smoothed_values()

        if hasattr(trainer.criterion, "post_validate"):
            stats = trainer.criterion.post_validate(
                stats=stats,
                agg=agg,
            )

        # log validation stats
        stats = get_valid_stats(cfg, trainer, subset, stats)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        if np.isnan(stats[cfg.checkpoint.best_checkpoint_metric]):
            logger.info(
                f"validation value for {cfg.checkpoint.best_checkpoint_metric} is NaN. "
                "Changed the best checkpoint metric to loss."
            )
            cfg.checkpoint.best_checkpoint_metric = "loss"
            cfg.checkpoint.maximize_best_checkpoint_metric = False

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])

    return valid_losses


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def get_valid_stats(cfg: Config, trainer: Trainer, subset: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()

    if not hasattr(get_valid_stats, "best"):
        get_valid_stats.best = dict()

    prev_best = getattr(get_valid_stats, "best").get(subset, stats[cfg.checkpoint.best_checkpoint_metric])
    best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
    get_valid_stats.best[subset] = best_function(stats[cfg.checkpoint.best_checkpoint_metric], prev_best)

    key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
    stats[key] = get_valid_stats.best[subset]

    return stats


def _flatten_config(cfg: Config):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


@hydra.main(config_path=os.path.join("..", "configs"), config_name="config")
def hydra_main(cfg: Config) -> None:
    add_defaults(cfg)

    with open_dict(cfg):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        cfg.job_logging_cfg = OmegaConf.to_container(HydraConfig.get().job_logging, resolve=True)

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.set_struct(cfg, True)

    distributed_utils.call_main(cfg, main)


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except Exception:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"
    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
