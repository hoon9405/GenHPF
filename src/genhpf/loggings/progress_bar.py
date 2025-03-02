"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

import json
import logging
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Number
from typing import Optional

import torch

from .meters import AverageMeter, StopwatchMeter, TimeMeter

from genhpf.utils import distributed_utils as dist_utils

logger = logging.getLogger(__name__)


def progress_bar(
    iterator,
    log_format: Optional[str] = None,
    log_interval: int = 100,
    log_file: Optional[str] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    default_log_format: str = "tqdm",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    if log_format is None:
        log_format = default_log_format
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file)
        logger.addHandler(handler)

    if log_format == "tqdm" and not sys.stderr.isatty():
        log_format = "simple"

    if log_format == "json":
        bar = JsonProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == "none":
        bar = NoopProgressBar(iterator, epoch, prefix)
    elif log_format == "simple":
        bar = SimpleProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == "tqdm":
        bar = TqdmProgressBar(iterator, epoch, prefix)
    elif log_format == "csv":
        bar = CsvProgressBar(iterator, epoch, prefix, log_interval)
    else:
        raise ValueError("Unknown log format: {}".format(log_format))

    if wandb_project:
        bar = WandBProgressBarWrapper(bar, wandb_project, wandb_entity, run_name=wandb_run_name)

    return bar


def build_progress_bar(
    args,
    iterator,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    default: str = "tqdm",
    no_progress_bar: str = "none",
):
    """Legacy wrapper that takes an argparse.Namespace."""
    if getattr(args, "no_progress_bar", False):
        default = no_progress_bar
    tensorboard_logdir = None
    return progress_bar(
        iterator,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch,
        prefix=prefix,
        tensorboard_logdir=tensorboard_logdir,
        default_log_format=default,
    )


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


class BaseProgressBar(object):
    """Abstract class for progress bars."""

    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.n = getattr(iterable, "n", 0)
        self.epoch = epoch
        self.prefix = ""
        if epoch is not None:
            self.prefix += "epoch {:03d}".format(epoch)
        if prefix is not None:
            self.prefix += (" | " if self.prefix != "" else "") + prefix

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def update_config(self, config):
        """Log latest configuration."""
        pass

    def _str_commas(self, stats):
        return ", ".join(key + "=" + stats[key].strip() for key in stats.keys())

    def _str_pipes(self, stats):
        return " | ".join(key + " " + stats[key].strip() for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name


class JsonProgressBar(BaseProgressBar):
    """Log output in JSON format."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.n):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        step = step or self.i or 0

        if step > 0 and self.log_interval is not None and step % self.log_interval == 0:
            update = (
                self.epoch - 1 + (self.i + 1) / float(self.size)
                if self.epoch is not None
                else None
            )
            stats = self._format_stats(stats, epoch=self.epoch, update=update)
            with rename_logger(logger, tag):
                logger.info(json.dumps(stats))

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self.stats = stats
        if tag is not None:
            self.stats = OrderedDict(
                [(tag + "_" + k, v) for k, v in self.stats.items()]
            )
        stats = self._format_stats(self.stats, epoch=self.epoch)
        with rename_logger(logger, tag):
            logger.info(json.dumps(stats))

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:
            postfix["epoch"] = epoch
        if update is not None:
            postfix["update"] = round(update, 3)
        # Preprocess stats according to datatype
        for key in stats.keys():
            if not key.startswith("_"):
                postfix[key] = format_stat(stats[key])
        return postfix

class NoopProgressBar(BaseProgressBar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        pass


class SimpleProgressBar(BaseProgressBar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.n):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        step = step or self.i or 0
        if step > 0 and self.log_interval is not None and step % self.log_interval == 0:
            stats = self._format_stats(stats)
            postfix = self._str_commas(stats)
            with rename_logger(logger, tag):
                logger.info(
                    "{}:  {:5d} / {:d} {}".format(
                        self.prefix, self.i + 1, self.size, postfix
                    )
                )

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        with rename_logger(logger, tag):
            logger.info("{} | {}".format(self.prefix, postfix))


class TqdmProgressBar(BaseProgressBar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        from tqdm import tqdm

        self.tqdm = tqdm(
            iterable,
            self.prefix,
            leave=False,
            disable=(logger.getEffectiveLevel() > logging.INFO),
        )

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        with rename_logger(logger, tag):
            logger.info("{} | {}".format(self.prefix, postfix))

try:
    import csv
except ImportError:
    csv = None

class CsvProgressBar(BaseProgressBar):
    """Log to csv."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.n):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to csv."""
        self._log_to_csv(stats, tag, step)
        self._print_log(stats, tag, step)

    def print(self, stats, tag=None, step=None):
        """Log end-of-epoch stats."""
        self._log_to_csv(stats, tag, step)
        self._print_log(stats, tag, step)

    def _print_log(self, stats, tag=None, step=None):
        """Print intermediate stats."""
        step = step or self.i or 0
        
        if step > 0 and self.log_interval is not None and step % self.log_interval == 0:
            stats = self._format_stats(stats)
            postfix = self._str_commas(stats)
            with rename_logger(logger, tag):
                logger.info(
                    "{}:  {:5d} / {:d} {}".format(
                        self.prefix, self.i + 1, self.size, postfix
                    )
                )

    def _log_to_csv(self, stats, tag=None, step=None):
        if csv is None:
            return
        if dist_utils.get_data_parallel_world_size() > 1 and dist_utils.get_data_parallel_rank() > 0:
            return

        csv_logs = {}

        if step is None:
            csv_logs["step"] = stats["num_updates"] if "num_updates" in stats else None
        else:
            csv_logs["step"] = step

        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                csv_logs[key] = stats[key].val
            elif isinstance(stats[key], Number):
                csv_logs[key] = stats[key]

        fname = "log.csv" if tag is None else tag + ".csv"
        if not os.path.exists(fname):
            cols = ["step"]
            for key in csv_logs.keys() - {"step"}:
                cols.append(key)
            with open(fname, "w") as f:
                wr = csv.writer(f)
                wr.writerow(cols)
        
        with open(fname, "r") as f:
            rd = csv.reader(f)
            headers = next(iter(rd))

        appended_lth = 0
        for key in csv_logs.keys():
            if key not in headers:
                headers.append(key)
                appended_lth += 1
        
        if appended_lth > 0:
            lines = [headers]
            with open(fname, "r") as f:
                rd = csv.reader(f)
                # drop headers
                next(iter(rd))
                for line in rd:
                    for _ in range(appended_lth):
                        line.append("")
                    lines.append(line)
            with open(fname, "w") as f:
                wr = csv.writer(f)
                wr.writerows(lines)

        with open(fname, "a") as f:
            wr = csv.DictWriter(f, fieldnames=headers)
            wr.writerow(csv_logs)

try:
    import wandb
except ImportError:
    wandb = None

class WandBProgressBarWrapper(BaseProgressBar):
    """Log to Weights & Biases."""

    def __init__(self, wrapped_bar, wandb_project, wandb_entity, run_name=None):
        self.wrapped_bar = wrapped_bar
        if wandb is None:
            logger.warning("wandb not found, pip install wandb")
            return

        # reinit=False to ensure if wandb.init() is called multiple times
        # within one process it still references the same run
        wandb.init(project=wandb_project, entity=wandb_entity, reinit=False, name=run_name)

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to wandb."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        if wandb is not None:
            wandb.config.update(config)
        self.wrapped_bar.update_config(config)

    def _log_to_wandb(self, stats, tag=None, step=None):
        if wandb is None:
            return
        if step is None:
            step = stats["num_updates"] if "num_updates" in stats else None

        prefix = "" if tag is None else tag + "/"

        wandb_logs = {}
        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                wandb_logs[prefix + key] = stats[key].val
            elif isinstance(stats[key], Number):
                wandb_logs[prefix + key] = stats[key]

        wandb.log(wandb_logs, step=step)