import logging
import os
import pprint
import sys
from itertools import chain

import polars as pl
import torch.distributed
import torch.utils.data

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("genhpf.test")

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from genhpf import criterions, models
from genhpf.configs import Config
from genhpf.configs.initialize import add_defaults, hydra_init
from genhpf.datasets import load_dataset
from genhpf.loggings import metrics, progress_bar
from genhpf.utils import distributed_utils, utils


def main(cfg: Config) -> None:
    assert (
        cfg.checkpoint.load_checkpoint is not None
    ), "Please specify the checkpoint to load with `checkpoint.load_checkpoint`"

    assert cfg.dataset.batch_size is not None, "batch_size must be specified"
    metrics.reset()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
    else:
        data_parallel_world_size = 1

    # print args
    logger.info(pprint.pformat(dict(cfg)))

    # load model
    model = models.build_model(cfg.model)
    logger.info(f"loading model from {cfg.checkpoint.load_checkpoint}")
    model_state_dict = torch.load(cfg.checkpoint.load_checkpoint, map_location="cpu")["model"]
    model.load_state_dict(model_state_dict, strict=True)
    logger.info(f"loaded model from {cfg.checkpoint.load_checkpoint}")

    logger.info(model)
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # Move model to GPU
    model.eval()
    if use_cuda:
        model.cuda()

    # build criterion
    criterion = criterions.build_criterion(cfg.criterion)
    criterion.eval()

    def _fp_convert_sample(sample):
        def apply_float(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype=torch.float)
            return t

        sample = utils.apply_to_sample(apply_float, sample)

        return sample

    assert cfg.dataset.test_subset is not None, "Please specify the test subset with `dataset.test_subset`"
    test_subsets = cfg.dataset.test_subset.split(",")

    if cfg.dataset.combine_test_subsets:
        datasets = [("combined-test", load_dataset(cfg.dataset.data, test_subsets, cfg))]
    else:
        datasets = [
            (subset.strip(), load_dataset(cfg.dataset.data, [subset], cfg)) for subset in test_subsets
        ]

    output_meds_predictions = False
    if cfg.dataset.data_format == "meds" and cfg.meds.output_predictions:
        if len(test_subsets) > 1:
            raise NotImplementedError(
                "MEDS dataset does not currently support multiple test subsets when `output_predictions` "
                "is enabled. Please specify only one test subset."
            )
        if cfg.dataset.combine_test_subsets:
            raise NotImplementedError(
                "MEDS dataset does not currently support `combine_test_subsets` when `output_predictions` "
                "is enabled. Please set `dataset.combine_test_subsets` to False."
            )
        if len(cfg.criterion.task_names) > 1:
            raise NotImplementedError(
                "MEDS dataset does not currently support multiple tasks when `output_predictions` "
                "is enabled. Please specify only one task."
            )
        if cfg.criterion.num_labels[0] > 1:
            raise NotImplementedError(
                "MEDS dataset currently only supports binary classification when `output_predictions` "
                "is enabled. Please specify only one label by setting `criterion.num_labels` to 1."
            )

        assert (
            cfg.meds.labels_dir is not None and cfg.meds.output_dir is not None
        ), "Please specify labels_dir and output_dir in the MEDS config to output predictions."
        assert data_parallel_world_size == 1, (
            "MEDS dataset does not currently support distributed testing when `output_predictions` "
            "is enabled. Please set `distributed_training.distributed_world_size` to 1."
        )

        output_meds_predictions = True

        labels = pl.read_parquet(os.path.join(cfg.meds.labels_dir, f"{test_subsets[0]}/*.parquet"))
        labels = labels.sort(by=["subject_id", "prediction_time"])
        labels = labels.with_columns(pl.col("subject_id").cum_count().over("subject_id").alias("suffix"))
        labels = labels.with_columns(
            pl.col("subject_id").cast(str) + "_" + pl.col("suffix").cast(str).alias("subject_id")
        )
        labels = labels.drop("suffix")
        labels = labels.select(["subject_id", "prediction_time", "boolean_value"])

        meds_pred_output = {
            "subject_id": [],
            "predicted_boolean_value": [],
            "predicted_boolean_probability": [],
        }

    for subset, dataset in datasets:
        logger.info(f"begin validation on '{subset}' subset")

        # initialize data iterator
        batch_sampler = (
            torch.utils.data.DistributedSampler(dataset, shuffle=False)
            if torch.distributed.is_initialized()
            else None
        )
        batch_iterator = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            collate_fn=dataset.collator,
            sampler=batch_sampler,
        )

        progress = progress_bar.progress_bar(
            batch_iterator,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            log_file=cfg.common.log_file,
            epoch=0,
            default_log_format=("tqdm" if cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            wandb_entity=(
                cfg.common.wandb_entity if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            wandb_run_name=os.environ.get("WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            with torch.no_grad():
                sample = utils.prepare_sample(sample)
                sample = _fp_convert_sample(sample)
                _loss, _sample_size, log_output, net_output = criterion(model, sample, return_net_output=True)
                log_outputs.append(log_output)
                if output_meds_predictions:
                    meds_pred_output["subject_id"].extend(sample["id"])
                    logits = model.get_logits(sample, net_output)
                    probs = torch.sigmoid(logits).view(-1).cpu()
                    meds_pred_output["predicted_boolean_probability"].extend(probs.tolist())
                    meds_pred_output["predicted_boolean_value"].extend(
                        (probs >= cfg.criterion.threshold).int().tolist()
                    )

        if output_meds_predictions:
            meds_pred_output = pl.DataFrame(meds_pred_output)
            meds_pred_output = meds_pred_output.join(labels, on="subject_id", how="left")
            meds_pred_output = meds_pred_output.select(
                [
                    pl.col("subject_id"),
                    pl.col("prediction_time"),
                    pl.col("boolean_value"),
                    pl.col("predicted_boolean_value"),
                    pl.col("predicted_boolean_probability"),
                ]
            )
            meds_pred_output = meds_pred_output.with_columns(
                pl.col("subject_id").map_elements(lambda x: x.split("_")[0], return_dtype=pl.String).cast(int)
            )
            meds_pred_output = (
                meds_pred_output.with_columns(pl.col("predicted_boolean_value").cast(bool))
            )
            if not os.path.exists(cfg.meds.output_dir):
                os.makedirs(cfg.meds.output_dir)
            meds_pred_output.write_parquet(os.path.join(cfg.meds.output_dir, f"{subset}.parquet"))

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate(new_root=True) as agg:
            criterion.__class__.reduce_metrics(log_outputs)
            del log_outputs
            log_outputs = agg.get_smoothed_values()

        if hasattr(criterion, "post_validate"):
            stats = criterion.post_validate(stats=log_outputs, agg=agg)

        progress.print(stats, tag=subset, step=None)


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
