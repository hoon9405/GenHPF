from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

from genhpf.configs.constants import LOG_FORMAT_CHOICES


@dataclass
class BaseConfig:
    """base configuration class"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(self, attribute_name: str, meta: str, default: Optional[Any] = None) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith("${"):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif getattr(self, attribute_name) != self.__dataclass_fields__[attribute_name].default:
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")


@dataclass
class CommonConfig(BaseConfig):
    debug: bool = field(default=False, metadata={"help": "enable debug mode"})
    no_progress_bar: bool = field(default=False, metadata={"help": "disable progress bar"})
    log_interval: int = field(default=100, metadata={"help": "log progress every N batches"})
    log_format: Optional[LOG_FORMAT_CHOICES] = field(default=None, metadata={"help": "log format to use"})
    log_file: Optional[str] = field(default=None, metadata={"help": "log file to copy metrics to."})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Weights and Biases project name to use for logging"}
    )
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "Weights and Biases entity(team) name to use for logging"}
    )
    seed: int = field(default=42, metadata={"help": "random seed"})
    all_gather_list_size: int = field(
        default=32768,
        metadata={"help": "number of bytes reserved for gathering stats from workers"},
    )


@dataclass
class DistributedTrainingConfig(BaseConfig):
    distributed_world_size: int = field(default=1, metadata={"help": "total number of GPUs across all nodes"})
    distributed_rank: Optional[int] = field(default=0, metadata={"help": "rank of the current worker"})
    distributed_backend: str = field(default="nccl", metadata={"help": "distributed backend"})
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={"help": "typically tcp://hostname:port that will be used to " "init distributed training"},
    )
    distributed_port: int = field(default=12355, metadata={"help": "port number for distributed training"})
    device_id: int = field(default=0, metadata={"help": "which GPU to use"})
    bucket_cap_mb: int = field(default=25, metadata={"help": "bucket size for reduction"})
    find_unused_parameters: bool = field(
        default=False, metadata={"help": "disable unused parameter detection when using distributed training"}
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as " "batchnorm population statistics"
        },
    )


@dataclass
class DatasetConfig(BaseConfig):
    data_format: str = field(
        default="genhpf", metadata={"help": "data format. supported formats: genhpf, meds"}
    )
    data: str = field(default=MISSING, metadata={"help": "path to the data directory"})
    label: bool = field(default=False, metadata={"help": "whether to load labels from the dataset"})
    vocab_size: int = field(default=MISSING, metadata={"help": "size of the vocabulary"})
    pad_token_id: int = field(default=0, metadata={"help": "pad token id"})
    sep_token_id: int = field(default=102, metadata={"help": "sep token id"})
    dummy_token_id: int = field(default=101, metadata={"help": "dummy token id"})
    ignore_index: int = field(
        default=-100,
        metadata={
            "help": "specifies a target value that is ignored and does not contribute to "
            "the input gradient. only applied to cross-entropy loss"
        },
    )
    apply_mask: bool = field(default=False, metadata={"help": "whether to apply masking to the input tokens"})
    mask_prob: float = field(default=0.15, metadata={"help": "probability for masking tokens"})
    mask_unit: str = field(
        default="token", metadata={"help": "unit for masking. supported units: token, event"}
    )
    mask_token_id: int = field(default=103, metadata={"help": "mask token id"})
    num_workers: int = field(default=1, metadata={"help": "how many subprocesses to use for data loading"})
    batch_size: Optional[int] = field(default=None, metadata={"help": "number of examples in a batch"})
    train_subset: str = field(
        default="train", metadata={"help": "data subset name to use for training (e.g., train, valid, test)"}
    )
    valid_subset: Optional[str] = field(
        default="valid", metadata={"help": "comma separated list of data subset names to use for validation"}
    )
    test_subset: Optional[str] = field(
        default="test", metadata={"help": "comma separated list of data subset names to use for test"}
    )
    combine_train_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "whether to combine all training subsets into one dataset",
            "argparse_alias": "--combine-train",
        },
    )
    combine_valid_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "whether to combine all validation subsets into one dataset",
            "argparse_alias": "--combine-val",
        },
    )
    combine_test_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "whether to combine all test subsets into one dataset",
            "argparse_alias": "--combine-test",
        },
    )
    disable_validation: Optional[bool] = field(
        default=None, metadata={"help": "whether to disable validation during training"}
    )


@dataclass
class OptimizationConfig(BaseConfig):
    max_epoch: int = field(default=0, metadata={"help": "maximum number of epochs to train"})
    lr: float = field(default=1e-4, metadata={"help": "learning rate"})
    adam_betas: Any = field(default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"})
    adam_eps: float = field(default=1e-8, metadata={"help": "epsilon for Adam optimizer"})
    weight_decay: float = field(default=1e-8, metadata={"help": "weight decay for Adam optimizer"})


@dataclass
class CheckpointConfig(BaseConfig):
    save_dir: str = field(default="checkpoints", metadata={"help": "path to save checkpoints"})
    checkpoint_prefix: str = field(
        default="checkpoint", metadata={"help": "prefix to add to the checkpoint file name"}
    )
    checkpoint_suffix: str = field(default="", metadata={"help": "suffix to add to the checkpoint file name"})
    load_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "path to a checkpoint to load model weights from, if provided"}
    )
    no_save: bool = field(default=False, metadata={"help": "don't save checkpoints"})
    save_interval: int = field(default=1, metadata={"help": "save a checkpoint every N epochs"})
    keep_last_epochs: int = field(default=-1, metadata={"help": "keep last N epoch checkpoints"})
    no_last_checkpoints: bool = field(default=False, metadata={"help": "don't store last checkpoints"})
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=False,
        metadata={"help": 'select the largest metric value for saving "best" checkpoints'},
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )


@dataclass
class MEDSConfig(BaseConfig):
    output_predictions: bool = field(
        default=False,
        metadata={
            "help": "whether to output predictions. if turned on, `genhpf-test` will automatically "
            "output predictions of the test set specified by `dataset.test_subset` in the "
            "`meds.output_dir` directory."
        },
    )
    labels_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "a path to the label directory for MEDS dataset. this is required to store "
            "output predictions in the format expected by `meds-evaluation` when "
            "`meds.output_predictions` is turned on."
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "a path to the output directory to store output predictions. "
            "this is only used when `meds.output_predictions` is turned on."
        },
    )


@dataclass
class Config(BaseConfig):
    common: CommonConfig = field(default_factory=CommonConfig)
    distributed_training: DistributedTrainingConfig = field(default_factory=DistributedTrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    meds: MEDSConfig = field(default_factory=MEDSConfig)
    model: Any = MISSING
    criterion: Any = None
