import importlib
import os

from hydra.core.config_store import ConfigStore

from genhpf.criterions.criterion import BaseCriterion #noqa
from genhpf.configs.utils import merge_with_parent

CRITERION_REGISTRY = {}
CRITERION_DATACLASS_REGISTRY = {}

def build_criterion(cfg) -> BaseCriterion:
    criterion = None
    criterion_type = getattr(cfg, "_name", None)

    if criterion_type in CRITERION_REGISTRY:
        criterion = CRITERION_REGISTRY[criterion_type]
        # set defaults from dataclass
        dc = CRITERION_DATACLASS_REGISTRY[criterion_type]
        cfg = merge_with_parent(dc(), cfg)

    assert criterion is not None, (
        f"Could not infer criterion type from {str(criterion_type)}. "
        + "Available criterions: "
        + str(CRITERION_REGISTRY.keys())
        + " Requested criterion type: "
        + str(criterion_type)
    )

    return criterion.build_criterion(cfg)

def register_criterion(name, dataclass=None):
    """
    New criterion types can be added with the :func:`register_criterion`
    function decorator.

    Args:
        name (str): the name of the criterion
    """

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion ({name})")
        if not issubclass(cls, BaseCriterion):
            raise ValueError(
                f"Criterion ({name}: {cls.__name__}) must extend Base Criterion"
            )
        CRITERION_REGISTRY[name] = cls
        if dataclass is not None:
            CRITERION_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="criterion", node=node, provider="genhpf")

        return cls

    return register_criterion_cls

def import_criterions(criterions_dir, namespace):
    for file in os.listdir(criterions_dir):
        path = os.path.join(criterions_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            criterion_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + criterion_name)

# automatically import any Python files in the criterions/ directory
criterions_dir = os.path.dirname(__file__)
import_criterions(criterions_dir, "genhpf.criterions")