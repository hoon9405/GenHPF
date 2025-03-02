import importlib
import os

from hydra.core.config_store import ConfigStore

from .genhpf import GenHPF #noqa
from genhpf.configs.utils import merge_with_parent

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}

def build_model(cfg):
    model = None
    model_type = getattr(cfg, "_name", None)

    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type]
        # set defaults from dataclass
        dc = MODEL_DATACLASS_REGISTRY[model_type]
        cfg = merge_with_parent(dc(), cfg)

    assert model is not None, (
        f"Could not infer model type from {str(model_type)}. "
        + "Available models: "
        + str(MODEL_REGISTRY.keys())
        + " Requested model type: "
        + str(model_type)
    )

    model_instance = model.build_model(cfg)
    return model_instance

def register_model(name, dataclass=None):
    """
    New model types can be added with the :func:`register_model`
    function decorator.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, GenHPF):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend GenHPF")
        MODEL_REGISTRY[name] = cls
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="model", node=node, provider="genhpf")

        return cls

    return register_model_cls

def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "genhpf.models")