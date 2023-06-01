import importlib
import os

MODEL_REGISTRY = {}

def build_model(args):
    model = None
    model_type = getattr(args, "model", None)

    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type]

    assert model is not None, (
        f"Could not infer model type from {model_type}. "
        f"Available models: "
        + str(MODEL_REGISTRY.keys())
        + " Requested model type: "
        + model_type
    )

    return model.build_model(args)

def register_model(name):
    """
    New model types can be added with the :func:`register_model`
    function decorator.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")

        MODEL_REGISTRY[name] = cls

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
import_models(models_dir, "models")