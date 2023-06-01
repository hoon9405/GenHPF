import importlib
import os

CRITERION_REGISTRY = {}

def build_criterion(args):
    criterion = None
    criterion_type = getattr(args, "criterion", None)

    if criterion_type in CRITERION_REGISTRY:
        criterion = CRITERION_REGISTRY[criterion_type]

    assert criterion is not None, (
        f"Could not infer criterion type from {criterion_type}. "
        f"Available criterions: "
        + str(CRITERION_REGISTRY.keys())
        + " Requested criterion type: "
        + criterion_type
    )

    return criterion.build_criterion(args)

def register_criterion(name):
    """
    New criterion types can be added with the :func:`register_criterion`
    function decorator.

    Args:
        name (str): the name of the criterion
    """

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion ({name})")

        CRITERION_REGISTRY[name] = cls

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
import_criterions(criterions_dir, "criterions")