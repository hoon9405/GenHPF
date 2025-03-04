import importlib
import os

from .ehr import EHR

EHR_REGISTRY = {}

__all__ = "EHR"


def register_ehr(name):
    def register_ehr(cls):
        if name in EHR_REGISTRY:
            raise ValueError("Cannot register duplicate EHR ({})".format(name))
        EHR_REGISTRY[name] = cls

        return cls

    return register_ehr


def import_ehrs(ehrs_dir, namespace):
    for file in os.listdir(ehrs_dir):
        path = os.path.join(ehrs_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            ehrs_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + ehrs_name)


# automatically import any Python files in the ehrs/ directory
ehrs_dir = os.path.dirname(__file__)
import_ehrs(ehrs_dir, "ehrs")
