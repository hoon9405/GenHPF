"""isort:skip_file"""

from importlib.metadata import PackageNotFoundError, version

__package_name__ = "genhpf"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:
    __version__ = "unknown"
