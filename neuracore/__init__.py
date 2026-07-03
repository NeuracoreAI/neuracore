"""Init."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .api.core import *  # noqa: F403
from .api.datasets import *  # noqa: F403
from .api.endpoints import *  # noqa: F403
from .api.logging import *  # noqa: F403
from .api.training import *  # noqa: F403
from .core.exceptions import *  # noqa: F403

try:
    # Version lives in pyproject.toml; read it from installed metadata so it
    # never drifts.
    __version__ = _pkg_version("neuracore")
except PackageNotFoundError:
    # Uninstalled source tree (bare checkout): keep a sentinel rather than
    # crash on import.
    __version__ = "0.0.0+unknown"
