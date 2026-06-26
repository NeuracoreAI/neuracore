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
    # The version lives in pyproject.toml [project].version (the single source
    # of truth). Read it back from the installed metadata so it never drifts.
    __version__ = _pkg_version("neuracore")
except PackageNotFoundError:
    # Running from a source tree that was never installed (e.g. a bare checkout
    # used as a sys.path entry). Keep a sentinel rather than crashing on import.
    __version__ = "0.0.0+unknown"
