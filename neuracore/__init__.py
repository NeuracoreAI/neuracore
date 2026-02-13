"""Init."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .api.core import *  # noqa: F403
from .api.datasets import *  # noqa: F403
from .api.endpoints import *  # noqa: F403
from .api.logging import *  # noqa: F403
from .api.training import *  # noqa: F403
from .core.exceptions import *  # noqa: F403

try:
    __version__ = version("neuracore")
except PackageNotFoundError as exc:
    pyproject_toml = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_toml.exists():
        raise RuntimeError(
            "Could not determine neuracore version: package metadata is missing and "
            "pyproject.toml was not found."
        ) from exc

    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject_toml.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(
            "Could not determine neuracore version: `version` is missing in "
            "pyproject.toml."
        ) from exc

    __version__ = match.group(1)
