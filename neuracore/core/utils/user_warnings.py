"""Warnings that point at the user's code rather than inside Neuracore."""

import os
import sys
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + os.sep


def warn_user(message: str, category: type[Warning] = UserWarning) -> None:
    """Warn, reporting the first caller outside the neuracore package.

    The same warning can be reached through several Neuracore call paths
    (for example ``nc.create_dataset`` and ``Dataset.create``), so a fixed
    ``stacklevel`` would point inside the SDK on some of them.

    Args:
        message: The warning text.
        category: The warning class.
    """
    frame = sys._getframe(1)
    stacklevel = 2
    while frame.f_back is not None and frame.f_code.co_filename.startswith(
        _PACKAGE_DIR
    ):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)
