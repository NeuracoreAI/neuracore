"""Client authentication helper for integration tests.

Single login primitive for the suite: guarded on the in-process auth state,
so repeat calls are free and each process performs at most one real login.
"""

from __future__ import annotations

import neuracore as nc
from neuracore.core.auth import get_auth
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
)


def ensure_login() -> None:
    """Authenticate the current process from the environment if needed.

    The parent pytest process is covered by an autouse conftest fixture, so
    only the first test in a session performs a real login. Pool workers on
    macOS use the ``spawn`` start method and get a fresh Auth singleton
    instead of the parent's in-process access token — every authenticated
    call would fail with "Not authenticated. Please call login() first." —
    so they call this again themselves. Linux forks and inherits the token,
    making the worker call a no-op there.
    """

    if get_auth().is_authenticated:
        return
    with Timer(MAX_TIME_TO_START_S, label="nc.login", always_log=True):
        nc.login()
