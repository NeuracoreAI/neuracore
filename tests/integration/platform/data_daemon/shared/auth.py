"""Client authentication helper for integration tests.

Single login primitive for the suite: guarded on the in-process auth state,
so repeat calls within a test are free. The parent process logs out between
tests (see the autouse ``login_parent_process`` fixture), so each test
performs exactly one real login.
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

    The parent pytest process is covered by an autouse conftest fixture that
    logs in before each test and logs out after it. Pool workers on macOS
    use the ``spawn`` start method and get a fresh Auth
    singleton instead of the parent's in-process access token.
    """

    if get_auth().is_authenticated:
        return
    with Timer(MAX_TIME_TO_START_S, label="nc.login", always_log=True):
        nc.login()
