"""Behavioural correctness tests for daemon process startup.

Verifies that connecting a robot starts the daemon, that concurrent callers
always resolve to a single daemon process, the PID file is consistent, and no
duplicate runner processes are created. These tests force online mode so
startup never inherits an offline profile.
"""

import uuid

import psutil
import pytest

import neuracore as nc
from neuracore.data_daemon.daemon_control import pid_is_running
from neuracore.data_daemon.helpers import get_daemon_pid_path
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    collect_daemon_pids_from_parallel_startup,
    get_runner_pids,
)
from tests.integration.platform.data_daemon.shared.profiles import scoped_online_mode
from tests.integration.platform.data_daemon.shared.runners import (
    online_daemon_running,
    scoped_daemon_storage_env,
    stop_daemon_and_verify,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
)


def test_connect_robot_starts_the_daemon() -> None:
    """Verify that connecting a robot starts the daemon within the connect budget.

    A producer that only connects and logs never calls ``start_recording``, so
    this launch is the only one a web-started recording gets. Nothing starts a
    daemon ahead of the connect, so the test gets what a user gets.
    """
    if not has_configured_org():
        pytest.skip(
            "Daemon startup on connect requires NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    with scoped_daemon_storage_env(), scoped_online_mode():
        try:
            stop_daemon_and_verify()

            with Timer(MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True):
                nc.connect_robot(
                    f"startup_robot_{uuid.uuid4().hex[:10]}", overwrite=False
                )

            assert_exactly_one_daemon_pid()
        finally:
            stop_daemon_and_verify()


def test_ensure_single_daemon_process() -> None:
    """Verify that only one daemon process is spawned under parallel startup.

    Simulates a burst of concurrent callers (one per logical CPU core) all
    racing to start the daemon at the same time. All callers must receive the
    same PID, the PID file must exist and agree with that PID, the process
    must actually be running, and there must be exactly one runner subprocess.
    """
    worker_count = psutil.cpu_count(logical=False) or 4
    with online_daemon_running():
        pids = collect_daemon_pids_from_parallel_startup(worker_count)

        assert len(pids) == worker_count
        assert len(set(pids)) == 1, (
            f"Expected all {worker_count} callers to receive the same daemon PID, "
            f"but got distinct PIDs: {sorted(set(pids))}"
        )
        pid = pids[0]

        pid_path = get_daemon_pid_path()
        assert pid_path.exists(), f"PID file missing after daemon startup: {pid_path}"
        assert pid_is_running(pid), f"Daemon PID {pid} is not running"
        assert pid_path.read_text(encoding="utf-8").strip() == str(pid), (
            f"PID file content does not match returned PID: "
            f"file={pid_path.read_text(encoding='utf-8').strip()!r} pid={pid}"
        )

        runner_pids = get_runner_pids()
        assert (
            pid in runner_pids
        ), f"Daemon PID {pid} not found among runner processes: {sorted(runner_pids)}"
        assert len(runner_pids) == 1, (
            f"Expected exactly one daemon runner process, "
            f"found {len(runner_pids)}: {sorted(runner_pids)}"
        )
