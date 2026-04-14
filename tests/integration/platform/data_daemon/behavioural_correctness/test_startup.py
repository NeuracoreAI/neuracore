"""Behavioural correctness tests for daemon process startup.

Verifies that concurrent callers always resolve to a single daemon process,
the PID file is consistent, and no duplicate runner processes are created.
These tests force online mode so startup never inherits an offline profile.
"""

import psutil

import neuracore as nc
from neuracore.data_daemon.helpers import get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_os_control import pid_is_running
from tests.integration.platform.data_daemon.shared.process_control import (
    collect_daemon_pids_from_parallel_startup,
    get_runner_pids,
)
from tests.integration.platform.data_daemon.shared.profiles import scoped_online_mode


def test_ensure_single_daemon_process() -> None:
    """Verify that only one daemon process is spawned under parallel startup.

    Simulates a burst of concurrent callers (one per logical CPU core) all
    racing to start the daemon at the same time. All callers must receive the
    same PID, the PID file must exist and agree with that PID, the process
    must actually be running, and there must be exactly one runner subprocess.
    """
    nc.login()

    worker_count = psutil.cpu_count(logical=False) or 4
    with scoped_online_mode():
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
