"""Behavioural correctness tests for daemon process startup.

Verifies that connecting a robot starts the daemon, that concurrent callers
always resolve to a single daemon process, the PID file is consistent, and no
duplicate runner processes are created. These tests force online mode so
startup never inherits an offline profile.
"""

import queue
import time
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
from tests.integration.platform.data_daemon.shared.test_case.child_process import (
    ChildProcess,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
    REMOTE_CONTROL_REQUEST_TIMEOUT_S,
)
from tests.integration.platform.data_daemon.shared.test_case.recording_control import (
    PEER_AWAIT_OPEN,
    ControlProcessSpec,
    OpenAck,
    peer_control_process,
    remote_control_post,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    cloud_resource_deleter,
)


def test_connect_robot_starts_the_daemon() -> None:
    """Verify a connect starts the daemon, and that it picks up an open recording.

    A producer that only connects and logs never calls ``start_recording``, so
    this launch is the only one a web-started recording gets. The backend's
    announcement is one-shot: a daemon that was not running when it fired hears
    of the recording only in the snapshot it is sent on subscribing, and
    without that it reports the source idle for the recording's whole life.
    Nothing starts a daemon ahead of either connect, so the test gets what a
    user gets, and the second one connects with the window already open.
    """
    if not has_configured_org():
        pytest.skip(
            "Daemon startup on connect requires NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    robot_name = f"startup_robot_{uuid.uuid4().hex[:10]}"
    dataset_name = f"testing_dataset_startup_{uuid.uuid4().hex[:6]}"

    with scoped_daemon_storage_env(), scoped_online_mode():
        with cloud_resource_deleter(dataset_name, [robot_name]):
            peer = ChildProcess("peer-startup-control")
            commands = peer.queue()
            acks = peer.queue()
            peer_started = False
            peer_retired = False
            recording_id = None
            try:
                stop_daemon_and_verify()
                nc.create_dataset(dataset_name)
                dataset_id = nc.get_dataset(dataset_name).id

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True
                ):
                    robot = nc.connect_robot(robot_name, overwrite=False)

                assert_exactly_one_daemon_pid()

                robot_id, instance = str(robot.id), int(robot.instance)
                stop_daemon_and_verify()

                pending = remote_control_post(
                    "/recording/start",
                    {
                        "robot_id": robot_id,
                        "instance": instance,
                        "dataset_id": dataset_id,
                        "start_time": time.time(),
                    },
                )
                recording_id = str(pending["id"])

                peer.start(
                    peer_control_process,
                    (
                        # The peer only ever waits here, so the fields the stop
                        # orders read are never used.
                        ControlProcessSpec(
                            robot_name=robot_name,
                            wait=True,
                            stop_sla_s=MAX_TIME_TO_START_S,
                            assert_deadline=False,
                        ),
                        peer.ready_event,
                        commands,
                        acks,
                        peer.result_queue,
                    ),
                )
                peer_started = True
                assert peer.await_ready(), (
                    "the peer never connected, so nothing launched a daemon: "
                    f"{peer.collect().failure}"
                )
                assert_exactly_one_daemon_pid()

                commands.put((PEER_AWAIT_OPEN, float(pending["start_time"])))
                failure = ""
                try:
                    opened = acks.get(
                        timeout=REMOTE_CONTROL_REQUEST_TIMEOUT_S + MAX_TIME_TO_START_S
                    )
                except queue.Empty:
                    opened = None
                    peer_retired = True
                    failure = peer.collect().failure
                assert isinstance(opened, OpenAck), (
                    f"recording {recording_id} was minted while no daemon was "
                    "running, and never reached the daemon the peer's connect "
                    f"launched:\n{failure or opened!r}"
                )
            finally:
                if recording_id is not None:
                    remote_control_post(
                        "/recording/stop",
                        {"recording_id": recording_id, "end_time": time.time()},
                    )
                if peer_started and not peer_retired:
                    commands.put(None)
                    peer.collect()
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
