"""Behavioural correctness tests for daemon process startup.

A starting daemon must neither duplicate itself nor lose what is already in
flight: callers racing to launch it, a recording announced while it was down, or
a producer already spooling video into it. Tests that need the backend force
online mode so startup never inherits an offline profile.
"""

import queue
import time
import uuid
from collections.abc import Callable

import psutil
import pytest

import neuracore as nc
from neuracore.data_daemon.daemon_control import pid_is_running
from neuracore.data_daemon.helpers import get_daemon_pid_path
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    assert_no_daemon_pids,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    assert_disk_frame_codes,
    assert_disk_recording_properties,
    assert_video_artifacts,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    collect_daemon_pids_from_parallel_startup,
    get_runner_pids,
    stop_daemon,
)
from tests.integration.platform.data_daemon.shared.profiles import (
    scoped_holdback_ms,
    scoped_online_mode,
)
from tests.integration.platform.data_daemon.shared.runners import (
    offline_daemon_running,
    online_daemon_running,
    scoped_daemon_storage_env,
    stop_daemon_and_verify,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    PerThread,
    ProcessPerCamera,
    Synchronous,
    case_ids,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.child_process import (
    ChildProcess,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_FLAT,
    MAX_TIME_TO_START_S,
    REMOTE_CONTROL_REQUEST_TIMEOUT_S,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextResult,
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.context_worker import (
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.recording_control import (
    PEER_AWAIT_OPEN,
    ControlProcessSpec,
    OpenAck,
    peer_control_process,
    remote_control_post,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

_START_RECORDING_CASES = DataDaemonTestBatch(
    cases=(
        PerThread(
            duration_sec=2,
            recording_count=2,
            joint_count=1,
            video_count=1,
            depth_count=1,
            image_width=64,
            image_height=64,
            video_fps=30,
            video_detail=DETAIL_FLAT,
        ),
        ProcessPerCamera(
            duration_sec=2,
            recording_count=2,
            joint_count=1,
            video_count=1,
            image_width=64,
            image_height=64,
            video_fps=30,
            video_detail=DETAIL_FLAT,
        ),
    ),
).as_cases()


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

    case = Synchronous()
    dataset_name = f"testing_dataset_startup_{uuid.uuid4().hex[:6]}"
    spec = build_context_specs(case, dataset_name=dataset_name)[0]
    robot_name = spec.robot_name

    with scoped_daemon_storage_env(), scoped_online_mode():
        stop_daemon_and_verify()
        with scoped_storage_state(case, spec):
            peer = ChildProcess("peer-startup-control")
            commands = peer.queue()
            acks = peer.queue()
            peer_started = False
            peer_retired = False
            recording_id = None
            try:
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


def _stop_daemon_under_stream() -> None:
    """Stop the daemon under a logging producer, so the first start launches it."""
    stop_daemon()
    assert_no_daemon_pids()


@pytest.mark.parametrize(
    "case", _START_RECORDING_CASES, ids=case_ids(_START_RECORDING_CASES)
)
def test_start_recording_starts_the_daemon_without_loss(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Verify a start starts the daemon, and that its recording keeps every frame.

    The producer is already logging when the daemon goes down, and it spools
    video whether or not a daemon runs, so the start that launches the daemon
    opens a window over chunks already open on disk. Frames in them that fall
    inside the window are owed like any other. The second recording starts
    against a settled daemon, so a loss in the first alone is startup's.
    """
    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    with (
        scoped_storage_state(case, specs),
        scoped_holdback_ms(50),
    ):
        try:
            with offline_daemon_running():
                results = run_case_contexts(
                    case,
                    specs=specs,
                    wait_for_traces=True,
                    before_first_recording=_stop_daemon_under_stream,
                )
                # One daemon: the start launched it, and nothing else did.
                assert_exactly_one_daemon_pid()

                assert_disk_recording_properties(results)
                assert_video_artifacts(results, case)
                assert_disk_frame_codes(results, case)
        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
                test_wall_s=test_wall_timer(),
            )
