"""Same-source recording tests for independent Rust producer processes."""

from __future__ import annotations

import multiprocessing
import queue
import time
import traceback
import uuid
from typing import Any

import pytest

import neuracore as nc
from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.db_constants import (
    TRACE_WRITE_WRITTEN,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    fetch_all_traces,
    fetch_recordings_for_source,
    wait_for_dataset_ready,
    wait_for_recording_index_for_source,
    wait_for_recordings_finalized,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
)

pytestmark = pytest.mark.skipif(
    not is_rust_daemon_enabled(),
    reason="Same-source multiprocess producers require NCD_RUST_DAEMON",
)

_PRODUCER_COUNT = 3
_SAMPLES_PER_PRODUCER = 8
_PROCESS_TIMEOUT_S = 180.0
_TRACE_WRITE_TIMEOUT_S = 30.0


def _wait_for_sse_recording(robot: object, timeout_s: float) -> str:
    """Wait until this producer's local state manager applies the start SSE."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        recording_id = robot.get_current_recording_id()  # type: ignore[attr-defined]
        if recording_id is not None:
            return str(recording_id)
        time.sleep(0.05)
    raise TimeoutError("Producer did not receive the recording start SSE")


def _same_source_producer(
    producer_index: int,
    robot_name: str,
    dataset_name: str,
    robot_connection_lock: Any,
    ready_barrier: Any,
    logged_barrier: Any,
    recording_started: Any,
    recording_stopped: Any,
    verification_complete: Any,
    capture_start_s: Any,
    result_queue: Any,
) -> None:
    """Connect one producer; only producer zero owns recording lifecycle."""
    multiprocessing.current_process().name = f"same-source-{producer_index}"
    robot = None
    try:
        ensure_login()
        nc.get_dataset(dataset_name)
        # Robot registration is setup rather than the multiprocess behavior under test.
        # THe lock is stop the robot from being registered by more
        # than on process. at the same time.
        with robot_connection_lock:
            robot = nc.connect_robot(robot_name, instance=0, overwrite=False)
        ready_barrier.wait(timeout=_PROCESS_TIMEOUT_S)

        recording_index: int | None = None
        if producer_index == 0:
            nc.start_recording(robot_name=robot_name, instance=0)
            recording_index = wait_for_recording_index_for_source(
                str(robot.id),
                int(robot.instance),
                timeout_s=60.0,
            )
            capture_start_s.value = time.time()
            recording_started.set()
            observed_recording_id = None
        else:
            if not recording_started.wait(timeout=_PROCESS_TIMEOUT_S):
                raise TimeoutError("Recording owner did not publish StartRecording")
            observed_recording_id = _wait_for_sse_recording(robot, timeout_s=60.0)

        timestamp_start = float(capture_start_s.value)
        gripper_name = f"producer_{producer_index}_gripper"
        for sample_index in range(_SAMPLES_PER_PRODUCER):
            nc.log_parallel_gripper_open_amount(
                name=gripper_name,
                value=(producer_index + 1) / 4.0 + sample_index / 1000.0,
                robot_name=robot_name,
                instance=0,
                timestamp=timestamp_start + sample_index * 0.01,
            )

        logged_barrier.wait(timeout=_PROCESS_TIMEOUT_S)

        if producer_index == 0:
            observed_recording_id = robot.get_cloud_recording_id(timeout_s=60.0)
            if observed_recording_id is None:
                raise TimeoutError("Daemon did not resolve the cloud recording id")
            try:
                nc.stop_recording(robot_name=robot_name, instance=0, wait=False)
            finally:
                recording_stopped.set()
        elif not recording_stopped.wait(timeout=_PROCESS_TIMEOUT_S):
            raise TimeoutError("Recording owner did not stop the recording")

        result_queue.put({
            "ok": True,
            "producer_index": producer_index,
            "robot_id": str(robot.id),
            "robot_instance": int(robot.instance),
            "recording_id": str(observed_recording_id),
            "recording_index": recording_index,
            "gripper_name": gripper_name,
        })

        # Keep every producer and its SSE consumer alive while the parent
        # verifies that the daemon finalized all three traces.
        verification_complete.wait(timeout=_PROCESS_TIMEOUT_S)
    except BaseException:  # noqa: BLE001 - propagate full child traceback
        result_queue.put({
            "ok": False,
            "producer_index": producer_index,
            "traceback": traceback.format_exc(),
        })
        recording_started.set()
        recording_stopped.set()
        for barrier in (ready_barrier, logged_barrier):
            try:
                barrier.abort()
            except Exception:  # noqa: BLE001
                pass
    finally:
        if robot is not None:
            robot.close()


def _collect_worker_results(result_queue: Any) -> list[dict[str, Any]]:
    """Collect one result from each producer with a shared overall deadline."""
    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + _PROCESS_TIMEOUT_S
    while len(results) < _PRODUCER_COUNT:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            results.append(result_queue.get(timeout=remaining))
        except queue.Empty:
            break
    return results


def _wait_for_three_written_traces(
    recording_index: int,
    expected_names: set[str],
) -> list[dict[str, Any]]:
    """Wait for the daemon to finalize the trace contributed by each process."""
    deadline = time.monotonic() + _TRACE_WRITE_TIMEOUT_S
    last_traces: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last_traces = fetch_all_traces(
            recording_index,
            columns=["data_type", "data_type_name", "write_status"],
        )
        matching = [
            trace
            for trace in last_traces
            if trace.get("data_type_name") in expected_names
        ]
        if {
            str(trace["data_type_name"]) for trace in matching
        } == expected_names and all(
            trace.get("write_status") == TRACE_WRITE_WRITTEN for trace in matching
        ):
            return matching
        time.sleep(0.1)

    raise AssertionError(
        "Daemon did not finalize one gripper trace per producer; "
        f"expected={sorted(expected_names)}, traces={last_traces}"
    )


def test_three_processes_log_to_one_sse_started_recording() -> None:
    """Only one process starts; all three log to that daemon-owned recording."""
    if not has_configured_org():
        pytest.skip(
            "Same-source multiprocess test requires NEURACORE_ORG_ID "
            "or a saved current organization."
        )

    run_id = uuid.uuid4().hex[:10]
    dataset_name = f"multi_producer_dataset_{run_id}"
    robot_name = f"multi_producer_robot_{run_id}"
    case = DataDaemonTestCase(
        duration_sec=1,
        parallel_contexts=_PRODUCER_COUNT,
        joint_count=0,
        requires_rust_daemon=True,
        storage_state_action=STORAGE_STATE_DELETE,
    )

    process_context = multiprocessing.get_context("spawn")
    robot_connection_lock = process_context.Lock()
    ready_barrier = process_context.Barrier(_PRODUCER_COUNT)
    logged_barrier = process_context.Barrier(_PRODUCER_COUNT)
    recording_started = process_context.Event()
    recording_stopped = process_context.Event()
    verification_complete = process_context.Event()
    capture_start_s = process_context.Value("d", 0.0)
    result_queue = process_context.Queue()
    processes: list[multiprocessing.Process] = []

    with scoped_storage_state(case, dataset_name=dataset_name):
        with online_daemon_running():
            assert_exactly_one_daemon_pid()
            nc.create_dataset(
                dataset_name,
                description="Three independent producers sharing one Rust recording",
            )

            for producer_index in range(_PRODUCER_COUNT):
                process = process_context.Process(
                    target=_same_source_producer,
                    args=(
                        producer_index,
                        robot_name,
                        dataset_name,
                        robot_connection_lock,
                        ready_barrier,
                        logged_barrier,
                        recording_started,
                        recording_stopped,
                        verification_complete,
                        capture_start_s,
                        result_queue,
                    ),
                )
                processes.append(process)
                process.start()

            try:
                results = _collect_worker_results(result_queue)
                assert len(results) == _PRODUCER_COUNT, (
                    f"Received {len(results)} of {_PRODUCER_COUNT} producer results: "
                    f"{results}"
                )
                failures = [result for result in results if not result["ok"]]
                assert not failures, "Producer process failure(s):\n" + "\n".join(
                    result["traceback"] for result in failures
                )

                results.sort(key=lambda result: int(result["producer_index"]))
                owner = results[0]
                robot_ids = {str(result["robot_id"]) for result in results}
                recording_ids = {str(result["recording_id"]) for result in results}
                expected_names = {str(result["gripper_name"]) for result in results}

                assert (
                    len(robot_ids) == 1
                ), f"Producers resolved different robots: {results}"
                assert (
                    len(recording_ids) == 1
                ), f"Producers observed different recording ids: {results}"
                assert owner["recording_index"] is not None

                rows = fetch_recordings_for_source(
                    str(owner["robot_id"]), int(owner["robot_instance"])
                )
                assert len(rows) == 1, (
                    "Only the lifecycle owner may create a recording; "
                    f"daemon rows={rows}"
                )

                matching_traces = _wait_for_three_written_traces(
                    int(owner["recording_index"]), expected_names
                )
                assert {trace["data_type_name"] for trace in matching_traces} == (
                    expected_names
                )

                wait_for_dataset_ready(
                    dataset_name,
                    expected_recording_count=1,
                    timeout_s=120.0,
                )
                wait_for_recordings_finalized(
                    dataset_name,
                    recording_ids,
                    timeout_s=120.0,
                )
            finally:
                verification_complete.set()
                deadline = time.monotonic() + 30.0
                for process in processes:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                hung = [process for process in processes if process.is_alive()]
                for process in hung:
                    process.terminate()
                    process.join(timeout=5.0)
                assert not hung, (
                    "Producer process(es) did not exit: "
                    f"{[(process.name, process.pid) for process in hung]}"
                )
                bad_exits = [
                    (process.name, process.exitcode)
                    for process in processes
                    if process.exitcode != 0
                ]
                assert not bad_exits, f"Producer process exit failures: {bad_exits}"
