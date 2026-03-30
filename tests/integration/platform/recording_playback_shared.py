import logging
import multiprocessing
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    ensure_daemon_running,
    force_kill,
    pid_is_running,
    read_pid_from_file,
    shutdown,
    terminate_pid,
    wait_for_exit,
)

# Add examples dir to path.
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent.parent.parent / "examples"))
# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

MAX_TIME_TO_START_S = 20
LEAST_TIME_TO_STOP_S = 10
HIGH_TIME_TO_DATASET_READY_S = 500
MAX_TIME_TO_LOG_S = 0.5

TEST_PROFILE_PATHS: set[Path] = set()

MATRIX_SESSION_RUNS: list[dict[str, object]] = []

logger = logging.getLogger(__name__)


class Timer:
    """Context manager that asserts a block of code completes within a time limit.

    Tracks per-label statistics (count, total, max) across all uses.
    Per-call logging is suppressed; use ``_log_run_analysis`` to emit a
    summary of averages at the end of a test run.
    """

    _stats: dict[str, dict[str, float]] = {}

    def __init__(
        self,
        max_time: float = MAX_TIME_TO_LOG_S,
        label: str | None = None,
        always_log: bool = False,
        log_threshold: float | None = None,
    ) -> None:
        self.max_time = max_time
        self.label = label

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> bool | None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        had_exception = len(args) > 0 and args[0] is not None

        if self.label:
            stats = self._stats.setdefault(
                self.label, {"count": 0, "total": 0.0, "max": 0.0}
            )
            stats["count"] += 1
            stats["total"] += self.interval
            stats["max"] = max(stats["max"], self.interval)

        if had_exception:
            return False

        assert self.interval < self.max_time, (
            f"{self.label or 'Function'} took too long: "
            f"{self.interval:.3f}s >= {self.max_time:.3f}s"
        )
        return None


def get_runner_pids() -> set[int]:
    """Retrieves the PIDs of the neuracore data daemon runner processes.

    Returns:
        set[int]: A set of PIDs of the runner processes.
    """
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    runner_pids: set[int] = set()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        if "neuracore.data_daemon.runner_entry" in args:
            runner_pids.add(int(pid_text))
    return runner_pids


def daemon_cleanup() -> None:
    """Cleanup the data daemon by killing all runner processes and removing
    the pid file, SQLite database file, and socket file.

    This function is idempotent and can be safely called multiple times.
    """
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    socket_path = Path(SOCKET_PATH)
    daemon_pids = set(get_runner_pids())

    if pid_path.exists():
        try:
            daemon_pids.add(int(pid_path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            pass

    for pid in daemon_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            subprocess.run(["kill", "-9", str(pid)], check=False)

    for path in (pid_path, db_path, socket_path):
        try:
            path.unlink(missing_ok=True)
        except IsADirectoryError:
            pass

    for suffix in (".shm", ".wal"):
        try:
            db_path.with_suffix(db_path.suffix + suffix).unlink(missing_ok=True)
        except OSError:
            pass


FRAME_BYTE_LENGTH = 16
GRID_SIZE = 4
DEFAULT_FILL_VALUE = 100
MAX_COLOR_VALUE = 255
HALF_DIVISOR = 2
COLOR_CHANNELS = 3


def encode_frame_number(frame_num: int, width: int, height: int) -> np.ndarray:
    """Encode a frame number into a video frame.

    The frame number is encoded into the top-left 4x4 grid of the frame.
    The frame number is converted to bytes and each byte is used to set
    the RGB values of the corresponding pixel. The R and B channels are set
    to the value of the byte, and the G channel is set to the value
    divided by 2.

    Args:
        frame_num: The frame number to encode
        width: The width of the frame
        height: The height of the frame

    Returns:
        np.ndarray: The encoded frame as a numpy array with shape (height, width, 3)
    """
    img = np.zeros((height, width, COLOR_CHANNELS), dtype=np.uint8)
    img.fill(DEFAULT_FILL_VALUE)

    frame_bytes = frame_num.to_bytes(FRAME_BYTE_LENGTH, byteorder="big")

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            idx = row * GRID_SIZE + col

            if idx < len(frame_bytes):
                pixel_value = frame_bytes[idx]

                img[row, col, 0] = pixel_value
                img[row, col, 1] = MAX_COLOR_VALUE - pixel_value
                img[row, col, 2] = pixel_value // HALF_DIVISOR

    return img


def decode_frame_number(img: np.ndarray) -> int:
    """Decodes a frame number from a video frame.

    The frame number is encoded in the top-left 4x4 grid of the frame.
    The frame number is converted from bytes and each byte is used to set
    the RGB values of the corresponding pixel. The R and B channels are set
    to the value of the byte, and the G channel is set to the value
    divided by 2.

    Args:
        img: The video frame as a numpy array with shape (height, width, 3)

    Returns:
        int: The decoded frame number
    """
    frame_bytes = bytearray()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            frame_bytes.append(img[i, j, 0])
    return int.from_bytes(frame_bytes[:FRAME_BYTE_LENGTH], byteorder="big")


def wait_for_dataset_ready(
    dataset_name: str,
    expected_recording_count: int = 1,
    timeout_s: float = 120.0,
    poll_interval_s: float = 1.5,
) -> None:
    """Wait until a dataset has a certain number of recordings.

    Args:
        dataset_name: Name of the dataset to wait for
        expected_recording_count: Number of recordings to wait for. Defaults to 1
        timeout_s: Maximum time to wait, in seconds. Defaults to 120.0
        poll_interval_s: Interval between polls, in seconds. Defaults to 1.5

    Raises:
        TimeoutError: If the dataset does not reach the expected number of
        recordings within the timeout period
    """

    wait_start = time.perf_counter()
    last_error: Exception | None = None

    while True:
        elapsed_s = time.perf_counter() - wait_start
        try:
            dataset = nc.get_dataset(dataset_name)
            if len(dataset) >= expected_recording_count:
                return
        except Exception as exc:
            last_error = exc

        if elapsed_s >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for dataset '{dataset_name}' to have "
                f"{expected_recording_count} recording(s) after {timeout_s}s"
            ) from last_error

        time.sleep(min(poll_interval_s, max(0.0, timeout_s - elapsed_s)))


def _multi_producer_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Runs a single producer worker.

    Args:
        spec: A dictionary containing the following keys:
            "robot_name": The name of the robot to connect to
            "dataset_name": The name of the dataset to create
            "fps": The desired frame rate
            "duration_sec": The duration of the recording in seconds
            "with_video": A boolean indicating whether to log RGB images
            "frame_offset": An integer indicating the starting frame number
            "producer_id": An integer identifying the producer
            "producer_marker": A float indicating the producer marker value
            "wait": A boolean indicating whether to wait for the recording to finish

    Returns:
        A dictionary containing the following keys:
            "producer_id": The producer ID
            "dataset_name": The dataset name
            "recording_id": The recording ID
            "duration_sec": The duration of the recording in seconds
            "with_video": A boolean indicating whether RGB images were logged
            "frame_offset": The starting frame number
            "frame_count": The number of frames logged
            "producer_marker": The producer marker value
            "stop_called_at": The time at which the stop_recording() call was made
            "stop_returned_at": The time at which the stop_recording() call returned
    """

    nc.login()

    robot = nc.connect_robot(spec["robot_name"])
    nc.create_dataset(spec["dataset_name"])
    nc.start_recording()

    recording_id = robot.get_current_recording_id()
    if recording_id is None:
        raise RuntimeError("Expected active recording_id after nc.start_recording()")

    fps = float(spec["fps"])
    duration_sec = float(spec["duration_sec"])
    time_step = 1.0 / fps
    t = 0.0
    frame_count = 0

    while t < duration_sec:
        if spec["with_video"]:
            frame_code = int(spec["frame_offset"]) + frame_count
            rgb_img = encode_frame_number(
                frame_code,
                int(spec["image_width"]),
                int(spec["image_height"]),
            )
            nc.log_rgb("camera_0", rgb_img, timestamp=t)

        nc.log_custom_1d(
            "producer_marker",
            np.array([float(spec["producer_marker"])], dtype=np.float32),
            timestamp=t,
        )
        frame_count += 1
        t += time_step

    stop_called_at = time.time()
    nc.stop_recording(wait=bool(spec["wait"]))
    stop_returned_at = time.time()

    return {
        "producer_id": int(spec["producer_id"]),
        "dataset_name": spec["dataset_name"],
        "recording_id": str(recording_id),
        "duration_sec": float(duration_sec),
        "with_video": bool(spec["with_video"]),
        "frame_offset": int(spec["frame_offset"]),
        "frame_count": frame_count,
        "producer_marker": float(spec["producer_marker"]),
        "stop_called_at": stop_called_at,
        "stop_returned_at": stop_returned_at,
    }


def run_multi_producers(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run multiple producers in parallel.

    Parameters:
    specs (list[dict[str, Any]]): A list of dictionaries, where each
    dictionary specifies the arguments for a single producer.

    Returns:
    list[dict[str, Any]]: A list of dictionaries, where each
    dictionary contains the results from a single producer.
    """
    with multiprocessing.Pool(len(specs)) as executor:
        return sorted(
            list(executor.map(_multi_producer_worker, specs)),
            key=lambda result: result["producer_id"],
        )


def collect_daemon_pids_from_parallel_startup(worker_count: int) -> list[int]:
    """Collect daemon PIDs from parallel startup.

    This function starts multiple daemon processes in parallel and waits for them to
    finish. It returns a list of daemon PIDs.

    Parameters:
    worker_count (int): The number of daemon processes to start.

    Returns:
    list[int]: A list of daemon PIDs.
    """

    def worker(
        barrier: multiprocessing.Barrier, results: dict[int, int], i: int
    ) -> None:
        barrier.wait()
        results[i] = ensure_daemon_running()

    barrier = multiprocessing.Barrier(worker_count)
    manager = multiprocessing.Manager()
    results = manager.dict()
    processes = []

    for i in range(worker_count):
        process = multiprocessing.Process(target=worker, args=(barrier, results, i))
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=25)
        assert (
            not process.is_alive()
        ), f"worker process {process.pid} did not finish before timeout"
        assert (
            process.exitcode == 0
        ), f"worker process {process.pid} exited with code {process.exitcode}"

    return list(results.values())


def build_multi_producer_specs(
    *,
    num_producers: int,
    wait: bool,
    with_video: bool,
    fps: int,
    duration_sec: float,
    image_width: int = 96,
    image_height: int = 72,
) -> list[dict[str, Any]]:
    """Builds a list of dictionaries specifying the arguments for multiple producers.

    Each dictionary in the returned list contains the arguments for a single producer.

    Parameters:
    num_producers (int): The number of producers to create.
    wait (bool): If True, the producers will wait for the daemon to
    finish before exiting.
    with_video (bool): If True, the producers will record video.
    fps (int): The frames per second to record at.
    duration_sec (float): The duration of the recording in seconds.
    image_width (int): The width of the image to record, in pixels. Defaults to 96.
    image_height (int): The height of the image to record, in pixels. Defaults to 72.

    Returns:
    list[dict[str, Any]]: A list of dictionaries, where each dictionary
    contains the arguments for a single producer.
    """
    specs = []
    for producer_id in range(num_producers):
        specs.append({
            "producer_id": producer_id,
            "robot_name": f"multi_robot_{producer_id}_{uuid.uuid4().hex[:8]}",
            "dataset_name": f"multi_dataset_{producer_id}_{uuid.uuid4().hex[:8]}",
            "wait": wait,
            "with_video": with_video,
            "fps": fps,
            "duration_sec": duration_sec,
            "image_width": image_width,
            "image_height": image_height,
            "frame_offset": producer_id * 1_000_000,
            "producer_marker": float(10_000 + producer_id),
            "api_key": os.environ.get("NEURACORE_API_KEY"),
        })
    return specs


def read_dataset_signals(dataset_name: str) -> dict[str, Any]:
    """Reads a dataset and extracts relevant signals.

    Parameters:
    dataset_name (str): The name of the dataset to read.

    Returns:
    dict[str, Any]: A dictionary containing the following keys:

    - recording_ids (set[str]): The ids of the recordings in the dataset.
    - synced_recording_ids (set[str]): The ids of the synchronized recordings.
    - recording_sync_timestamps (dict[str, list[float]]): A dictionary
    mapping recording ids to lists of timestamps.
    - producer_markers (set[float]): A set of producer marker values.
    - producer_marker_samples (int): The number of producer marker samples.
    - frame_codes (set[int]): A set of frame codes.
    - frame_code_counts (dict[int, int]): A dictionary mapping frame codes to counts.
    - rgb_samples (int): The total number of RGB image samples.
    - has_rgb (bool): If True, the dataset contains RGB image samples.
    - sync_points (int): The total number of synchronized points.
    """
    dataset = nc.get_dataset(dataset_name)
    recordings = list(dataset)
    recording_ids = {recording.id for recording in recordings}

    synced_dataset = dataset.synchronize()

    producer_marker_values: list[float] = []
    frame_code_counts: dict[int, int] = {}
    has_rgb = False
    sync_points = 0
    synced_recording_ids: set[str] = set()
    recording_sync_timestamps: dict[str, list[float]] = {}

    for synced_episode in synced_dataset:
        recording_id = getattr(synced_episode, "id", None)
        if recording_id is None:
            raise AssertionError("Synchronized recording is missing an id")
        recording_id = str(recording_id)
        synced_recording_ids.add(recording_id)
        timestamp_list = recording_sync_timestamps.setdefault(recording_id, [])

        for sync_point in synced_episode:
            sync_points += 1
            timestamp_list.append(float(sync_point.timestamp))

            if DataType.CUSTOM_1D in sync_point.data:
                custom_map = sync_point[DataType.CUSTOM_1D]
                if "producer_marker" in custom_map:
                    marker_data = np.asarray(custom_map["producer_marker"].data)
                    if marker_data.size > 0:
                        producer_marker_values.append(float(marker_data.flat[0]))

            if DataType.RGB_IMAGES in sync_point.data:
                has_rgb = True
                for _, cam_data in sync_point[DataType.RGB_IMAGES].items():
                    np_img = np.array(cam_data.frame)
                    frame_code = decode_frame_number(np_img)
                    frame_code_counts[frame_code] = (
                        frame_code_counts.get(frame_code, 0) + 1
                    )

    return {
        "recording_ids": recording_ids,
        "synced_recording_ids": synced_recording_ids,
        "recording_sync_timestamps": recording_sync_timestamps,
        "producer_markers": set(producer_marker_values),
        "producer_marker_samples": len(producer_marker_values),
        "frame_codes": set(frame_code_counts),
        "frame_code_counts": frame_code_counts,
        "rgb_samples": sum(frame_code_counts.values()),
        "has_rgb": has_rgb,
        "sync_points": sync_points,
    }


def assert_dataset_isolation(result: dict[str, Any]) -> None:
    """Assert that the given dataset is isolated.

    Checks that the dataset only contains data from the given recording,
    and that the producer marker samples are within the given recording's duration.

    If the dataset contains video, checks that the frame codes are
    continuous and within the given recording's duration, and that there
    are no duplicate frame codes.

    Args:
        result (dict[str, Any]): A dictionary containing the dataset name,
        recording id, duration in seconds, frame count, frame offset,
        producer marker,
        and whether the dataset contains video.

    Returns:
        None
    """
    signals = read_dataset_signals(result["dataset_name"])

    expected_recording_id = result["recording_id"]
    assert signals["recording_ids"] == {expected_recording_id}
    assert signals["synced_recording_ids"] == {expected_recording_id}

    timestamps = signals["recording_sync_timestamps"].get(expected_recording_id, [])
    assert timestamps, "Expected synchronized timestamps for producer recording"
    assert all(np.isfinite(ts) for ts in timestamps)

    producer_range_timestamps = [
        ts for ts in timestamps if 0.0 <= ts <= result["duration_sec"] + 1.0
    ]
    assert len(producer_range_timestamps) >= (result["frame_count"] - 1)

    expected_marker = result["producer_marker"]
    assert signals["producer_markers"] == {expected_marker}
    assert signals["producer_marker_samples"] == result["frame_count"]

    if result["with_video"]:
        assert signals["has_rgb"]
        expected_codes = set(
            range(
                result["frame_offset"],
                result["frame_offset"] + result["frame_count"],
            )
        )
        assert signals["frame_codes"] == expected_codes

        duplicate_codes = {
            code: count
            for code, count in signals["frame_code_counts"].items()
            if count > 1
        }
        assert not duplicate_codes
        assert signals["rgb_samples"] == result["frame_count"]
    else:
        assert not signals["has_rgb"]
        assert not signals["frame_codes"]


def stop_data_daemon() -> None:
    """Stop the Neuracore data daemon.

    This function stops the Neuracore data daemon by running
    the `stop` subcommand.
    It does not wait for the daemon to fully shut down,
    and does not check for any errors.
    """
    subprocess.run(
        [sys.executable, "-m", "neuracore.data_daemon", "stop"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _is_zombie_process(pid: int) -> bool:
    """Check if a process with the given PID is a zombie process.

    This function checks the state of a process with the given PID by running the
    `ps` command with the `-o stat` option. If the process is a zombie process,
    it will return `True`. If the process does not exist, it will return `False`.

    :param pid: The PID of the process to check
    :return: Whether the process is a zombie process
    """
    try:
        state = subprocess.check_output(
            ["ps", "-o", "stat=", "-p", str(pid)], text=True
        ).strip()
    except subprocess.SubprocessError:
        return False
    return state.startswith("Z")


def stop_daemon_for_mode_switch() -> None:
    """Stop the Neuracore data daemon and all its runner processes.

    This function is meant to be used when switching between different modes of
    operation, such as switching between recording and playback modes. It first
    asks the CLI to stop the daemon, and then checks if any of the runner
    processes are still running. If any are, it attempts to terminate them
    cleanly, and then force-kills them if that does not work. Finally, it waits
    for the processes to exit and then shuts down the daemon and its socket.

    :return: None
    """
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()

    # Ask the CLI to stop first; it handles normal daemon lifecycle paths.
    stop_data_daemon()

    candidate_pids: set[int] = set(get_runner_pids())
    pid_value = read_pid_from_file(pid_path)
    if pid_value is not None and pid_is_running(pid_value):
        candidate_pids.add(pid_value)

    for pid in sorted(candidate_pids):
        if not pid_is_running(pid):
            continue
        if not terminate_pid(pid):
            pytest.skip(f"No permission to terminate data daemon pid={pid}")

    for pid in sorted(candidate_pids):
        if not pid_is_running(pid):
            continue
        if wait_for_exit(pid, timeout_s=5.0):
            continue
        if not force_kill(pid):
            pytest.skip(f"No permission to force-kill data daemon pid={pid}")
        if not wait_for_exit(pid, timeout_s=5.0):
            if _is_zombie_process(pid):
                continue
            assert False, f"Failed to stop data daemon pid={pid}"

    shutdown(
        pid_path=pid_path,
        socket_paths=(Path(str(SOCKET_PATH)),),
        db_path=db_path,
    )


@contextmanager
def daemon_mode(*, offline: bool):
    """Context manager for temporarily switching modes.

    When entering the context manager, the daemon is
    stopped and then restarted in offline
    mode if offline is True. When exiting the context manager,
    the daemon is stopped again and then restarted in online mode.

    This context manager is useful for tests that
    require the daemon to be in offline mode,
    such as tests that record data directly to disk.

    :param offline: Whether to switch the daemon into offline mode
    :type offline: bool
    :yield: None
    :rtype: None
    """
    previous_offline = os.environ.get("NCD_OFFLINE")

    stop_daemon_for_mode_switch()
    os.environ["NCD_OFFLINE"] = "1" if offline else "0"
    ensure_daemon_running(timeout_s=10.0)
    try:
        yield
    finally:
        stop_daemon_for_mode_switch()
        if previous_offline is None:
            os.environ.pop("NCD_OFFLINE", None)
        else:
            os.environ["NCD_OFFLINE"] = previous_offline
        ensure_daemon_running(timeout_s=10.0)


@contextmanager
def use_offline_daemon_profile():
    """Context manager for temporarily using an offline data daemon profile.

    This context manager creates a temporary offline profile,
    sets the environment variable
    to point to it, stops the data daemon, and then restarts it.
      When exiting the context
    manager, it stops the data daemon again and then resets
    the environment variable to its
    previous value.

    This context manager is useful for tests that require the
    daemon to be in offline mode,
    such as tests that record data directly to disk.

    :yield: None
    :rtype: None
    """
    profile_name = f"offline_profile_{uuid.uuid4().hex[:8]}"
    profile_path = (
        Path.home() / ".neuracore" / "data_daemon" / "profiles" / f"{profile_name}.yaml"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("offline: true\n", encoding="utf-8")
    TEST_PROFILE_PATHS.add(profile_path)

    previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")
    os.environ["NEURACORE_DAEMON_PROFILE"] = profile_name
    stop_data_daemon()
    try:
        yield
    finally:
        stop_data_daemon()
        if previous_profile is None:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
        else:
            os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile


def run_minimal_recording_flow(label_prefix: str = "offline") -> str:
    """Run a minimal recording flow and return the recording ID.

    This function creates a robot connection, creates a
    dataset, starts recording,
    logs a single frame, and then stops recording.
    It then returns the recording ID.

    :param label_prefix: The prefix to use for the
    robot name and dataset name.
    :type label_prefix: str
    :return: The recording ID.
    :rtype: str
    """
    robot_name = f"{label_prefix}_robot_{uuid.uuid4().hex[:8]}"
    dataset_name = f"{label_prefix}_dataset_{uuid.uuid4().hex[:8]}"

    nc.login()
    robot = nc.connect_robot(
        robot_name,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    nc.create_dataset(dataset_name)
    nc.start_recording()

    recording_id = robot.get_current_recording_id()
    assert recording_id is not None

    frame = encode_frame_number(0, 64, 64)
    nc.log_rgb("camera_0", frame, timestamp=0.0)
    nc.log_custom_1d(
        "producer_marker",
        np.array([1.0], dtype=np.float32),
        timestamp=0.0,
    )
    nc.stop_recording(wait=False)

    return str(recording_id)


def fetch_trace_registration_stats(recording_id: str) -> tuple[int, int]:
    """Fetch trace registration stats for a recording.

    Returns a tuple of two integers: total traces and non-pending traces.

    :param recording_id: The recording ID to fetch stats for.
    :type recording_id: str
    :return: A tuple of two integers.
    :rtype: tuple[int, int]
    """
    db_path = get_daemon_db_path()
    with sqlite3.connect(db_path) as conn:
        total_traces = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()[0]
        non_pending = conn.execute(
            "SELECT COUNT(*) FROM traces "
            "WHERE recording_id = ? AND registration_status != 'pending'",
            (recording_id,),
        ).fetchone()[0]
    return int(total_traces), int(non_pending)


def fetch_expected_trace_count_reported(recording_id: str) -> int | None:
    """Fetch expected trace count reported for a recording.

    :param recording_id: The recording ID to fetch reported expected trace count for.
    :type recording_id: str
    :return: The reported expected trace count, or None if the recording
    ID is not found in the database.
    :rtype: int | None
    """
    db_path = get_daemon_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT expected_trace_count_reported FROM "
            "recordings WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
    if row is None:
        return None
    return int(row[0])


def wait_for_recording_to_exist_in_db(
    recording_id: str, timeout_s: float = 15.0
) -> None:
    """Wait for a recording to exist in the daemon database.

    :param recording_id: The recording ID to wait for.
    :type recording_id: str
    :param timeout_s: The timeout in seconds to wait for the recording to appear.
    :type timeout_s: float
    :raises pytest.fail: If the recording does not appear before the timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        total_traces, _ = fetch_trace_registration_stats(recording_id)
        reported = fetch_expected_trace_count_reported(recording_id)
        if total_traces > 0 and reported is not None:
            return
        time.sleep(0.2)

    pytest.fail(f"Recording {recording_id} did not appear in daemon DB before timeout")


def fetch_recording_recovery_stats(recording_id: str) -> dict[str, int | str | None]:
    """Fetch recovery stats for a recording.

    Args:
        recording_id: The recording ID to fetch stats for.

    Returns:
        A dict containing the following recovery stats:
        - expected_trace_count: The expected trace count for the recording.
        - expected_trace_count_reported: The expected trace count
        reported by the client.
        - progress_reported: The progress reported by the client.
        - total_traces: The total trace count for the recording.
        - non_pending_registration_traces: The trace count with
          non-pending registration status.
        - registered_traces: The trace count with registered registration status.
        - upload_progress_traces: The trace count with upload in progress status.
        - uploaded_traces: The trace count with uploaded upload status.

    :rtype: dict[str, int | str | None]
    """
    db_path = get_daemon_db_path()
    with sqlite3.connect(db_path) as conn:
        recording_row = conn.execute(
            "SELECT expected_trace_count, expected_trace_count_reported, "
            "progress_reported "
            "FROM recordings WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()

        trace_row = conn.execute(
            "SELECT "
            "COUNT(*) AS total_traces, "
            "SUM(CASE WHEN registration_status != 'pending' THEN 1 ELSE 0 END) "
            "AS non_pending_registration_traces, "
            "SUM(CASE WHEN registration_status = 'registered' THEN 1 ELSE 0 END) "
            "AS registered_traces, "
            "SUM(CASE WHEN upload_status IN ('queued', 'uploading', 'uploaded') "
            "THEN 1 ELSE 0 END) AS upload_progress_traces, "
            "SUM(CASE WHEN upload_status = 'uploaded' THEN 1 ELSE 0 END) "
            "AS uploaded_traces "
            "FROM traces WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()

    expected_trace_count = None
    expected_trace_count_reported = None
    progress_reported = None
    if recording_row is not None:
        expected_trace_count = int(recording_row[0])
        expected_trace_count_reported = int(recording_row[1])
        progress_reported = recording_row[2]

    return {
        "expected_trace_count": expected_trace_count,
        "expected_trace_count_reported": expected_trace_count_reported,
        "progress_reported": progress_reported,
        "total_traces": int(trace_row[0]),
        "non_pending_registration_traces": int(trace_row[1] or 0),
        "registered_traces": int(trace_row[2] or 0),
        "upload_progress_traces": int(trace_row[3] or 0),
        "uploaded_traces": int(trace_row[4] or 0),
    }


def wait_for_online_recovery(recording_id: str, timeout_s: float = 90.0) -> None:
    """
    Wait for online recovery to make progress for a recording.

    :param recording_id: The recording ID to wait for online recovery to progress.
    :type recording_id: str
    :param timeout_s: The timeout in seconds to wait for online recovery to progress.
    :type timeout_s: float
    :raises pytest.fail: If online recovery does not progress for the
    recording before the timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        stats = fetch_recording_recovery_stats(recording_id)

        fully_uploaded = (
            stats["total_traces"] == 0 and stats["progress_reported"] == "reported"
        )
        registration_attempted_or_done = stats["non_pending_registration_traces"] > 0
        expected_count_locally_set = (stats["expected_trace_count"] or 0) > 0

        if (
            fully_uploaded
            or registration_attempted_or_done
            or expected_count_locally_set
        ):
            return

        time.sleep(0.5)

    stats = fetch_recording_recovery_stats(recording_id)
    pytest.fail(
        "Online recovery did not progress for recording "
        f"{recording_id}; stats={stats}"
    )


def cleanup_test_profiles() -> None:
    """Clean up test profiles created by tests.

    This function is responsible for deleting any test profiles created
    by tests, and resetting the TEST_PROFILE_PATHS set to an empty
    set.

    :return: None
    """
    for profile_path in list(TEST_PROFILE_PATHS):
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass
    TEST_PROFILE_PATHS.clear()
