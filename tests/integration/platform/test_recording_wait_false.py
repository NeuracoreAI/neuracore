import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from neuracore_types import DataType

import neuracore as nc
from examples.common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH
from neuracore.core.robot import Robot
from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    launch_new_daemon_subprocess,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    fetch_all_traces,
    fetch_recording,
)
from tests.integration.platform.data_daemon.shared.test_case import (
    build_test_case_context,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.append(str(REPO_ROOT / "examples"))

logger = logging.getLogger(__name__)

MAX_TIME_TO_START_S = 20
MAX_TIME_TO_STOP_S = 5

_TEST_DAEMON_ENV_OVERRIDES = {
    "NCD_BANDWIDTH_LIMIT": str(200 * 1024 * 1024),
    "NCD_MAX_CONCURRENT_UPLOADS": "30",
    "NCD_VIDEO_ACK_TIMEOUT_SECONDS": "30",
    "NCD_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS": "30",
    "NCD_OFFLINE": "1",
}


class Timer:
    def __init__(
        self, *, label: str, max_time: float, always_log: bool = False
    ) -> None:
        self.label = label
        self.max_time = max_time
        self.always_log = always_log

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_args) -> bool:
        elapsed = time.perf_counter() - self.start

        if self.always_log or elapsed >= self.max_time:
            logger.info(
                "Timer %-36s %.3fs (limit=%.3fs)",
                self.label,
                elapsed,
                self.max_time,
            )

        if exc_type is not None:
            return False

        assert (
            elapsed < self.max_time
        ), f"{self.label} took too long: {elapsed:.3f}s >= {self.max_time:.3f}s"
        return False


def _log_rgb_stream_transport_stats(
    robot: Robot,
    *,
    camera_name: str,
    phase: str,
) -> None:
    stream_id = f"{DataType.RGB_IMAGES.value}:{camera_name}"
    stream = robot.get_data_stream(stream_id)

    if stream is None:
        logger.info(
            "RGB transport stats unavailable phase=%s stream_id=%s",
            phase,
            stream_id,
        )
        return

    producer_channel: ProducerChannel | None = getattr(
        stream, "_producer_channel", None
    )

    if producer_channel is None:
        logger.info(
            "RGB transport stats unavailable phase=%s stream_id=%s "
            "producer_channel=None",
            phase,
            stream_id,
        )
        return

    stats = producer_channel.get_transport_stats().to_dict()

    logger.info(
        "RGB transport stats phase=%s queue=%s/%s pending_seq=%s "
        "last_enqueued=%s last_sent=%s sender_alive=%s heartbeat_alive=%s "
        "slots=%s free=%s inflight=%s max_inflight=%s acked=%s ack_timeouts=%s "
        "last_ack_seq=%s last_ack_latency=%.3fs max_ack_latency=%.3fs "
        "worker_queue=%s/%s worker_alive=%s worker_error=%s unhealthy_reason=%s "
        "trace_id=%s recording_id=%s",
        phase,
        stats["send_queue_qsize"],
        stats["send_queue_maxsize"],
        stats["pending_sequence_count"],
        stats["last_enqueued_sequence_number"],
        stats["last_socket_sent_sequence_number"],
        stats["sender_thread_alive"],
        stats["heartbeat_thread_alive"],
        stats.get("slot_count"),
        stats.get("free_slot_count"),
        stats.get("in_flight_slot_count"),
        stats.get("max_inflight_slot_count") or stats.get("max_in_flight_slot_count"),
        stats.get("acked_sequence_count"),
        stats.get("ack_timeout_count"),
        stats.get("last_acked_sequence_id"),
        float(stats.get("last_ack_latency_s") or 0.0),
        float(stats.get("max_ack_latency_s") or 0.0),
        stats.get("worker_queue_qsize"),
        stats.get("worker_queue_maxsize"),
        stats.get("worker_thread_alive"),
        stats.get("worker_error"),
        stats.get("unhealthy_reason"),
        stats["trace_id"],
        stats["recording_id"],
    )


def _get_rgb_transport_stats(
    robot: Robot | None = None,
    *,
    camera_name: str | None = None,
    producer_channel: ProducerChannel | None = None,
) -> dict[str, object]:
    """Return current RGB producer transport stats for one camera stream."""
    if producer_channel is None:
        if robot is None or camera_name is None:
            raise AssertionError(
                "Expected either producer_channel or robot/camera_name to be set"
            )

        stream_id = f"{DataType.RGB_IMAGES.value}:{camera_name}"
        stream = robot.get_data_stream(stream_id)

        if stream is None:
            raise AssertionError(f"Missing RGB stream for stream_id={stream_id}")

        producer_channel = getattr(stream, "_producer_channel", None)
        if producer_channel is None:
            raise AssertionError(f"Missing producer channel for stream_id={stream_id}")

    return producer_channel.get_transport_stats().to_dict()


def _wait_for_rgb_transport_to_drain(
    robot: Robot | None = None,
    *,
    camera_name: str | None = None,
    producer_channel: ProducerChannel | None = None,
    timeout_s: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_stats: dict[str, object] | None = None

    while time.monotonic() < deadline:
        stats = _get_rgb_transport_stats(
            robot,
            camera_name=camera_name,
            producer_channel=producer_channel,
        )
        last_stats = stats

        if (
            stats["pending_sequence_count"] == 0
            and stats["send_queue_qsize"] == 0
            and stats["send_error_count"] == 0
            and stats.get("worker_queue_qsize", 0) == 0
            and stats.get("in_flight_slot_count", 0) == 0
            and stats.get("worker_error") is None
            and stats.get("unhealthy_reason") is None
        ):
            return stats

        time.sleep(0.05)

    raise AssertionError(
        "RGB producer transport did not drain cleanly. "
        f"camera={camera_name} producer_channel={producer_channel is not None} "
        f"last_stats={last_stats}"
    )


def _transport_showed_post_stop_progress(
    *,
    pre_stop_stats: dict[str, object],
    drained_stats: dict[str, object],
) -> bool:
    """Return True when stop(wait=False) left producer-side work to flush."""
    pre_stop_sent = int(pre_stop_stats["last_socket_sent_sequence_number"])
    drained_sent = int(drained_stats["last_socket_sent_sequence_number"])
    pre_stop_enqueued = int(pre_stop_stats["last_enqueued_sequence_number"])
    drained_enqueued = int(drained_stats["last_enqueued_sequence_number"])

    return (
        int(pre_stop_stats["send_queue_qsize"]) > 0
        or int(pre_stop_stats["pending_sequence_count"]) > 0
        or drained_sent > pre_stop_sent
        or drained_enqueued > pre_stop_enqueued
    )


def _wait_for_rgb_trace_handed_off_to_daemon(
    *,
    recording_id: str,
    camera_name: str,
    timeout_s: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_recording_row: dict[str, object] | None = None
    last_rows: list[dict[str, object]] = []

    while time.monotonic() < deadline:
        try:
            recording_row = fetch_recording(recording_id)
            rows = fetch_all_traces(
                recording_id,
                columns=[
                    "trace_id",
                    "data_type",
                    "data_type_name",
                    "write_status",
                    "error_code",
                    "error_message",
                ],
            )
        except sqlite3.OperationalError:
            time.sleep(0.1)
            continue

        last_recording_row = recording_row
        last_rows = rows

        rgb_rows = [
            row
            for row in rows
            if row.get("data_type") == DataType.RGB_IMAGES.value
            and row.get("data_type_name") == camera_name
        ]

        if len(rgb_rows) == 1:
            rgb_row = rgb_rows[0]
            if (
                recording_row is not None
                and recording_row.get("expected_trace_count") == 1
                and recording_row.get("stopped_at") is not None
                and rgb_row.get("write_status") != "failed"
                and rgb_row.get("error_code") is None
                and rgb_row.get("error_message") is None
            ):
                logger.info(
                    "RGB trace handed off to daemon recording_id=%s trace_id=%s "
                    "expected_trace_count=%s stopped_at=%s write_status=%s",
                    recording_id,
                    rgb_row.get("trace_id"),
                    recording_row.get("expected_trace_count"),
                    recording_row.get("stopped_at"),
                    rgb_row.get("write_status"),
                )

                return {
                    "recording_row": recording_row,
                    "trace_row": rgb_row,
                }

        time.sleep(0.1)

    raise AssertionError(
        "Timed out waiting for RGB trace handoff to daemon. "
        f"recording_id={recording_id} camera={camera_name} "
        f"last_recording_row={last_recording_row} last_rows={last_rows}"
    )


def _stop_robot_producer_channels(robot: Robot) -> None:
    for stream in robot._data_streams.values():
        producer_channel = getattr(stream, "_producer_channel", None)
        if isinstance(producer_channel, ProducerChannel):
            producer_channel.stop_producer_channel()


def _cleanup_robot_after_test(robot: Robot) -> None:
    """Best-effort local teardown so each test starts with a clean client state."""
    try:
        robot._stop_all_streams(wait_for_producer_drain=False)
    except Exception:
        logger.warning(
            "Failed to stop all robot streams during test cleanup",
            exc_info=True,
        )

    try:
        _stop_robot_producer_channels(robot)
    except Exception:
        logger.warning(
            "Failed to stop producer channels during test cleanup",
            exc_info=True,
        )

    try:
        robot.close()
    except Exception:
        logger.warning(
            "Failed to close robot resources during test cleanup",
            exc_info=True,
        )


@pytest.fixture(autouse=True)
def isolated_daemon_for_test():
    if os.getenv("NCD_SKIP_DAEMON_CLEANUP_FOR_DEBUG") == "1":
        yield
        return

    cleanup_script = REPO_ROOT / "cleanup.sh"

    previous_env = {key: os.environ.get(key) for key in _TEST_DAEMON_ENV_OVERRIDES}
    daemon_log_handle = None
    daemon_log_path: Path | None = None

    try:
        subprocess.run(["bash", str(cleanup_script)], cwd=REPO_ROOT, check=True)

        os.environ.update(_TEST_DAEMON_ENV_OVERRIDES)
        daemon_log_file = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="neuracore-daemon-test-",
            suffix=".log",
            delete=False,
        )
        daemon_log_handle = daemon_log_file
        daemon_log_path = Path(daemon_log_file.name)

        daemon_process = launch_new_daemon_subprocess(
            pid_path=get_daemon_pid_path(),
            db_path=get_daemon_db_path(),
            background=False,
            timeout_s=10.0,
            stdout=daemon_log_file.fileno(),
            stderr=daemon_log_file.fileno(),
        )

        logger.info(
            "Launched clean data daemon for test pid=%s log_path=%s",
            daemon_process.pid,
            daemon_log_path,
        )

        yield

    finally:
        if daemon_log_handle is not None:
            daemon_log_handle.flush()
            daemon_log_handle.close()

        if daemon_log_path is not None and daemon_log_path.exists():
            try:
                tail_lines = daemon_log_path.read_text(encoding="utf-8").splitlines()[
                    -120:
                ]
                if tail_lines:
                    logger.info(
                        "Daemon log tail path=%s\n%s",
                        daemon_log_path,
                        "\n".join(tail_lines),
                    )
            except Exception:
                logger.warning(
                    "Failed to read daemon log tail path=%s",
                    daemon_log_path,
                    exc_info=True,
                )

        subprocess.run(["bash", str(cleanup_script)], cwd=REPO_ROOT, check=True)

        for key, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _run_wait_false_rgb_payload_test(
    *,
    dataset_cleanup,
    label: str,
    frame_width: int,
    frame_height: int,
    frame_count: int,
    log_rgb_max_time_s: float,
    transport_drain_timeout_s: float,
    trace_written_timeout_s: float,
) -> None:
    robot_name = f"test_robot_{label}_{uuid.uuid4().hex[:8]}"
    dataset_name = f"test_dataset_{label}_{uuid.uuid4().hex[:8]}"
    dataset_cleanup(dataset_name)

    camera_name = "camera_0"
    bytes_per_frame = frame_width * frame_height * 3
    expected_raw_trace_bytes = frame_count * bytes_per_frame
    total_payload_mib = expected_raw_trace_bytes / (1024 * 1024)

    logger.info(
        "Starting wait=False RGB payload test label=%s frames=%d resolution=%dx%d "
        "bytes_per_frame=%d total_payload_mib=%.2f",
        label,
        frame_count,
        frame_width,
        frame_height,
        bytes_per_frame,
        total_payload_mib,
    )

    robot: Robot | None = None

    try:
        with Timer(label="nc.login", max_time=MAX_TIME_TO_START_S, always_log=True):
            nc.login()

        with Timer(
            label="nc.connect_robot",
            max_time=MAX_TIME_TO_START_S,
            always_log=True,
        ):
            robot = nc.connect_robot(
                robot_name,
                urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
                overwrite=False,
            )

        with Timer(
            label="nc.create_dataset",
            max_time=MAX_TIME_TO_START_S,
            always_log=True,
        ):
            nc.create_dataset(
                dataset_name,
                description=f"RGB stop_recording(wait=False) payload test: {label}",
            )

        with Timer(
            label="nc.start_recording",
            max_time=MAX_TIME_TO_START_S,
            always_log=True,
        ):
            nc.start_recording()

        recording_id = robot.get_current_recording_id()
        assert recording_id is not None, "Expected active recording ID after start"

        _log_rgb_stream_transport_stats(
            robot,
            camera_name=camera_name,
            phase=f"{label}_after_start",
        )

        for frame_num in range(frame_count):
            timestamp = frame_num * 0.1
            frame = build_test_case_context.encode_frame_number(
                frame_num, frame_width, frame_height
            )

            with Timer(
                label=f"{label}.nc.log_rgb",
                max_time=log_rgb_max_time_s,
                always_log=False,
            ):
                nc.log_rgb(camera_name, frame, timestamp=timestamp)

            if frame_num in {0, frame_count // 2, frame_count - 1}:
                _log_rgb_stream_transport_stats(
                    robot,
                    camera_name=camera_name,
                    phase=f"{label}_after_frame_{frame_num}",
                )

        _log_rgb_stream_transport_stats(
            robot,
            camera_name=camera_name,
            phase=f"{label}_before_stop",
        )
        stream_id = f"{DataType.RGB_IMAGES.value}:{camera_name}"
        stream = robot.get_data_stream(stream_id)
        assert stream is not None, f"Expected RGB stream before stop: {stream_id}"
        producer_channel = stream.get_producer_channel()
        assert (
            producer_channel is not None
        ), f"Expected producer channel before stop: {stream_id}"
        pre_stop_transport_stats = _get_rgb_transport_stats(
            producer_channel=producer_channel,
        )

        with Timer(
            label=f"{label}.nc.stop_recording(wait=False)",
            max_time=MAX_TIME_TO_STOP_S,
            always_log=True,
        ):
            nc.stop_recording(wait=False)

        drained_transport_stats = _wait_for_rgb_transport_to_drain(
            producer_channel=producer_channel,
            camera_name=camera_name,
            timeout_s=transport_drain_timeout_s,
        )

        _log_rgb_stream_transport_stats(
            robot,
            camera_name=camera_name,
            phase=f"{label}_after_transport_drain",
        )

        trace_results = _wait_for_rgb_trace_handed_off_to_daemon(
            recording_id=recording_id,
            camera_name=camera_name,
            timeout_s=trace_written_timeout_s,
        )

        recording_row = trace_results["recording_row"]
        trace_row = trace_results["trace_row"]

        assert recording_row["expected_trace_count"] == 1
        assert recording_row["stopped_at"] is not None
        assert trace_row["write_status"] != "failed"
        assert trace_row["error_code"] is None
        assert trace_row["error_message"] is None

        assert drained_transport_stats["send_error_count"] == 0
        assert drained_transport_stats["last_send_error"] is None
        assert drained_transport_stats["pending_sequence_count"] == 0
        assert drained_transport_stats["send_queue_qsize"] == 0
        assert _transport_showed_post_stop_progress(
            pre_stop_stats=pre_stop_transport_stats,
            drained_stats=drained_transport_stats,
        ), (
            "Expected producer transport activity to continue after "
            "stop_recording(wait=False), but producer-side sender state already "
            "looked fully flushed before stop. "
            f"pre_stop_transport_stats={pre_stop_transport_stats} "
            f"drained_transport_stats={drained_transport_stats}"
        )

        logger.info(
            "wait=False RGB payload test completed label=%s recording_id=%s "
            "frames_sent=%d total_payload_mib=%.2f",
            label,
            recording_id,
            frame_count,
            total_payload_mib,
        )

    finally:
        if robot is not None:
            _cleanup_robot_after_test(robot)


def test_stop_recording_wait_false_rgb_small_payload(dataset_cleanup):
    _run_wait_false_rgb_payload_test(
        dataset_cleanup=dataset_cleanup,
        label="small",
        frame_width=640,
        frame_height=480,
        frame_count=10,
        log_rgb_max_time_s=5.0,
        transport_drain_timeout_s=30.0,
        trace_written_timeout_s=60.0,
    )


def test_stop_recording_wait_false_rgb_medium_payload(dataset_cleanup):
    _run_wait_false_rgb_payload_test(
        dataset_cleanup=dataset_cleanup,
        label="medium",
        frame_width=1280,
        frame_height=720,
        frame_count=60,
        log_rgb_max_time_s=10.0,
        transport_drain_timeout_s=60.0,
        trace_written_timeout_s=120.0,
    )


@pytest.mark.slow
def test_stop_recording_wait_false_rgb_realistic_large_payload(dataset_cleanup):
    _run_wait_false_rgb_payload_test(
        dataset_cleanup=dataset_cleanup,
        label="realistic_large",
        frame_width=int(os.getenv("NCD_LARGE_RGB_FRAME_WIDTH", "1920")),
        frame_height=int(os.getenv("NCD_LARGE_RGB_FRAME_HEIGHT", "1080")),
        frame_count=int(os.getenv("NCD_LARGE_RGB_FRAME_COUNT", "180")),
        log_rgb_max_time_s=float(os.getenv("NCD_LARGE_RGB_LOG_MAX_TIME_S", "20")),
        transport_drain_timeout_s=float(
            os.getenv("NCD_LARGE_RGB_DRAIN_TIMEOUT_S", "180")
        ),
        trace_written_timeout_s=float(
            os.getenv("NCD_LARGE_RGB_TRACE_TIMEOUT_S", "300")
        ),
    )


@pytest.mark.slow
@pytest.mark.stress
def test_stop_recording_wait_false_rgb_blowaway_payload(dataset_cleanup):
    """Unrealistically heavy RGB payload stress test.

    Defaults:
    - 4K RGB frames: 3840x2160
    - 300 frames
    - ~7.0 GiB raw RGB payload

    Override with:
    NCD_BLOWAWAY_RGB_FRAME_WIDTH
    NCD_BLOWAWAY_RGB_FRAME_HEIGHT
    NCD_BLOWAWAY_RGB_FRAME_COUNT
    NCD_BLOWAWAY_RGB_LOG_MAX_TIME_S
    NCD_BLOWAWAY_RGB_DRAIN_TIMEOUT_S
    NCD_BLOWAWAY_RGB_TRACE_TIMEOUT_S
    """
    old_limit = os.environ.get("NCD_BANDWIDTH_LIMIT")
    os.environ["NCD_BANDWIDTH_LIMIT"] = str(1024 * 1024 * 1024)

    try:
        _run_wait_false_rgb_payload_test(
            dataset_cleanup=dataset_cleanup,
            label="blowaway",
            frame_width=int(os.getenv("NCD_BLOWAWAY_RGB_FRAME_WIDTH", "3840")),
            frame_height=int(os.getenv("NCD_BLOWAWAY_RGB_FRAME_HEIGHT", "2160")),
            frame_count=int(os.getenv("NCD_BLOWAWAY_RGB_FRAME_COUNT", "300")),
            log_rgb_max_time_s=float(
                os.getenv("NCD_BLOWAWAY_RGB_LOG_MAX_TIME_S", "60")
            ),
            transport_drain_timeout_s=float(
                os.getenv("NCD_BLOWAWAY_RGB_DRAIN_TIMEOUT_S", "600")
            ),
            trace_written_timeout_s=float(
                os.getenv("NCD_BLOWAWAY_RGB_TRACE_TIMEOUT_S", "900")
            ),
        )
    finally:
        if old_limit is None:
            os.environ.pop("NCD_BANDWIDTH_LIMIT", None)
        else:
            os.environ["NCD_BANDWIDTH_LIMIT"] = old_limit


@pytest.mark.slow
@pytest.mark.stress
def test_stop_recording_wait_false_multi_camera_4k_burst(dataset_cleanup):
    """
    Burst stress test:
    - 4 simultaneous RGB cameras
    - 4K frames
    - short burst to avoid blowing container RAM/disk

    Default total payload:
    3840 * 2160 * 3 * 80 * 4 ~= 7.42 GiB
    """

    robot_name = f"multi_cam_burst_robot_{uuid.uuid4().hex[:8]}"
    dataset_name = f"multi_cam_burst_dataset_{uuid.uuid4().hex[:8]}"
    dataset_cleanup(dataset_name)

    camera_count = int(os.getenv("NCD_BURST_CAMERA_COUNT", "24"))
    frame_width = int(os.getenv("NCD_BURST_RGB_FRAME_WIDTH", "3840"))
    frame_height = int(os.getenv("NCD_BURST_RGB_FRAME_HEIGHT", "2160"))
    frame_count = int(os.getenv("NCD_BURST_RGB_FRAME_COUNT", "80"))

    camera_names = [f"camera_{i}" for i in range(camera_count)]

    bytes_per_frame = frame_width * frame_height * 3
    total_payload_bytes = bytes_per_frame * frame_count * camera_count

    logger.info(
        "Starting multi-camera 4K burst test cameras=%d frames=%d "
        "resolution=%dx%d bytes_per_frame=%d total_payload_gib=%.2f",
        camera_count,
        frame_count,
        frame_width,
        frame_height,
        bytes_per_frame,
        total_payload_bytes / (1024**3),
    )

    robot: Robot | None = None

    try:
        nc.login()

        robot = nc.connect_robot(
            robot_name,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )

        nc.create_dataset(
            dataset_name,
            description="Multi-camera 4K RGB burst stress test",
        )

        nc.start_recording()

        recording_id = robot.get_current_recording_id()
        assert recording_id is not None

        enqueue_start = time.perf_counter()

        for frame_num in range(frame_count):
            timestamp = frame_num * 0.1

            # Generate once per frame number, reuse across cameras to avoid
            # extra RAM churn.
            frame = build_test_case_context.encode_frame_number(
                frame_num, frame_width, frame_height
            )

            for camera_name in camera_names:
                nc.log_rgb(camera_name, frame, timestamp=timestamp)

            if frame_num in {0, frame_count // 2, frame_count - 1}:
                for camera_name in camera_names:
                    _log_rgb_stream_transport_stats(
                        robot,
                        camera_name=camera_name,
                        phase=f"burst_after_frame_{frame_num}_{camera_name}",
                    )

        enqueue_elapsed = time.perf_counter() - enqueue_start

        logger.info(
            "Finished enqueueing 4K burst frames elapsed=%.2fs throughput_gib_s=%.2f",
            enqueue_elapsed,
            (total_payload_bytes / (1024**3)) / max(enqueue_elapsed, 0.001),
        )

        producer_channels_by_camera: dict[str, ProducerChannel] = {}
        for camera_name in camera_names:
            stream_id = f"{DataType.RGB_IMAGES.value}:{camera_name}"
            stream = robot.get_data_stream(stream_id)
            assert stream is not None, f"Expected RGB stream before stop: {stream_id}"
            producer_channel = stream.get_producer_channel()
            assert (
                producer_channel is not None
            ), f"Expected producer channel before stop: {stream_id}"
            producer_channels_by_camera[camera_name] = producer_channel

        with Timer(
            label="multi_camera_4k_burst.nc.stop_recording(wait=False)",
            max_time=MAX_TIME_TO_STOP_S,
            always_log=True,
        ):
            nc.stop_recording(wait=False)

        for camera_name in camera_names:
            _wait_for_rgb_transport_to_drain(
                producer_channel=producer_channels_by_camera[camera_name],
                camera_name=camera_name,
                timeout_s=float(os.getenv("NCD_BURST_RGB_DRAIN_TIMEOUT_S", "300")),
            )
        default_timeout = 200
        timeout_for_trace_written = str(default_timeout * camera_count)

        for camera_name in camera_names:
            _wait_for_rgb_trace_handed_off_to_daemon(
                recording_id=recording_id,
                camera_name=camera_name,
                timeout_s=float(
                    os.getenv("NCD_BURST_RGB_TRACE_TIMEOUT_S", timeout_for_trace_written)
                ),
            )

        logger.info(
            "Multi-camera 4K burst test completed recording_id=%s "
            "cameras=%d frames_per_camera=%d total_payload_gib=%.2f",
            recording_id,
            camera_count,
            frame_count,
            total_payload_bytes / (1024**3),
        )

    finally:
        if robot is not None:
            _cleanup_robot_after_test(robot)
