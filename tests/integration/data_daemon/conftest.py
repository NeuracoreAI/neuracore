"""Shared fixtures for data daemon integration tests.

Provides reusable socket and daemon context fixtures for testing
client-to-socket and client-to-daemon communication paths.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import zmq

import neuracore as nc
import neuracore.data_daemon.const as const_module
from neuracore.data_daemon.communications_management import (
    communications_manager as comms_module,
)
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon
from neuracore.data_daemon.communications_management.producer import Producer

logger = logging.getLogger(__name__)


class CaptureRDM:
    """Mock RecordingDiskManager that captures enqueued messages.

    Use this to verify the actual payload content that would be written to disk.
    Access captured messages via the `enqueued` list.
    """

    def __init__(self) -> None:
        self.enqueued: list = []

    def enqueue(self, message) -> None:
        """Capture a message instead of writing to disk."""
        self.enqueued.append(message)

    def shutdown(self) -> None:
        """No-op shutdown for compatibility."""
        pass


@dataclass
class DaemonRDMCapture:
    """Context for a running daemon with capture for payload verification."""

    daemon: Daemon
    capture: CaptureRDM
    context: zmq.Context
    producer: Producer


def _wait_for(predicate, timeout: float, interval: float = 0.05) -> bool:
    """Poll predicate until it returns True or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _run_daemon_loop(
    daemon: Daemon,
    comm: CommunicationsManager,
    stop_event: threading.Event,
    ready_event: threading.Event,
    error_bucket: list[BaseException],
) -> None:
    """Run the daemon message loop in a background thread."""
    try:
        comm.start_consumer()
        comm.start_publisher()
    except BaseException as exc:
        error_bucket.append(exc)
        ready_event.set()
        return
    ready_event.set()
    if comm.consumer_socket is None:
        raise RuntimeError("consumer socket not initialized")
    comm.consumer_socket.setsockopt(zmq.RCVTIMEO, 200)
    comm.consumer_socket.setsockopt(zmq.LINGER, 0)

    while not stop_event.is_set():
        msg = None
        try:
            msg = comm.receive_message()
        except zmq.Again:
            msg = None

        if msg is not None:
            daemon.handle_message(msg)

        daemon._cleanup_expired_channels()
        daemon._drain_channel_messages()

    comm.cleanup_daemon()


@pytest.fixture
def ipc_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Set up isolated IPC socket paths for testing.

    Uses ipc:// transport with temp files to allow cross-context communication.
    Yields the base directory path for recordings.
    """
    base_dir = tmp_path / "ndd"
    base_dir.mkdir(parents=True, exist_ok=True)
    short_id = uuid4().hex[:8]
    socket_path = Path(f"/tmp/ndd-{short_id}.sock")
    events_path = Path(f"/tmp/nde-{short_id}.sock")

    mpsa = monkeypatch.setattr

    mpsa(const_module, "BASE_DIR", base_dir)
    mpsa(const_module, "SOCKET_PATH", socket_path)
    mpsa(const_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)
    mpsa(comms_module, "BASE_DIR", base_dir)
    mpsa(comms_module, "SOCKET_PATH", socket_path)
    mpsa(comms_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)

    yield base_dir

    if socket_path.exists():
        socket_path.unlink()
    if events_path.exists():
        events_path.unlink()


@pytest.fixture
def daemon_with_capture(
    ipc_paths: Path,
) -> Generator[DaemonRDMCapture, None, None]:
    """Start a daemon with a CaptureRDM to verify payload content.

    Use this when you need to verify the actual payload data that was
    reassembled from chunks, not just that data was written.
    """
    capture_rdm = CaptureRDM()
    context = zmq.Context()
    comm = CommunicationsManager(context=context)
    daemon = Daemon(
        comm_manager=comm,
        recording_disk_manager=capture_rdm,
    )

    stop_event = threading.Event()
    ready_event = threading.Event()
    error_bucket: list[BaseException] = []
    thread = threading.Thread(
        target=_run_daemon_loop,
        args=(daemon, comm, stop_event, ready_event, error_bucket),
        daemon=True,
    )
    thread.start()

    assert _wait_for(ready_event.is_set, timeout=2)
    if error_bucket:
        raise error_bucket[0]

    producer_comm = CommunicationsManager(context=context)
    producer = Producer(comm_manager=producer_comm, chunk_size=64)

    try:
        yield DaemonRDMCapture(
            daemon=daemon,
            capture=capture_rdm,
            context=context,
            producer=producer,
        )
    finally:
        stop_event.set()
        thread.join(timeout=2)
        context.destroy(linger=0)


@pytest.fixture
def mock_socket() -> Generator[MagicMock, None, None]:
    """Mock the ZMQ socket to capture sent bytes.

    Mocks CommunicationsManager.create_producer_socket to return a mock socket.
    The real send_message() runs and serializes to bytes, which we capture.
    Use send.call_args_list to inspect captured calls.
    """
    mock_sock = MagicMock()

    with patch.object(
        CommunicationsManager, "create_producer_socket", return_value=mock_sock
    ):
        yield mock_sock


TEST_ROBOT = "basic_test_robot"


@pytest.fixture
def stream_to_daemon_with_capture(
    daemon_with_capture: DaemonRDMCapture,
) -> Generator[DaemonRDMCapture, None, None]:
    """Combine daemon capture with client API streaming setup.

    Setup:
        - nc.login(), nc.connect_robot(), nc.create_dataset(), nc.start_recording()
        - daemon_with_capture provides real daemon for verification

    Teardown:
        - nc.stop_recording()
    """
    dataset_name = f"test_dataset_{uuid4().hex[:8]}"
    recording_started = False

    logger.info("Setting up streaming test with daemon capture")
    nc.login()
    nc.connect_robot(TEST_ROBOT)
    nc.create_dataset(dataset_name)

    nc.start_recording()
    recording_started = True
    logger.info(f"Recording started for dataset: {dataset_name}")

    try:
        yield daemon_with_capture
    finally:
        logger.info("Tearing down streaming test")
        if recording_started:
            try:
                nc.stop_recording(wait=True)
                logger.info("Recording stopped successfully")
            except Exception as exception:
                logger.warning(f"Error stopping recording: {exception}")
                try:
                    nc.cancel_recording()
                    logger.info("Recording cancelled")
                except Exception as cancel_error:
                    logger.error(f"Failed to cancel recording: {cancel_error}")
