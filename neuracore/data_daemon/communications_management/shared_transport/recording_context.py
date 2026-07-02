"""Recording-scoped context for driving the data daemon."""

from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType

from neuracore.data_daemon.models import CommandType
from neuracore.data_daemon.rust_selection import rust_daemon_enabled

from .communications_manager import CommunicationsManager, MessageEnvelope

logger = logging.getLogger(__name__)

_NATIVE_MODULE: ModuleType | None = None

_NATIVE_IMPORT_HINT = (
    "neuracore.data_daemon._native_producer is not available. Build the Rust "
    "data_daemon_producer crate with maturin and ensure the resulting "
    "extension is on sys.path, or unset NCD_RUST_DAEMON to fall back to the "
    "legacy Python producer."
)


def _load_native() -> ModuleType:
    """Lazily import and cache the PyO3 producer module for the process."""
    global _NATIVE_MODULE
    if _NATIVE_MODULE is None:
        try:
            _NATIVE_MODULE = import_module("neuracore.data_daemon._native_producer")
        except ImportError as error:
            raise RuntimeError(_NATIVE_IMPORT_HINT) from error
    return _NATIVE_MODULE


def notify_daemon_config_changed() -> None:
    """Try ask a running Rust daemon to reload its profile immediately.

    This is a no-op under the legacy Python producer.
    """
    if not rust_daemon_enabled():
        return
    try:
        _load_native().refresh_config()
    except Exception as error:  # noqa: BLE001 - best-effort, never fatal
        logger.debug("Could not notify the daemon of a config change: %s", error)


class RecordingContext:
    """Recording-scoped interface to the data daemon.

    Under the Rust daemon this is a *thin shipper* bridge: ``start_recording``
    / ``log_joints`` / ``log_frame`` / ``log_json`` / ``stop_recording`` /
    ``cancel_recording`` forward straight through to ``_native_producer`` over
    iceoryx2, tagged only with the **source** ``(robot_id, robot_instance)``.
    The daemon owns all recording identity — there is no recording id on the
    wire. Routing is by a producer-stamped *publish* timestamp (wall clock),
    decoupled from the data's own capture timestamp, so the daemon partitions
    recordings by when data was published rather than what clock it carries.

    Under the legacy daemon it keeps its original role: sending recording
    control messages over the ZMQ producer socket. That path is unchanged.
    """

    def __init__(
        self,
        recording_id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initialize the recording context.

        Under the Rust daemon the ZMQ producer socket is unused — every
        envelope flows through ``_native_producer`` over iceoryx2 — so we skip
        creating it.
        """
        self.recording_id = recording_id
        self._rust_mode = rust_daemon_enabled()
        self._robot_id: str | None = None
        self._robot_instance: int = 0
        self._recording_marker_ns: int = 0
        if self._rust_mode:
            self._comm = None
        else:
            self._comm = comm_manager or CommunicationsManager()
            self._comm.create_producer_socket()

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set or clear the recording identifier for this context."""
        self.recording_id = recording_id

    # -- Native (Rust daemon) interface -------------------------------------

    def start_recording(
        self,
        robot_id: str,
        robot_instance: int = 0,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Announce a recording to the Rust daemon for a source.

        Publishes exactly one ``StartRecording`` envelope tagged with the
        source ``(robot_id, robot_instance)``. No recording id is on the wire —
        the daemon allocates and owns recording identity.

        ``timestamp`` optionally pins the recording window's lower bound (Unix
        seconds), matching the ``log_*`` methods; when ``None`` the producer
        stamps the publish clock now.
        """
        if not self._rust_mode:
            return
        if not robot_id:
            raise ValueError("robot_id is required to start a recording.")
        self._robot_id = robot_id
        self._robot_instance = robot_instance
        timestamp_ns = int(timestamp * 1_000_000_000) if timestamp is not None else None

        self._recording_marker_ns = _load_native().start_recording(
            robot_id,
            robot_instance,
            robot_name,
            dataset_id,
            dataset_name,
            timestamp_ns,
        )

    def log_joints(
        self,
        data_type: str,
        timestamp: float,
        joined_names: str,
        values: list[float],
    ) -> None:
        r"""Forward a batch of joint scalar samples to the daemon.

        Args:
            data_type:  Type of joint data e.g. DataType.JOINT_POSITIONS.
            timestamp: the Unix timestamp of the sample.
            joined_names: a single ``\0``-joined string of joint names.
            values: a flat list of joint values.
        """
        if not values:
            return
        robot_id = self._require_source("log_joints")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_joints(
            robot_id,
            self._robot_instance,
            data_type,
            joined_names,
            values,
            timestamp_ns,
            timestamp,
        )

    def log_frame(
        self,
        data_type: str,
        name: str,
        width: int,
        height: int,
        payload: bytes | memoryview,
        timestamp: float,
    ) -> None:
        """Forward one video frame to the daemon.

        Args:
            data_type: Type of video data e.g. DataType.RGB_IMAGES.
            name: the camera/sensor name e.g. left_wrist_camera.
            width: Video frame width.
            height: Video frame height.
            payload: Raw video frame bytes.
            timestamp: the Unix timestamp of the sample.
        """
        robot_id = self._require_source("log_frame")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_frame(
            robot_id,
            self._robot_instance,
            data_type,
            name,
            int(width),
            int(height),
            payload,
            timestamp_ns,
            timestamp,
        )

    def log_json(
        self,
        data_type: str,
        name: str,
        payload: bytes,
        timestamp: float,
    ) -> None:
        """Forward one JSON sample to the daemon.

        The generic single-sample path for any non-joint, non-video data type
        (scalars, poses, gripper amounts, language, point clouds, ...). The
        ``data_type`` is an opaque wire label and ``payload`` is already
        serialized, so the daemon stores it verbatim as a per-trace JSON sample.
        """
        robot_id = self._require_source("log_json")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_json(
            robot_id,
            self._robot_instance,
            data_type,
            name,
            payload,
            timestamp_ns,
            timestamp,
        )

    def cancel_recording(
        self,
        recording_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Cancel the source's active recording — the daemon discards it.

        A cancel is a recording stop that discards data, so ``timestamp``
        behaves exactly like ``stop_recording``'s: it optionally pins the
        recording's capture stop time (Unix seconds); when ``None`` the producer
        stamps wall-clock now.
        """
        if not self._rust_mode:
            return
        if not self._robot_id:
            return
        timestamp_ns = int(timestamp * 1_000_000_000) if timestamp is not None else None
        _load_native().cancel_recording(
            self._robot_id, self._robot_instance, timestamp_ns
        )

    def _require_source(self, operation: str) -> str:
        """Return the active source's robot id or raise if logging before start."""
        if not self._rust_mode:
            raise RuntimeError(f"{operation} is only available under the rust daemon.")
        if not self._robot_id:
            raise RuntimeError(
                f"{operation} called before start_recording set a source."
            )
        return self._robot_id

    # -- Lifecycle ----------------------------------------------------------

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Send a recording-stopped control message.

        Under the Rust daemon this publishes one ``StopRecording`` tagged with
        the source and the publish-clock stop boundary, which the daemon uses to
        close the recording window. ``timestamp`` optionally pins that boundary
        (Unix seconds), matching the ``log_*`` methods; when ``None`` the
        producer stamps wall-clock now. ``recording_id`` /
        ``producer_stop_sequence_numbers`` are only used by the legacy path.
        """
        if self._rust_mode:
            if not self._robot_id:
                return
            timestamp_ns = (
                int(timestamp * 1_000_000_000) if timestamp is not None else None
            )
            _load_native().stop_recording(
                self._robot_id, self._robot_instance, timestamp_ns
            )
            return

        effective_recording_id = recording_id or self.recording_id
        if not effective_recording_id:
            raise ValueError("recording_id is required to stop a recording.")
        recording_stopped_payload: dict[str, object] = {
            "recording_id": effective_recording_id
        }
        if producer_stop_sequence_numbers:
            recording_stopped_payload["producer_stop_sequence_numbers"] = (
                producer_stop_sequence_numbers
            )
        self._send(
            CommandType.RECORDING_STOPPED,
            {"recording_stopped": recording_stopped_payload},
        )
        self.recording_id = effective_recording_id

    def get_recording_id(
        self,
        timestamp_ns: int | None = None,
        timeout_s: float = 30.0,
    ) -> str | None:
        """Resolve the daemon-owned cloud recording id for this source.

        The producer never sees the cloud recording id — the
        daemon allocates it and POSTs ``/recording/start`` asynchronously. This
        asks the daemon over the native ``queries`` request-response service for
        the id of the recording identified by this source and the capture
        ``timestamp_ns`` marker (defaulting to the marker captured at
        ``start_recording``). The daemon answers authoritatively from its own
        state; the native call blocks (with the GIL released) until the id is
        minted or ``timeout_s`` elapses.

        It MAY block and is for non-performance-critical paths only (tests,
        ``nc.stop_recording(wait=True)``). Returns ``None`` on timeout or in the
        legacy daemon mode.
        """
        if not self._rust_mode or not self._robot_id:
            return None
        marker_ns = (
            timestamp_ns if timestamp_ns is not None else self._recording_marker_ns
        )
        return _load_native().get_recording_id(
            self._robot_id, self._robot_instance, marker_ns, timeout_s
        )

    def close(self) -> None:
        """Close sockets and cleanup context resources owned by this instance."""
        if self._comm is not None:
            self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> None:
        """Send a management message to the daemon.

        Args:
            command: The CommandType to send to the daemon.
            payload: A dictionary containing any additional data required by the daemon
                to process the message.

        Returns:
            None
        """
        if self._comm is None:
            raise RuntimeError(
                "Cannot send a control message: no CommunicationsManager is "
                "available (rust daemon mode bypasses the ZMQ producer socket)."
            )
        envelope = MessageEnvelope(
            producer_id=None,
            command=command,
            payload=payload or {},
        )
        self._comm.send_message(envelope)
