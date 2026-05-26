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


class RecordingContext:
    """Recording-scoped interface to the data daemon.

    Under the Rust daemon this is the single entry point the logging layer
    uses to drive the native producer: ``start_recording`` /
    ``log_joints`` / ``log_frame`` / ``log_scalar`` / ``stop_recording`` /
    ``cancel_recording`` forward straight through to ``_native_producer`` over
    iceoryx2. The native crate owns all trace state — trace ids are minted and
    discarded inside Rust — so this object holds only the recording id.

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
        recording_id: str,
        robot_id: str | None = None,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Announce a recording to the Rust daemon.

        Publishes exactly one ``StartRecording`` envelope. The native layer is
        idempotent — a repeated call for the same recording is a no-op — so a
        mistaken ``start, start`` does not emit a duplicate.
        """
        if not self._rust_mode:
            return
        if not recording_id:
            raise ValueError("recording_id is required to start a recording.")
        self.recording_id = recording_id
        _load_native().start_recording(
            recording_id, robot_id, robot_name, dataset_id, dataset_name
        )

    def log_joints(
        self,
        data_type: str,
        timestamp: float,
        items: list[tuple[str, float]],
    ) -> None:
        """Forward a batch of ``(joint_name, value)`` samples to the daemon.

        The Rust crate lazily creates a trace per joint on first sight and
        packs the whole batch into one IPC message.
        """
        if not items:
            return
        recording_id = self._require_recording_id("log_joints")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_joints(
            recording_id, data_type, items, timestamp_ns, timestamp
        )

    def log_frame(
        self,
        data_type: str,
        name: str,
        width: int,
        height: int,
        payload: memoryview,
        timestamp: float,
    ) -> None:
        """Forward one video frame to the daemon.

        ``payload`` may be either a ``bytes`` object or a flat ``memoryview``
        (e.g. ``memoryview(numpy_array).cast("B")``); the native side reads
        the buffer via the Python buffer protocol and copies straight into
        the NUT writer's destination, so there's no benefit to materialising
        a ``bytes`` first.
        """
        recording_id = self._require_recording_id("log_frame")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_frame(
            recording_id,
            data_type,
            name,
            int(width),
            int(height),
            payload,
            timestamp_ns,
            timestamp,
        )

    def log_scalar(
        self,
        data_type: str,
        name: str,
        payload: bytes,
        timestamp: float,
    ) -> None:
        """Forward one scalar/custom sample to the daemon."""
        recording_id = self._require_recording_id("log_scalar")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_scalar(
            recording_id, data_type, name, payload, timestamp_ns, timestamp
        )

    def cancel_recording(self, recording_id: str | None = None) -> None:
        """Cancel a recording — the daemon discards every in-flight trace."""
        effective_recording_id = recording_id or self.recording_id
        if not effective_recording_id:
            raise ValueError("recording_id is required to cancel a recording.")
        if self._rust_mode:
            _load_native().cancel_recording(effective_recording_id)
        self.recording_id = effective_recording_id

    def _require_recording_id(self, operation: str) -> str:
        """Return the active recording id or raise if logging before start."""
        if not self._rust_mode:
            raise RuntimeError(f"{operation} is only available under the rust daemon.")
        if not self.recording_id:
            raise RuntimeError(
                f"{operation} called before start_recording set a recording id."
            )
        return self.recording_id

    # -- Lifecycle ----------------------------------------------------------

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        """Send a recording-stopped control message.

        Under the Rust daemon this ends every trace the recording minted and
        publishes one ``StopRecording``; the native layer is idempotent, so a
        mistaken ``stop, stop`` is harmless.
        """
        effective_recording_id = recording_id or self.recording_id
        if not effective_recording_id:
            raise ValueError("recording_id is required to stop a recording.")

        if self._rust_mode:
            _load_native().stop_recording(effective_recording_id)
        else:
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
