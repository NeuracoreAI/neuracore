"""Recording-scoped context for sending recording control messages to the daemon."""

from __future__ import annotations

import logging
from importlib import import_module

from neuracore.data_daemon.models import CommandType
from neuracore.data_daemon.rust_selection import rust_daemon_enabled

from .communications_manager import CommunicationsManager, MessageEnvelope

logger = logging.getLogger(__name__)


class RecordingContext:
    """Recording-scoped context for sending recording control messages."""

    def __init__(
        self,
        recording_id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initialize the recording context.

        Under the rust daemon the ZMQ producer socket is unused — lifecycle
        envelopes flow through ``_native_producer`` over iceoryx2 instead — so
        we skip creating it to avoid spinning up a CommunicationsManager that
        nothing on the other side will consume.
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

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        """Send a recording-stopped control message."""
        effective_recording_id = recording_id or self.recording_id
        if not effective_recording_id:
            raise ValueError("recording_id is required to stop a recording.")

        if self._rust_mode:
            try:
                native = import_module("neuracore.data_daemon._native_producer")
            except ImportError:
                logger.exception(
                    "Failed to import _native_producer; "
                    "the rust daemon will not receive StopRecording for %s",
                    effective_recording_id,
                )
            else:
                native.stop_recording(effective_recording_id)
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
