"""Recording-scoped context for sending recording control messages to the daemon."""

from __future__ import annotations

from neuracore.data_daemon.models import CommandType

from .communications_manager import CommunicationsManager, MessageEnvelope


class RecordingContext:
    """Recording-scoped context for sending recording control messages."""

    def __init__(
        self,
        recording_id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initialize the recording context."""
        self.recording_id = recording_id
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
        envelope = MessageEnvelope(
            producer_id=None,
            command=command,
            payload=payload or {},
        )
        self._comm.send_message(envelope)
