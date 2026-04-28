"""Trace and recording lifecycle coordination for the daemon consumer bridge."""

from __future__ import annotations

import logging
from collections.abc import Callable

from neuracore_types import DataType

from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.helpers import utc_now
from neuracore.data_daemon.models import MessageEnvelope

from .helpers import int_or_none, str_or_none, trace_metadata_dict
from .models import (
    ChannelState,
    CompletedChannelMessage,
    FinalChunkRegistry,
    FinalTraceWork,
    PendingSharedSlotSequenceRegistry,
    PendingTraceEnd,
    PendingTraceEndRegistry,
    ProducerSequenceRegistry,
    RecordingCloseRegistry,
    RecordingClosingState,
    RecordingDataDropRequest,
    SharedSlotSequenceProgressRequest,
    TraceMetadataRegistrationRequest,
    TraceMetadataRegistry,
    TraceMetadataSnapshot,
    TraceRecordingLookupRequest,
    TraceRecordingRegistry,
)

logger = logging.getLogger(__name__)


class TraceLifecycleCoordinator:
    """Own trace/recording state transitions for the daemon bridge."""

    def __init__(
        self,
        *,
        emitter: Emitter,
        enqueue_final_trace: Callable[[FinalTraceWork], None],
    ) -> None:
        """Initialize registries used to coordinate trace and recording lifecycles."""
        self._emitter = emitter
        self._enqueue_final_trace = enqueue_final_trace
        self._trace_recordings = TraceRecordingRegistry()
        self._trace_metadata = TraceMetadataRegistry()
        self._pending_trace_ends = PendingTraceEndRegistry()
        self._final_chunk_enqueued_traces = FinalChunkRegistry()
        self._closing_recordings = RecordingCloseRegistry()
        self._producer_last_sequence_numbers = ProducerSequenceRegistry()
        self._pending_shared_slot_sequences = PendingSharedSlotSequenceRegistry()

    def ensure_result_trace_registered(
        self,
        *,
        channel: ChannelState,
        result: CompletedChannelMessage,
    ) -> str | None:
        """Ensure trace/recording metadata is registered for one completed result."""
        trace_id = result.trace_id
        recording_id = self._trace_recordings.get_recording_id(trace_id)
        if recording_id is not None:
            return recording_id

        metadata = trace_metadata_dict(result.metadata)
        recording_id = str_or_none(metadata.get("recording_id"))
        if recording_id is None:
            return None

        channel.set_trace_id(trace_id)
        self.register_trace(recording_id, trace_id)
        self.register_trace_metadata(
            TraceMetadataRegistrationRequest(
                trace_id=trace_id,
                metadata=TraceMetadataSnapshot(
                    dataset_id=str_or_none(metadata.get("dataset_id")),
                    dataset_name=str_or_none(metadata.get("dataset_name")),
                    robot_name=str_or_none(metadata.get("robot_name")),
                    robot_id=str_or_none(metadata.get("robot_id")),
                    robot_instance=int_or_none(metadata.get("robot_instance")),
                    data_type=(
                        str_or_none(metadata.get("data_type")) or result.data_type.value
                    ),
                    data_type_name=str_or_none(metadata.get("data_type_name")),
                ),
            )
        )
        return recording_id

    def get_trace_metadata(self, trace_id: str) -> dict[str, str | int | None]:
        """Return stored metadata for one trace."""
        return self._trace_metadata.get(trace_id)

    def get_trace_recording(self, request: TraceRecordingLookupRequest) -> str | None:
        """Return the recording currently associated with one trace, if any."""
        return self._trace_recordings.get_recording_id(request.trace_id)

    def note_producer_sequence(self, producer_id: str, sequence_number: int) -> None:
        """Record the most recent sequence number observed for a producer."""
        self._producer_last_sequence_numbers.update(producer_id, sequence_number)

    def set_max_producer_sequence(self, producer_id: str, sequence_number: int) -> None:
        """Raise the stored producer sequence only when the new value is larger."""
        self._producer_last_sequence_numbers.set_max(producer_id, sequence_number)

    def channel_stop_cutoff_sequence_number(
        self,
        producer_id: str,
        channel: ChannelState,
    ) -> int | None:
        """Return stop cutoff sequence for the channel's active trace, if known."""
        trace_id = channel.trace_id
        if trace_id is None:
            return None
        recording_id = self._trace_recordings.get_recording_id(trace_id)
        if recording_id is None:
            return None
        closing_state = self._closing_recordings.get_closing(recording_id)
        if closing_state is None:
            return None
        cutoffs = closing_state.producer_stop_sequence_numbers
        if not cutoffs:
            return None
        cutoff = cutoffs.get(producer_id)
        if cutoff is None:
            return None
        return int(cutoff)

    def register_trace(self, recording_id: str, trace_id: str) -> None:
        """Register a trace to a recording."""
        existing = self._trace_recordings.register(recording_id, trace_id)
        if existing is not None:
            logger.warning(
                "Trace %s moved from recording %s to %s",
                trace_id,
                existing,
                recording_id,
            )

    def register_trace_metadata(
        self,
        request: TraceMetadataRegistrationRequest,
    ) -> None:
        """Register metadata for a trace."""
        for key, existing_value, incoming_value in self._trace_metadata.register(
            request
        ):
            logger.warning(
                "Trace %s metadata mismatch for %s (%s -> %s)",
                request.trace_id,
                key,
                existing_value,
                incoming_value,
            )

    def mark_shared_slot_sequence_pending(
        self,
        request: SharedSlotSequenceProgressRequest,
    ) -> None:
        """Record that one shared-slot descriptor still needs spool processing."""
        self._pending_shared_slot_sequences.add(
            request.producer_id,
            request.sequence_number,
        )

    def mark_shared_slot_sequence_completed(
        self,
        request: SharedSlotSequenceProgressRequest,
    ) -> None:
        """Record that one shared-slot descriptor reached completion handoff."""
        self._pending_shared_slot_sequences.complete(
            request.producer_id,
            request.sequence_number,
        )

    def should_drop_recording_data(self, request: RecordingDataDropRequest) -> bool:
        """Return True when recording state says this data should be dropped."""
        channel = request.channel
        recording_id = request.recording_id
        trace_id = request.trace_id
        sequence_number = request.sequence_number
        if self._closing_recordings.is_closed(recording_id):
            logger.warning(
                "Dropping data for closed recording_id=%s trace_id=%s",
                recording_id,
                trace_id,
            )
            return True

        closing_state = self._closing_recordings.get_closing(recording_id)
        if closing_state is None or not closing_state.producer_stop_sequence_numbers:
            return False
        cutoff_sequence_number = closing_state.producer_stop_sequence_numbers.get(
            channel.producer_id
        )
        if cutoff_sequence_number is None:
            logger.warning(
                "Dropping data from producer_id=%s while recording_id=%s is closing "
                "(missing stop sequence number)",
                channel.producer_id,
                recording_id,
            )
            return True
        if sequence_number is None:
            logger.warning(
                "Dropping data for producer_id=%s recording_id=%s without "
                "sequence_number while recording is closing",
                channel.producer_id,
                recording_id,
            )
            return True
        if sequence_number > cutoff_sequence_number:
            logger.warning(
                "Dropping post-stop data for producer_id=%s recording_id=%s "
                "(sequence_number=%s, cutoff_sequence_number=%s)",
                channel.producer_id,
                recording_id,
                sequence_number,
                cutoff_sequence_number,
            )
            return True
        return False

    def handle_trace_end(self, channel: ChannelState, message: MessageEnvelope) -> None:
        """Record one deferred TRACE_END until stop cutoffs are reached."""
        payload = message.payload.get("trace_end", {})
        trace_id = payload.get("trace_id")
        if not trace_id:
            return

        registered_recording_id = self._trace_recordings.get_recording_id(str(trace_id))
        recording_id = registered_recording_id
        if not recording_id:
            recording_id = str_or_none(payload.get("recording_id"))
        if not recording_id:
            logger.warning(
                "TRACE_END received without recording for producer_id=%s trace_id=%s "
                "sequence_number=%s",
                channel.producer_id,
                trace_id,
                message.sequence_number,
            )
            return

        metadata = self._trace_metadata.get(str(trace_id))
        data_type_str = metadata.get("data_type")
        if data_type_str:
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                logger.warning(
                    "Dropping trace_id=%s for recording_id=%s due to unknown "
                    "data_type=%r on TRACE_END",
                    trace_id,
                    recording_id,
                    data_type_str,
                )
                self._drop_invalid_trace_state(str(recording_id), str(trace_id))
                return
        else:
            if registered_recording_id is not None:
                logger.warning(
                    "Dropping trace_id=%s for recording_id=%s because TRACE_END "
                    "arrived without registered data_type metadata",
                    trace_id,
                    recording_id,
                )
                self._drop_invalid_trace_state(str(recording_id), str(trace_id))
                return
            logger.warning(
                "TRACE_END received for trace_id=%s without registered metadata; "
                "ignoring finalization",
                trace_id,
            )
            return

        self._pending_trace_ends.add(
            PendingTraceEnd(
                producer_id=channel.producer_id,
                recording_id=str(recording_id),
                trace_id=str(trace_id),
                data_type=data_type,
                sequence_number=message.sequence_number,
            )
        )

    def handle_recording_stopped(self, message: MessageEnvelope) -> None:
        """Mark a recording as closing and notify downstream listeners."""
        payload = message.payload.get("recording_stopped", {})
        recording_id = payload.get("recording_id")
        if not recording_id:
            logger.warning(
                "RECORDING_STOPPED missing recording_id (producer_id=%s)",
                message.producer_id,
            )
            return

        producer_stop_sequence_numbers_raw = payload.get(
            "producer_stop_sequence_numbers", {}
        )
        producer_stop_sequence_numbers: dict[str, int] = {}
        if isinstance(producer_stop_sequence_numbers_raw, dict):
            for (
                producer_id,
                sequence_number,
            ) in producer_stop_sequence_numbers_raw.items():
                try:
                    producer_stop_sequence_numbers[str(producer_id)] = int(
                        sequence_number
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid stop sequence number for producer_id=%s: %r",
                        producer_id,
                        sequence_number,
                    )
        else:
            logger.warning(
                "recording_stopped.producer_stop_sequence_numbers must be a dict"
            )

        self._closing_recordings.mark_closing(
            recording_id,
            RecordingClosingState(
                producer_stop_sequence_numbers=producer_stop_sequence_numbers,
                stop_requested_at=utc_now(),
            ),
        )
        self._emitter.emit(Emitter.STOP_RECORDING_REQUESTED, recording_id)

    def finalize_closing_recordings(self) -> None:
        """Finalize recordings after stop cutoffs and final trace chunks are written."""
        to_close: list[str] = []

        for recording_id, closing_state in self._closing_recordings.items():
            if not self._has_reached_sequence_cutoffs(closing_state):
                continue

            traces = self._trace_recordings.traces_for_recording(recording_id)

            for trace_id in traces:
                if self._final_chunk_enqueued_traces.contains(trace_id):
                    continue

                pending_trace_end = self._pending_trace_ends.get(trace_id)
                if pending_trace_end is None:
                    logger.debug(
                        "Waiting for TRACE_END before finalizing "
                        "recording_id=%s trace_id=%s",
                        recording_id,
                        trace_id,
                    )
                    continue

                self._enqueue_final_trace(
                    FinalTraceWork(
                        producer_id=pending_trace_end.producer_id,
                        trace_id=trace_id,
                        recording_id=recording_id,
                        data_type=pending_trace_end.data_type,
                        metadata=dict(self._trace_metadata.get(trace_id)),
                    )
                )

                self._final_chunk_enqueued_traces.add(trace_id)

            if not self._trace_recordings.has_active_traces(recording_id):
                to_close.append(recording_id)

        for recording_id in to_close:
            expected_trace_count = self._trace_recordings.unique_trace_count(
                recording_id
            )
            self._emitter.emit(
                Emitter.SET_EXPECTED_TRACE_COUNT,
                recording_id,
                expected_trace_count,
            )
            self._closing_recordings.close(recording_id)
            self._trace_recordings.clear_unique_traces(recording_id)
            self._emitter.emit(Emitter.STOP_RECORDING, recording_id)

    def cleanup_trace_written(self, trace_id: str) -> str | None:
        """Drop daemon-side lifecycle state for one written trace."""
        recording_id = self._trace_recordings.get_recording_id(trace_id)
        if recording_id is None:
            return None

        self._trace_metadata.pop(trace_id)
        self._pending_trace_ends.pop(trace_id)
        self._final_chunk_enqueued_traces.discard(trace_id)
        self._trace_recordings.remove_trace(recording_id, trace_id)
        return str(recording_id)

    def _drop_invalid_trace_state(self, recording_id: str, trace_id: str) -> None:
        """Prune trace lifecycle state when metadata is too incomplete to finalize."""
        self._trace_metadata.pop(trace_id)
        self._pending_trace_ends.pop(trace_id)
        self._final_chunk_enqueued_traces.discard(trace_id)
        self._trace_recordings.remove_trace(recording_id, trace_id)

    def _has_reached_sequence_cutoffs(
        self,
        closing_state: RecordingClosingState,
    ) -> bool:
        """Return True when all producer sequence cutoffs have been observed."""
        stop_cutoffs = closing_state.producer_stop_sequence_numbers
        if not stop_cutoffs:
            return True

        for producer_id, cutoff_sequence_number in stop_cutoffs.items():
            last_sequence_number = self._producer_last_sequence_numbers.get(producer_id)
            if (
                last_sequence_number is None
                or last_sequence_number < cutoff_sequence_number
            ):
                return False
            if self._pending_shared_slot_sequences.has_pending_at_or_before(
                producer_id,
                cutoff_sequence_number,
            ):
                return False
        return True
