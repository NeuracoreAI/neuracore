import asyncio

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.consumer import (
    trace_lifecycle_coordinator,
)
from neuracore.data_daemon.communications_management.consumer.models import (
    ChannelState,
    CompletedChannelMessage,
    FinalTraceWork,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
    TraceRecordingLookupRequest,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import CommandType, MessageEnvelope

TraceLifecycleCoordinator = trace_lifecycle_coordinator.TraceLifecycleCoordinator


def _build_coordinator() -> TraceLifecycleCoordinator:
    loop = asyncio.new_event_loop()
    return TraceLifecycleCoordinator(
        emitter=Emitter(loop=loop),
        enqueue_final_trace=lambda _: None,
        set_channel_trace_id=lambda *_: None,
    )


class _MetadataWithoutDataType:
    def to_dict(self) -> dict[str, str]:
        return {
            "recording_id": "recording-1",
            "data_type_name": "camera",
        }


def test_cleanup_trace_written_handles_empty_pending_trace_end_registry() -> None:
    coordinator = _build_coordinator()
    coordinator.register_trace("recording-1", "trace-1")

    recording_id = coordinator.cleanup_trace_written("trace-1")

    assert recording_id == "recording-1"


def test_handle_trace_end_drops_registered_trace_missing_data_type_metadata() -> None:
    coordinator = _build_coordinator()
    coordinator.register_trace("recording-1", "trace-1")

    coordinator.handle_trace_end(
        ChannelState(producer_id="producer-1", trace_id="trace-1"),
        MessageEnvelope(
            producer_id="producer-1",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-1",
                    "recording_id": "recording-1",
                }
            },
        ),
    )

    assert (
        coordinator.get_trace_recording(TraceRecordingLookupRequest(trace_id="trace-1"))
        is None
    )


def test_ensure_result_trace_registered_uses_completed_result_data_type_fallback() -> (
    None
):
    coordinator = _build_coordinator()
    channel = ChannelState(producer_id="producer-1")

    coordinator.ensure_result_trace_registered(
        channel=channel,
        result=CompletedChannelMessage(
            trace_id="trace-1",
            data_type=DataType.RGB_IMAGES,
            payload=b"payload",
            metadata=_MetadataWithoutDataType(),
        ),
    )

    metadata = coordinator.get_trace_metadata("trace-1")
    assert metadata["data_type"] == DataType.RGB_IMAGES.value


def test_closing_recordings_advance_from_lifecycle_events() -> None:
    loop = asyncio.new_event_loop()
    emitter = Emitter(loop=loop)
    enqueued_trace_ids: list[str] = []
    stopped_recordings: list[str] = []
    emitter.on(
        Emitter.STOP_RECORDING,
        lambda recording_id: stopped_recordings.append(recording_id),
    )
    coordinator = TraceLifecycleCoordinator(
        emitter=emitter,
        enqueue_final_trace=lambda work: enqueued_trace_ids.append(work.trace_id),
        set_channel_trace_id=lambda *_: None,
    )
    channel = ChannelState(producer_id="producer-1", trace_id="trace-1")

    coordinator.register_trace("recording-1", "trace-1")
    coordinator.register_trace_metadata(
        TraceMetadataRegistrationRequest(
            trace_id="trace-1",
            metadata=TraceMetadataSnapshot(
                data_type=DataType.RGB_IMAGES.value,
                data_type_name="camera",
            ),
        )
    )
    coordinator.handle_trace_end(
        channel,
        MessageEnvelope(
            producer_id="producer-1",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-1",
                    "recording_id": "recording-1",
                }
            },
            sequence_number=5,
        ),
    )
    coordinator.handle_recording_stopped(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={
                "recording_stopped": {
                    "recording_id": "recording-1",
                    "producer_stop_sequence_numbers": {"producer-1": 5},
                }
            },
        )
    )

    assert enqueued_trace_ids == []
    assert stopped_recordings == []

    coordinator.note_producer_sequence("producer-1", 5)

    assert enqueued_trace_ids == ["trace-1"]
    assert stopped_recordings == []

    recording_id = coordinator.cleanup_trace_written("trace-1")

    assert recording_id == "recording-1"
    assert stopped_recordings == ["recording-1"]


def test_finalize_closing_recordings_enqueues_final_trace_once() -> None:
    loop = asyncio.new_event_loop()
    emitter = Emitter(loop=loop)
    enqueued_work: list[FinalTraceWork] = []
    coordinator = TraceLifecycleCoordinator(
        emitter=emitter,
        enqueue_final_trace=lambda work: enqueued_work.append(work),
        set_channel_trace_id=lambda *_: None,
    )
    channel = ChannelState(producer_id="producer-1", trace_id="trace-1")

    coordinator.register_trace("recording-1", "trace-1")
    coordinator.register_trace_metadata(
        TraceMetadataRegistrationRequest(
            trace_id="trace-1",
            metadata=TraceMetadataSnapshot(
                dataset_name="dataset-a",
                data_type=DataType.RGB_IMAGES.value,
                data_type_name="camera",
            ),
        )
    )
    coordinator.handle_trace_end(
        channel,
        MessageEnvelope(
            producer_id="producer-1",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-1",
                    "recording_id": "recording-1",
                }
            },
            sequence_number=5,
        ),
    )
    coordinator.handle_recording_stopped(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={
                "recording_stopped": {
                    "recording_id": "recording-1",
                    "producer_stop_sequence_numbers": {"producer-1": 5},
                }
            },
        )
    )

    coordinator.set_max_producer_sequence("producer-1", 5)
    coordinator.finalize_closing_recordings()
    coordinator.finalize_closing_recordings()

    assert len(enqueued_work) == 1
    assert enqueued_work[0] == FinalTraceWork(
        producer_id="producer-1",
        trace_id="trace-1",
        recording_id="recording-1",
        data_type=DataType.RGB_IMAGES,
        metadata={
            "dataset_id": None,
            "dataset_name": "dataset-a",
            "robot_name": None,
            "robot_id": None,
            "robot_instance": None,
            "data_type": DataType.RGB_IMAGES.value,
            "data_type_name": "camera",
        },
    )


def test_finalize_closing_recordings_closes_recording_after_cutoffs() -> None:
    loop = asyncio.new_event_loop()
    emitter = Emitter(loop=loop)
    expected_trace_counts: list[tuple[str, int]] = []
    stopped_recordings: list[str] = []
    emitter.on(
        Emitter.SET_EXPECTED_TRACE_COUNT,
        lambda recording_id, count: expected_trace_counts.append((recording_id, count)),
    )
    emitter.on(
        Emitter.STOP_RECORDING,
        lambda recording_id: stopped_recordings.append(recording_id),
    )
    coordinator = TraceLifecycleCoordinator(
        emitter=emitter,
        enqueue_final_trace=lambda _: None,
        set_channel_trace_id=lambda *_: None,
    )

    coordinator.handle_recording_stopped(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={
                "recording_stopped": {
                    "recording_id": "recording-1",
                    "producer_stop_sequence_numbers": {"producer-1": 5},
                }
            },
        )
    )

    coordinator.set_max_producer_sequence("producer-1", 5)
    coordinator.finalize_closing_recordings()
    coordinator.finalize_closing_recordings()

    assert expected_trace_counts == [("recording-1", 0)]
    assert stopped_recordings == ["recording-1"]


def test_note_producer_sequence_ignores_non_cutoff_updates() -> None:
    loop = asyncio.new_event_loop()
    emitter = Emitter(loop=loop)
    enqueued_trace_ids: list[str] = []
    coordinator = TraceLifecycleCoordinator(
        emitter=emitter,
        enqueue_final_trace=lambda work: enqueued_trace_ids.append(work.trace_id),
        set_channel_trace_id=lambda *_: None,
    )
    channel = ChannelState(producer_id="producer-1", trace_id="trace-1")

    coordinator.register_trace("recording-1", "trace-1")
    coordinator.register_trace_metadata(
        TraceMetadataRegistrationRequest(
            trace_id="trace-1",
            metadata=TraceMetadataSnapshot(
                data_type=DataType.RGB_IMAGES.value,
                data_type_name="camera",
            ),
        )
    )
    coordinator.handle_trace_end(
        channel,
        MessageEnvelope(
            producer_id="producer-1",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-1",
                    "recording_id": "recording-1",
                }
            },
            sequence_number=5,
        ),
    )
    coordinator.handle_recording_stopped(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={
                "recording_stopped": {
                    "recording_id": "recording-1",
                    "producer_stop_sequence_numbers": {"producer-1": 5},
                }
            },
        )
    )

    coordinator.note_producer_sequence("producer-1", 4)

    assert enqueued_trace_ids == []


def test_note_producer_sequence_consumes_cutoff_watch_once() -> None:
    coordinator = _build_coordinator()
    finalize_calls: list[int] = []

    coordinator.handle_recording_stopped(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={
                "recording_stopped": {
                    "recording_id": "recording-1",
                    "producer_stop_sequence_numbers": {"producer-1": 5},
                }
            },
        )
    )
    coordinator.finalize_closing_recordings = lambda: finalize_calls.append(1)

    coordinator.note_producer_sequence("producer-1", 4)
    coordinator.note_producer_sequence("producer-1", 5)
    coordinator.note_producer_sequence("producer-1", 6)

    assert finalize_calls == [1]


TraceLifecycleCoordinator = trace_lifecycle_coordinator.TraceLifecycleCoordinator
