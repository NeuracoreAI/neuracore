import asyncio

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.consumer.models import (
    ChannelState,
    CompletedChannelMessage,
    TraceRecordingLookupRequest,
)
from neuracore.data_daemon.communications_management.consumer.trace_lifecycle_coordinator import (
    TraceLifecycleCoordinator,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import CommandType, MessageEnvelope


def _build_coordinator() -> TraceLifecycleCoordinator:
    loop = asyncio.new_event_loop()
    return TraceLifecycleCoordinator(
        emitter=Emitter(loop=loop),
        enqueue_final_trace=lambda _: None,
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


def test_ensure_result_trace_registered_uses_completed_result_data_type_fallback() -> None:
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
