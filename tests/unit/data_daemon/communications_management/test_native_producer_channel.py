"""Unit tests for ``NativeProducerChannel``.

The PyO3 ``_native_producer`` cdylib is not built into the wheel yet, so we
stub it with the recording fake from ``conftest.py``. The fake's call log is
the only thing under test — we are validating that the Python adaptor
translates ProducerChannel-shaped lifecycle calls into the right native entry
points in the right order.
"""

from __future__ import annotations

import sys

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.producer import (
    native_producer_channel as adaptor,
)
from neuracore.data_daemon.communications_management.producer.native_producer_channel import (  # noqa: E501
    NativeProducerChannel,
)
from tests.unit.data_daemon.communications_management.conftest import (
    NativeProducerStub as _NativeStub,
)


def test_start_recording_session_emits_start_recording_then_start_trace(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )

    channel.start_recording_session()

    assert [call[0] for call in native_stub.calls] == [
        "start_recording",
        "start_trace",
    ]
    assert native_stub.calls[0][1] == ("rec-1",)
    start_trace_args = native_stub.calls[1][1]
    assert start_trace_args[0] == "rec-1"
    assert start_trace_args[2] == DataType.JOINT_POSITIONS.value
    assert channel.trace_id == start_trace_args[1]


def test_start_recording_session_requires_recording_id(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(data_type=DataType.JOINT_POSITIONS)
    with pytest.raises(ValueError):
        channel.start_recording_session()
    assert native_stub.calls == []


def test_send_frame_before_start_raises(native_stub: _NativeStub) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    with pytest.raises(RuntimeError):
        channel.send_frame(b"payload")
    assert native_stub.calls == []


def test_send_frame_publishes_frame_envelope_with_supplied_timestamp(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    channel.start_recording_session()

    channel.send_frame(b"payload", timestamp_ns=42)

    send_call = next(call for call in native_stub.calls if call[0] == "send_data")
    trace_id, payload, timestamp, timestamp_s = send_call[1]
    assert trace_id == channel.trace_id
    assert payload == b"payload"
    assert timestamp == 42
    assert timestamp_s is None


def test_send_frame_accepts_memoryview_and_bytearray(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    channel.start_recording_session()

    channel.send_frame(bytearray(b"abc"), timestamp_ns=1)
    channel.send_frame(memoryview(b"xyz"), timestamp_ns=2)

    sent_payloads = [call[1][1] for call in native_stub.calls if call[0] == "send_data"]
    assert sent_payloads == [b"abc", b"xyz"]


def test_announce_frame_resolution_replays_after_start(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(data_type=DataType.RGB_IMAGES, recording_id="rec-1")
    # Announce before the trace exists.
    channel.announce_frame_resolution(640, 480)
    assert all(call[0] != "open_frame_stream" for call in native_stub.calls)

    channel.start_recording_session()

    open_call = next(
        call for call in native_stub.calls if call[0] == "open_frame_stream"
    )
    assert open_call[1] == (channel.trace_id, 640, 480)


def test_announce_frame_resolution_is_idempotent_after_start(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(data_type=DataType.RGB_IMAGES, recording_id="rec-1")
    channel.start_recording_session()
    channel.announce_frame_resolution(320, 240)
    channel.announce_frame_resolution(320, 240)

    open_calls = [call for call in native_stub.calls if call[0] == "open_frame_stream"]
    assert len(open_calls) == 1


def test_announce_frame_resolution_rejects_non_positive_dimensions(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(data_type=DataType.RGB_IMAGES, recording_id="rec-1")
    with pytest.raises(ValueError):
        channel.announce_frame_resolution(0, 240)
    with pytest.raises(ValueError):
        channel.announce_frame_resolution(320, -1)


def test_cleanup_producer_channel_emits_end_trace_and_clears_state(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    channel.start_recording_session()
    trace_id = channel.trace_id

    channel.cleanup_producer_channel(stop_cutoff_sequence_number=0)

    assert any(call == ("end_trace", (trace_id,)) for call in native_stub.calls)
    assert channel.trace_id is None
    assert channel.recording_id is None


def test_cleanup_producer_channel_is_safe_without_active_trace(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )

    channel.cleanup_producer_channel(stop_cutoff_sequence_number=0)

    assert all(call[0] != "end_trace" for call in native_stub.calls)


def test_stop_producer_channel_does_not_emit_envelopes(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    channel.start_recording_session()
    pre_call_count = len(native_stub.calls)

    channel.stop_producer_channel()

    assert len(native_stub.calls) == pre_call_count


def test_legacy_sequence_compatibility_shims_return_zero(
    native_stub: _NativeStub,
) -> None:
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    assert channel.mark_recording_stop_requested() == 0
    assert channel.get_last_accepted_sequence_number() == 0
    assert channel.get_last_enqueued_sequence_number() == 0
    assert channel.get_last_sent_sequence_number() == 0
    assert channel.wait_until_sequence_sent(123) is True


def test_send_batched_joint_data_with_empty_items_is_noop(
    native_stub: _NativeStub,
) -> None:
    from neuracore.data_daemon.models import BatchedJointDataPayload

    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    payload = BatchedJointDataPayload(
        recording_id="rec-1",
        timestamp=1.0,
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=0,
        data_type=DataType.JOINT_POSITIONS,
        items=[],
    )
    channel.send_batched_joint_data(payload)
    assert native_stub.calls == []


def test_missing_native_module_surfaces_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure the loader retries (no cached module from another test).
    monkeypatch.setattr(adaptor, "_NATIVE_MODULE", None)
    monkeypatch.setitem(sys.modules, "neuracore.data_daemon._native_producer", None)
    channel = NativeProducerChannel(
        data_type=DataType.JOINT_POSITIONS, recording_id="rec-1"
    )
    with pytest.raises(RuntimeError, match="NCD_RUST_DAEMON"):
        channel.start_recording_session()
