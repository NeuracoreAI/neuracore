import pytest

from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.communications_management.shared_transport.registry import (
    SharedSlotUnhealthyError,
)
from neuracore.data_daemon.models import DataType


class _FakeSharedSlotTransport:
    def __init__(self) -> None:
        self.wait_until_payload_handed_off_calls: list[float] = []
        self.wait_until_drained_calls: list[float] = []
        self.finish_recording_session_calls = 0
        self.last_payload_sequence_number = 0

    def get_last_payload_sequence_number(self) -> int:
        return self.last_payload_sequence_number

    def wait_until_payload_handed_off(
        self,
        timeout_s: float = 30.0,
        max_sequence_number: int | None = None,
    ) -> None:
        self.wait_until_payload_handed_off_calls.append(timeout_s)
        self.last_payload_sequence_number = (
            0 if max_sequence_number is None else max_sequence_number
        )

    def wait_until_drained(
        self,
        timeout_s: float = 30.0,
        max_sequence_number: int | None = None,
    ) -> None:
        self.wait_until_drained_calls.append(timeout_s)
        self.last_payload_sequence_number = (
            0 if max_sequence_number is None else max_sequence_number
        )

    def finish_recording_session(self) -> None:
        self.finish_recording_session_calls += 1


def test_cleanup_producer_channel_wait_false_skips_slot_drain() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    wait_calls: list[int] = []
    end_trace_calls: list[str] = []

    channel._shared_slot_transport = transport
    channel.get_last_enqueued_sequence_number = lambda: 41
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel.end_trace = lambda: end_trace_calls.append("end")
    transport.last_payload_sequence_number = 41

    ProducerChannel.cleanup_producer_channel(
        channel,
        stop_cutoff_sequence_number=41,
        wait_for_slot_drain=False,
    )

    assert transport.wait_until_payload_handed_off_calls == [30.0]
    assert transport.wait_until_drained_calls == []
    assert transport.finish_recording_session_calls == 1
    assert end_trace_calls == ["end"]
    assert wait_calls == [41]


def test_cleanup_producer_channel_wait_true_drains_shared_slots() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    wait_calls: list[int] = []
    channel._shared_slot_transport = transport
    channel.get_last_enqueued_sequence_number = lambda: 99
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel.end_trace = lambda: None
    transport.last_payload_sequence_number = 99

    ProducerChannel.cleanup_producer_channel(
        channel,
        stop_cutoff_sequence_number=99,
        wait_for_slot_drain=True,
    )

    assert transport.wait_until_payload_handed_off_calls == [30.0]
    assert transport.wait_until_drained_calls == [30.0]
    assert transport.finish_recording_session_calls == 1
    assert wait_calls == [99]


def test_cleanup_producer_channel_raises_when_descriptor_cutoff_not_sent() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    end_trace_calls: list[str] = []

    channel._shared_slot_transport = transport
    channel.get_last_enqueued_sequence_number = lambda: 44
    channel.wait_until_sequence_sent = lambda sequence_number: False
    channel.end_trace = lambda: end_trace_calls.append("end")
    transport.last_payload_sequence_number = 44

    with pytest.raises(
        RuntimeError,
        match="Failed to send queued recording data up to stop cutoff before cleanup",
    ):
        ProducerChannel.cleanup_producer_channel(
            channel,
            stop_cutoff_sequence_number=44,
            wait_for_slot_drain=True,
        )

    assert transport.wait_until_payload_handed_off_calls == [30.0]
    assert transport.wait_until_drained_calls == []
    assert transport.finish_recording_session_calls == 0
    assert end_trace_calls == []


def test_stop_producer_channel_wait_false_skips_slot_drain() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    wait_calls: list[int] = []
    close_calls: list[str] = []

    channel._shared_slot_transport = transport
    channel._stop_heartbeat_service = lambda: close_calls.append("heartbeat")
    channel.get_last_enqueued_sequence_number = lambda: 12
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel._close_shared_slot_transport = lambda: close_calls.append("transport")
    channel._stop_message_sender = lambda: close_calls.append("sender")
    channel._comm = type(
        "_Comm", (), {"cleanup_producer": lambda self: close_calls.append("comm")}
    )()

    ProducerChannel.stop_producer_channel(channel, wait_for_slot_drain=False)

    assert wait_calls == [12]
    assert transport.wait_until_drained_calls == []
    assert close_calls == ["heartbeat", "transport", "sender", "comm"]


def test_stop_producer_channel_wait_true_drains_shared_slots() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    wait_calls: list[int] = []

    channel._shared_slot_transport = transport
    channel._stop_heartbeat_service = lambda: None
    channel.get_last_enqueued_sequence_number = lambda: 13
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel._close_shared_slot_transport = lambda: None
    channel._stop_message_sender = lambda: None
    channel._comm = type("_Comm", (), {"cleanup_producer": lambda self: None})()

    ProducerChannel.stop_producer_channel(channel, wait_for_slot_drain=True)

    assert wait_calls == [13]
    assert transport.wait_until_drained_calls == [30.0]


def test_stop_producer_channel_raises_when_cutoff_not_sent_and_still_cleans_up() -> (
    None
):
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    close_calls: list[str] = []

    channel._shared_slot_transport = transport
    channel._stop_heartbeat_service = lambda: close_calls.append("heartbeat")
    channel.get_last_enqueued_sequence_number = lambda: 21
    channel.wait_until_sequence_sent = lambda sequence_number: False
    channel._close_shared_slot_transport = lambda: close_calls.append("transport")
    channel._stop_message_sender = lambda: close_calls.append("sender")
    channel._comm = type(
        "_Comm", (), {"cleanup_producer": lambda self: close_calls.append("comm")}
    )()

    with pytest.raises(
        RuntimeError,
        match="Failed to send all enqueued messages before stopping producer channel",
    ):
        ProducerChannel.stop_producer_channel(channel, wait_for_slot_drain=True)

    assert transport.wait_until_drained_calls == []
    assert close_calls == ["heartbeat", "transport", "sender", "comm"]


def test_stop_producer_channel_swallows_cutoff_failure_after_sender_error() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    close_calls: list[str] = []

    channel._shared_slot_transport = transport
    channel._stop_heartbeat_service = lambda: close_calls.append("heartbeat")
    channel.get_last_enqueued_sequence_number = lambda: 22
    channel.wait_until_sequence_sent = lambda sequence_number: False
    channel._close_shared_slot_transport = lambda: close_calls.append("transport")
    channel._stop_message_sender = lambda: close_calls.append("sender")
    channel._comm = type(
        "_Comm", (), {"cleanup_producer": lambda self: close_calls.append("comm")}
    )()
    channel._message_sender = type(
        "_Sender",
        (),
        {"get_error": lambda self: RuntimeError("boom")},
    )()

    ProducerChannel.stop_producer_channel(channel, wait_for_slot_drain=True)

    assert transport.wait_until_drained_calls == []
    assert close_calls == ["heartbeat", "transport", "sender", "comm"]


def test_end_trace_waits_before_clearing_trace_state() -> None:
    channel = object.__new__(ProducerChannel)
    send_calls: list[tuple[object, dict]] = []
    wait_calls: list[int] = []

    channel.trace_id = "trace-1"
    channel.recording_id = "recording-1"
    channel._send = lambda command, payload=None: (
        send_calls.append((command, payload or {})) or 55
    )
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )

    ProducerChannel.end_trace(channel)

    assert len(send_calls) == 1
    assert wait_calls == [55]
    assert channel.trace_id is None
    assert channel.recording_id is None


def test_end_trace_keeps_trace_state_when_trace_end_not_sent() -> None:
    channel = object.__new__(ProducerChannel)

    channel.trace_id = "trace-1"
    channel.recording_id = "recording-1"
    channel._send = lambda command, payload=None: 56
    channel.wait_until_sequence_sent = lambda sequence_number: False

    with pytest.raises(
        RuntimeError,
        match="Failed to send TRACE_END before ending trace",
    ):
        ProducerChannel.end_trace(channel)

    assert channel.trace_id == "trace-1"
    assert channel.recording_id == "recording-1"


def _shared_slot_channel_for_send_tests() -> ProducerChannel:
    channel = object.__new__(ProducerChannel)
    channel._stop_cutoff_sequence_number = None
    channel.trace_id = "trace-1"
    channel.recording_id = "recording-1"
    channel.chunk_size = 1024
    channel._use_shared_slot_transport = True
    return channel


def test_send_data_parts_resets_shared_slot_transport_and_retries_once() -> None:
    channel = _shared_slot_channel_for_send_tests()
    send_calls: list[int] = []
    reset_calls: list[str] = []

    def fake_send(*_args, **_kwargs) -> None:
        send_calls.append(len(send_calls))
        if len(send_calls) == 1:
            raise SharedSlotUnhealthyError("stale shared slot")

    channel._send_data_parts_shared_slots = fake_send
    channel._ping_daemon_for_shared_slot_recovery = lambda timeout_s=2.0: True
    channel._reset_shared_slot_transport_for_recovery = lambda: reset_calls.append(
        "reset"
    )

    ProducerChannel.send_data_parts(
        channel,
        (b"frame",),
        data_type=DataType.RGB_IMAGES,
        robot_instance=0,
        data_type_name="camera",
        robot_name="robot",
        dataset_name="dataset",
    )

    assert send_calls == [0, 1]
    assert reset_calls == ["reset"]


def _raise_stale_shared_slot(*_args, **_kwargs) -> None:
    raise SharedSlotUnhealthyError("stale shared slot")


def test_send_data_parts_stops_shared_slot_logging_when_ping_fails() -> None:
    channel = _shared_slot_channel_for_send_tests()
    stop_calls: list[str] = []
    reset_calls: list[str] = []

    channel._send_data_parts_shared_slots = _raise_stale_shared_slot
    channel._ping_daemon_for_shared_slot_recovery = lambda timeout_s=2.0: False
    channel._reset_shared_slot_transport_for_recovery = lambda: reset_calls.append(
        "reset"
    )
    channel._stop_shared_slot_logging_after_failure = lambda: stop_calls.append("stop")

    with pytest.raises(RuntimeError, match="daemon did not respond"):
        ProducerChannel.send_data_parts(
            channel,
            (b"frame",),
            data_type=DataType.RGB_IMAGES,
            robot_instance=0,
            data_type_name="camera",
            robot_name="robot",
            dataset_name="dataset",
        )

    assert stop_calls == ["stop"]
    assert reset_calls == []


def test_send_data_parts_stops_shared_slot_logging_when_retry_fails() -> None:
    channel = _shared_slot_channel_for_send_tests()
    stop_calls: list[str] = []
    reset_calls: list[str] = []

    channel._send_data_parts_shared_slots = _raise_stale_shared_slot
    channel._ping_daemon_for_shared_slot_recovery = lambda timeout_s=2.0: True
    channel._reset_shared_slot_transport_for_recovery = lambda: reset_calls.append(
        "reset"
    )
    channel._stop_shared_slot_logging_after_failure = lambda: stop_calls.append("stop")

    with pytest.raises(RuntimeError, match="remained unhealthy"):
        ProducerChannel.send_data_parts(
            channel,
            (b"frame",),
            data_type=DataType.RGB_IMAGES,
            robot_instance=0,
            data_type_name="camera",
            robot_name="robot",
            dataset_name="dataset",
        )

    assert reset_calls == ["reset"]
    assert stop_calls == ["stop"]
