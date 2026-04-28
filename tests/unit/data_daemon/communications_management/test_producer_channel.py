from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)


class _FakeSharedSlotTransport:
    def __init__(self) -> None:
        self.wait_until_payload_handed_off_calls: list[float] = []
        self.wait_until_drained_calls: list[float] = []

    def wait_until_payload_handed_off(self, timeout_s: float = 30.0) -> None:
        self.wait_until_payload_handed_off_calls.append(timeout_s)

    def wait_until_drained(self, timeout_s: float = 30.0) -> None:
        self.wait_until_drained_calls.append(timeout_s)


def test_cleanup_producer_channel_wait_false_skips_slot_drain() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    wait_calls: list[int] = []
    sequence_numbers = iter([41, 42])
    end_trace_calls: list[str] = []

    channel._shared_slot_transport = transport
    channel.get_last_enqueued_sequence_number = lambda: next(sequence_numbers)
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel.end_trace = lambda: end_trace_calls.append("end")

    ProducerChannel.cleanup_producer_channel(channel, wait_for_slot_drain=False)

    assert transport.wait_until_payload_handed_off_calls == [30.0]
    assert transport.wait_until_drained_calls == []
    assert end_trace_calls == ["end"]
    assert wait_calls == [41, 42]


def test_cleanup_producer_channel_wait_true_drains_shared_slots() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeSharedSlotTransport()
    wait_calls: list[int] = []
    sequence_numbers = iter([99, 100])

    channel._shared_slot_transport = transport
    channel.get_last_enqueued_sequence_number = lambda: next(sequence_numbers)
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel.end_trace = lambda: None

    ProducerChannel.cleanup_producer_channel(channel, wait_for_slot_drain=True)

    assert transport.wait_until_payload_handed_off_calls == [30.0]
    assert transport.wait_until_drained_calls == [30.0]
    assert wait_calls == [99, 100]
