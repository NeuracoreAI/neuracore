from __future__ import annotations

from neuracore.core.robot import Robot


class _FakeProducerChannel:
    def __init__(self, channel_id: str, cutoff_sequence_number: int) -> None:
        self.channel_id = channel_id
        self.cutoff_sequence_number = cutoff_sequence_number


class _FakeStream:
    def __init__(
        self,
        *,
        is_recording: bool,
        producer_channel: _FakeProducerChannel | None,
    ) -> None:
        self._is_recording = is_recording
        self._producer_channel = producer_channel
        self.discard_calls = 0
        self.prepare_calls = 0
        self.stop_calls: list[tuple[int, bool]] = []

    def is_recording(self) -> bool:
        return self._is_recording

    def get_producer_channel(self) -> _FakeProducerChannel | None:
        return self._producer_channel

    def discard_recording_state(self) -> None:
        self.discard_calls += 1
        self._is_recording = False

    def prepare_recording_stopped(self) -> tuple[_FakeProducerChannel, int]:
        self.prepare_calls += 1
        assert self._producer_channel is not None
        return self._producer_channel, self._producer_channel.cutoff_sequence_number

    def stop_recording(
        self,
        *,
        stop_cutoff_sequence_number: int,
        wait_for_producer_drain: bool = True,
    ) -> None:
        self.stop_calls.append((stop_cutoff_sequence_number, wait_for_producer_drain))
        self._is_recording = False
        self._producer_channel = None


def test_stop_all_streams_skips_streams_without_active_producer() -> None:
    robot = object.__new__(Robot)
    robot._daemon_recording_context = None
    robot._temp_dir = None
    active_stream = _FakeStream(
        is_recording=True,
        producer_channel=_FakeProducerChannel("channel-1", 42),
    )
    missing_producer_stream = _FakeStream(
        is_recording=True,
        producer_channel=None,
    )
    inactive_stream = _FakeStream(
        is_recording=False,
        producer_channel=None,
    )
    inactive_with_producer_stream = _FakeStream(
        is_recording=False,
        producer_channel=_FakeProducerChannel("channel-2", 7),
    )
    robot._data_streams = {
        "active": active_stream,
        "missing": missing_producer_stream,
        "inactive": inactive_stream,
        "inactive-with-producer": inactive_with_producer_stream,
    }

    stop_sequence_numbers = robot._stop_all_streams(wait_for_producer_drain=False)

    assert stop_sequence_numbers == {"channel-1": 42, "channel-2": 7}
    assert active_stream.prepare_calls == 1
    assert active_stream.stop_calls == [(42, False)]
    assert missing_producer_stream.discard_calls == 1
    assert missing_producer_stream.prepare_calls == 0
    assert missing_producer_stream.stop_calls == []
    assert inactive_stream.discard_calls == 1
    assert inactive_with_producer_stream.prepare_calls == 1
    assert inactive_with_producer_stream.stop_calls == [(7, False)]
