import time
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from neuracore.data_daemon.communications_management.consumer.models import ChannelState
from neuracore.data_daemon.communications_management.consumer.spool_worker import (
    _SpoolShard,
)


def _build_shard(
    *, handle_descriptor: Callable[..., object] | Mock | None = None
) -> _SpoolShard:
    shared_slot_handler = Mock()
    shared_slot_handler.handle_descriptor = handle_descriptor or Mock()
    completion_worker = Mock()
    return _SpoolShard(
        chunk_spool=Mock(),
        shared_slot_handler=shared_slot_handler,
        completion_worker=completion_worker,
        acquire_spool_admission=lambda: None,
        release_spool_admission=lambda: None,
        should_drop_recording_data=lambda _: False,
        mark_sequence_completed=lambda _: None,
        register_trace=lambda *_: None,
        register_trace_metadata=lambda *_: None,
        get_trace_recording=lambda _: None,
        set_channel_trace_id=lambda *_: None,
        shard_index=0,
    )


def test_enqueue_raises_when_shard_thread_is_not_running() -> None:
    shard = _build_shard()
    channel = ChannelState(producer_id="producer-1")

    shard.close()

    with pytest.raises(RuntimeError, match="Daemon spool shard is not running"):
        shard.enqueue(channel, {})


def test_enqueue_raises_wrapped_error_after_worker_failure() -> None:
    shard = _build_shard(
        handle_descriptor=Mock(side_effect=RuntimeError("boom")),
    )
    channel = ChannelState(producer_id="producer-1")

    shard.enqueue(channel, {})

    deadline = time.monotonic() + 1.0
    while shard._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not shard._thread.is_alive()

    with pytest.raises(RuntimeError, match="Daemon spool shard failed") as excinfo:
        shard.enqueue(channel, {})

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "boom"
