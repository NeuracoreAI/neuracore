import time
from unittest.mock import Mock

import pytest

from neuracore.data_daemon.communications_management.consumer.models import (
    ChannelState,
    DecodedFrameWork,
)
from neuracore.data_daemon.communications_management.consumer.spool_worker import (
    _SpoolShard,
)


def _build_shard(
    *,
    chunk_spool: Mock | None = None,
    should_drop_recording_data=lambda _: False,
) -> _SpoolShard:
    completion_worker = Mock()
    return _SpoolShard(
        chunk_spool=chunk_spool or Mock(),
        completion_worker=completion_worker,
        acquire_spool_admission=lambda: None,
        release_spool_admission=lambda: None,
        should_drop_recording_data=should_drop_recording_data,
        mark_sequence_completed=lambda _: None,
        register_trace=lambda *_: None,
        register_trace_metadata=lambda *_: None,
        get_trace_recording=lambda _: None,
        set_channel_trace_id=lambda *_: None,
        shard_index=0,
    )


def _frame_work(channel: ChannelState, metadata: dict) -> DecodedFrameWork:
    return DecodedFrameWork(
        channel=channel,
        sequence_id=1,
        metadata=metadata,
        chunk=b"chunk",
    )


def test_enqueue_raises_when_shard_thread_is_not_running() -> None:
    shard = _build_shard()
    channel = ChannelState(producer_id="producer-1")

    shard.close()

    with pytest.raises(RuntimeError, match="Daemon spool shard is not running"):
        shard.enqueue(_frame_work(channel, {"trace_id": "t"}))


def test_enqueue_raises_wrapped_error_after_worker_failure() -> None:
    # chunk_spool.append raises so the worker loop records the error and stops.
    chunk_spool = Mock()
    chunk_spool.append = Mock(side_effect=RuntimeError("boom"))
    shard = _build_shard(chunk_spool=chunk_spool)
    channel = ChannelState(producer_id="producer-1")

    shard.enqueue(
        _frame_work(
            channel,
            {"trace_id": "t", "chunk_index": 0, "total_chunks": 1},
        )
    )

    deadline = time.monotonic() + 1.0
    while shard._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not shard._thread.is_alive()

    with pytest.raises(RuntimeError, match="Daemon spool shard failed") as excinfo:
        shard.enqueue(_frame_work(channel, {"trace_id": "t"}))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "boom"
