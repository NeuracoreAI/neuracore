from __future__ import annotations

import asyncio
import base64
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


class _FakeEmitter:
    def __init__(self) -> None:
        self._listeners: dict[Any, list[Callable[..., Any]]] = {}
        self.emitted: list[tuple[Any, tuple[Any, ...]]] = []

    def on(self, event: Any, fn: Callable[..., Any]) -> None:
        self._listeners.setdefault(event, []).append(fn)

    def remove_listener(self, event: Any, fn: Callable[..., Any]) -> None:
        fns = self._listeners.get(event, [])
        self._listeners[event] = [x for x in fns if x is not fn]

    def emit(self, event: Any, *args: Any) -> None:
        self.emitted.append((event, args))
        for fn in list(self._listeners.get(event, [])):
            res = fn(*args)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)


class _FakeVideoTrace:
    def __init__(self, *, output_dir: Path, **_: Any) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.payloads: list[bytes] = []
        self.finished = False
        self.raise_on_call_index: int | None = None

    def add_payload(self, payload: bytes) -> None:
        if (
            self.raise_on_call_index is not None
            and len(self.payloads) == self.raise_on_call_index
        ):
            raise ValueError("corrupt frame payload")
        self.payloads.append(payload)

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        (self.output_dir / "lossy.mp4").write_bytes(b"mp4")
        (self.output_dir / "lossless.mp4").write_bytes(b"mp4")
        (self.output_dir / "trace.json").write_text("[]", encoding="utf-8")


class _FakeJsonTrace:
    def __init__(self, *, output_dir: Path, **_: Any) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict[str, Any]] = []
        self.finished = False

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        (self.output_dir / "trace.json").write_text(
            json.dumps(self.frames, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )


class _FakeFilesystem:
    def __init__(self, root: Path) -> None:
        self.root = root

    def trace_dir_for(self, trace_key: Any) -> Path:
        return (
            self.root
            / f"rec_{trace_key.recording_id}"
            / f"trace_{trace_key.trace_id}"
            / str(trace_key.data_type)
        )

    def trace_bytes_on_disk(self, trace_key: Any) -> int:
        trace_dir = self.trace_dir_for(trace_key)
        total = 0
        if trace_dir.exists():
            for p in trace_dir.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        return total


class _FakeStorageBudget:
    def __init__(self, over_limit: bool = False) -> None:
        self._over_limit = over_limit

    def is_over_limit(self) -> bool:
        return self._over_limit


def _envelope_line(payload: bytes) -> bytes:
    return (
        json.dumps({"data": base64.b64encode(payload).decode("ascii")}) + "\n"
    ).encode("utf-8")


def _write_batch_file(path: Path, payloads: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for p in payloads:
            f.write(_envelope_line(p))


@dataclass(frozen=True)
class _LocalTraceKey:
    recording_id: str
    data_type: str
    trace_id: str


@dataclass(frozen=True)
class _LocalBatchJob:
    trace_key: _LocalTraceKey
    batch_path: Path
    trace_done: bool


@pytest.fixture
def fake_emitter() -> _FakeEmitter:
    return _FakeEmitter()


@pytest.fixture
def patched_modules(
    monkeypatch: pytest.MonkeyPatch,
    fake_emitter: _FakeEmitter,
):
    worker_module = importlib.import_module(
        "neuracore.data_daemon.recording_encoding_disk_manager.workers."
        "batch_encoder_worker"
    )
    manager_module = importlib.import_module(
        "neuracore.data_daemon.recording_encoding_disk_manager.lifecycle."
        "encoder_manager"
    )

    monkeypatch.setattr(
        worker_module,
        "get_emitter",
        lambda: fake_emitter,
        raising=True,
    )
    monkeypatch.setattr(
        manager_module,
        "get_emitter",
        lambda: fake_emitter,
        raising=True,
    )

    monkeypatch.setattr(
        manager_module,
        "get_content_type",
        lambda dt: "RGB" if str(dt).lower() in {"rgb", "video", "image"} else "JSON",
        raising=True,
    )

    monkeypatch.setattr(
        manager_module,
        "VideoTrace",
        _FakeVideoTrace,
        raising=True,
    )
    monkeypatch.setattr(
        manager_module,
        "JsonTrace",
        _FakeJsonTrace,
        raising=True,
    )

    monkeypatch.setattr(
        worker_module,
        "VideoTrace",
        _FakeVideoTrace,
        raising=True,
    )
    monkeypatch.setattr(
        worker_module,
        "JsonTrace",
        _FakeJsonTrace,
        raising=True,
    )

    return worker_module, manager_module


@pytest.fixture
def make_worker(tmp_path: Path, patched_modules):
    worker_module, manager_module = patched_modules

    filesystem = _FakeFilesystem(tmp_path / "out")
    budget = _FakeStorageBudget(over_limit=False)

    aborted: list[Any] = []

    def abort_trace(key: Any) -> None:
        aborted.append(key)

    manager = manager_module._EncoderManager(
        filesystem=filesystem, abort_trace=abort_trace
    )
    worker = worker_module._BatchEncoderWorker(
        filesystem=filesystem,
        encoder_manager=manager,
        storage_budget=budget,
        abort_trace=abort_trace,
    )

    return worker, manager, filesystem, budget, aborted, worker_module, manager_module


@pytest.mark.asyncio
async def test_video_trace_batch_feeds_payloads_in_order_and_finalises_on_trace_done(
    make_worker,
    fake_emitter: _FakeEmitter,
    tmp_path: Path,
) -> None:
    worker, manager, _, _, aborted, worker_module, _ = make_worker

    key = _LocalTraceKey(recording_id="r1", data_type="rgb", trace_id="t1")
    batch_path = tmp_path / "batches" / "batch_0.ndjson"

    meta = json.dumps(
        {"width": 4, "height": 3, "timestamp": 1.0},
        separators=(",", ":"),
    ).encode("utf-8")
    frame1 = b"\x00" * 36
    frame2 = b"\x01" * 36
    _write_batch_file(batch_path, [meta, frame1, frame2])

    enc = manager.safe_get_encoder(key)
    assert isinstance(enc, _FakeVideoTrace)

    job = _LocalBatchJob(trace_key=key, batch_path=batch_path, trace_done=True)
    await worker._on_batch_ready(job)

    assert aborted == []
    assert not batch_path.exists()
    assert enc.payloads == [meta, frame1, frame2]

    emitted = [
        e for e in fake_emitter.emitted if e[0] == worker_module.Emitter.TRACE_WRITTEN
    ]
    assert len(emitted) == 1
    _, args = emitted[0]
    assert args[0] == "t1"
    assert args[1] == "r1"
    assert isinstance(args[2], int)


@pytest.mark.asyncio
async def test_finalises_only_after_all_pending_batches_complete(
    make_worker, fake_emitter: _FakeEmitter, tmp_path: Path
) -> None:
    worker, _, _, _, aborted, worker_module, _ = make_worker

    key = _LocalTraceKey(recording_id="r2", data_type="rgb", trace_id="t2")

    batch0 = tmp_path / "batches" / "batch_0.ndjson"
    batch1 = tmp_path / "batches" / "batch_1.ndjson"

    meta = json.dumps({"width": 2, "height": 2, "timestamp": 1.0}).encode("utf-8")
    frame = b"\x00" * 12
    _write_batch_file(batch0, [meta, frame])
    _write_batch_file(batch1, [frame])

    job0 = _LocalBatchJob(trace_key=key, batch_path=batch0, trace_done=False)
    job1 = _LocalBatchJob(trace_key=key, batch_path=batch1, trace_done=True)

    await asyncio.gather(worker._on_batch_ready(job0), worker._on_batch_ready(job1))

    assert aborted == []

    emitted = [
        e for e in fake_emitter.emitted if e[0] == worker_module.Emitter.TRACE_WRITTEN
    ]
    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_corrupt_payload_does_not_crash_logs_and_aborts_trace(
    make_worker, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    worker, manager, _, _, aborted, _, _ = make_worker

    key = _LocalTraceKey(recording_id="r3", data_type="rgb", trace_id="t3")
    batch_path = tmp_path / "batches" / "batch_0.ndjson"

    meta = json.dumps({"width": 2, "height": 2, "timestamp": 1.0}).encode("utf-8")
    good = b"\x00" * 12
    bad = b"\xff" * 12
    _write_batch_file(batch_path, [meta, good, bad])

    enc = manager.safe_get_encoder(key)
    assert isinstance(enc, _FakeVideoTrace)
    enc.raise_on_call_index = 2

    job = _LocalBatchJob(trace_key=key, batch_path=batch_path, trace_done=True)

    with caplog.at_level("ERROR"):
        await worker._on_batch_ready(job)

    assert not batch_path.exists()
    assert aborted and aborted[-1] == key
    assert any("Failed to process batch" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_multiple_traces_concurrently_each_emit_trace_written(
    make_worker, fake_emitter: _FakeEmitter, tmp_path: Path
) -> None:
    worker, _, _, _, aborted, worker_module, _ = make_worker

    key_a = _LocalTraceKey(recording_id="ra", data_type="rgb", trace_id="ta")
    key_b = _LocalTraceKey(recording_id="rb", data_type="json", trace_id="tb")

    batch_a = tmp_path / "batches" / "batch_0a.ndjson"
    batch_b = tmp_path / "batches" / "batch_0b.ndjson"

    meta = json.dumps({"width": 2, "height": 2, "timestamp": 1.0}).encode("utf-8")
    frame = b"\x00" * 12
    _write_batch_file(batch_a, [meta, frame])

    payload_json = json.dumps({"i": 1}).encode("utf-8")
    _write_batch_file(batch_b, [payload_json])

    job_a = _LocalBatchJob(trace_key=key_a, batch_path=batch_a, trace_done=True)
    job_b = _LocalBatchJob(trace_key=key_b, batch_path=batch_b, trace_done=True)

    await asyncio.gather(worker._on_batch_ready(job_a), worker._on_batch_ready(job_b))

    assert aborted == []

    emitted = [
        e for e in fake_emitter.emitted if e[0] == worker_module.Emitter.TRACE_WRITTEN
    ]
    assert len(emitted) == 2
    got = {(args[0], args[1]) for _, args in emitted}
    assert got == {("ta", "ra"), ("tb", "rb")}


def test_shutdown_finalises_remaining_encoders_and_emits_trace_written(
    make_worker, fake_emitter: _FakeEmitter
) -> None:
    worker, manager, _, _, aborted, worker_module, _ = make_worker

    key = _LocalTraceKey(recording_id="rs", data_type="rgb", trace_id="ts")
    enc = manager.safe_get_encoder(key)
    assert isinstance(enc, _FakeVideoTrace)
    assert not enc.finished

    worker.shutdown()

    assert enc.finished
    assert aborted == []

    emitted = [
        e for e in fake_emitter.emitted if e[0] == worker_module.Emitter.TRACE_WRITTEN
    ]
    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_trace_aborted_mid_batch_process_does_not_emit_trace_written_and_cleanup(
    make_worker,
    fake_emitter: _FakeEmitter,
    tmp_path: Path,
) -> None:
    worker, manager, _, _, aborted, worker_module, _ = make_worker

    key = _LocalTraceKey(recording_id="r_abort", data_type="rgb", trace_id="t_abort")
    batch_path = tmp_path / "batches" / "batch_0.ndjson"

    meta = json.dumps({"width": 2, "height": 2, "timestamp": 1.0}).encode("utf-8")
    frame = b"\x00" * 12
    _write_batch_file(batch_path, [meta, frame, frame, frame])

    enc = manager.safe_get_encoder(key)
    assert isinstance(enc, _FakeVideoTrace)

    original_add_payload = enc.add_payload
    first_frame_seen = False

    def slow_add_payload(payload: bytes) -> None:
        nonlocal first_frame_seen
        if payload == frame and not first_frame_seen:
            first_frame_seen = True
            raise RuntimeError("simulate corrupt frame mid-batch")
        original_add_payload(payload)

    enc.add_payload = slow_add_payload  # type: ignore[method-assign]

    job = _LocalBatchJob(trace_key=key, batch_path=batch_path, trace_done=True)

    await worker._on_batch_ready(job)

    assert not batch_path.exists()
    assert aborted and aborted[-1] == key

    emitted = [
        e for e in fake_emitter.emitted if e[0] == worker_module.Emitter.TRACE_WRITTEN
    ]
    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_in_flight_count_balances_on_success_and_failure(
    make_worker,
    tmp_path: Path,
) -> None:
    worker, manager, _, _, aborted, _, _ = make_worker

    key_ok = _LocalTraceKey(recording_id="r_ok", data_type="rgb", trace_id="t_ok")
    p_ok = tmp_path / "batches" / "batch_0_ok.ndjson"
    meta = json.dumps({"width": 2, "height": 2, "timestamp": 1.0}).encode("utf-8")
    frame = b"\x00" * 12
    _write_batch_file(p_ok, [meta, frame])

    job_ok = _LocalBatchJob(trace_key=key_ok, batch_path=p_ok, trace_done=True)

    assert worker.in_flight_count == 0
    t_ok = asyncio.create_task(worker._on_batch_ready(job_ok))
    await asyncio.sleep(0)
    await t_ok
    assert worker.in_flight_count == 0
    assert aborted == []

    key_bad = _LocalTraceKey(recording_id="r_bad", data_type="rgb", trace_id="t_bad")
    p_bad = tmp_path / "batches" / "batch_0_bad.ndjson"
    _write_batch_file(p_bad, [meta, frame, frame])

    enc = manager.safe_get_encoder(key_bad)
    assert isinstance(enc, _FakeVideoTrace)
    enc.raise_on_call_index = 1

    job_bad = _LocalBatchJob(trace_key=key_bad, batch_path=p_bad, trace_done=True)

    assert worker.in_flight_count == 0
    t_bad = asyncio.create_task(worker._on_batch_ready(job_bad))
    await asyncio.sleep(0)
    await t_bad
    assert worker.in_flight_count == 0
    assert aborted and aborted[-1] == key_bad
