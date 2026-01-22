from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.models import CompleteMessage


class FakeEmitter:
    def __init__(self) -> None:
        self._handlers: dict[Any, list[Callable[..., Any]]] = {}

    def on(self, event: Any, fn: Callable[..., Any] | None = None):
        if fn is None:

            def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
                self._handlers.setdefault(event, []).append(f)
                return f

            return decorator

        self._handlers.setdefault(event, []).append(fn)
        return fn

    def emit(self, event: Any, *args: Any, **kwargs: Any) -> None:
        for fn in list(self._handlers.get(event, [])):
            fn(*args, **kwargs)


class FakeVideoTrace:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._w: int | None = None
        self._h: int | None = None
        self._frames = 0
        self._metas: list[dict[str, Any]] = []

    def add_payload(self, payload: bytes) -> None:
        try:
            obj = json.loads(payload.decode("utf-8"))
        except Exception:
            obj = None

        if isinstance(obj, dict):
            w = obj.get("width")
            h = obj.get("height")
            if isinstance(w, int) and isinstance(h, int):
                if self._w is None and self._h is None:
                    self._w = w
                    self._h = h
            self._metas.append(obj)
            return

        if self._w is None or self._h is None:
            raise RuntimeError("VideoTrace needs width/height before frames.")

        expected = self._w * self._h * 3
        if len(payload) != expected:
            raise ValueError("Unexpected frame size.")

        self._frames += 1

    def finish(self) -> None:
        (self.output_dir / "lossy.mp4").write_bytes(b"lossy")
        (self.output_dir / "lossless.mp4").write_bytes(b"lossless")
        (self.output_dir / "trace.json").write_text(
            json.dumps(self._metas, separators=(",", ":")),
            encoding="utf-8",
        )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _make_rgb24_frame_bytes(frame_index: int, width: int, height: int) -> bytes:
    buf = bytearray(width * height * 3)
    shift = (frame_index * 7) % 256
    i = 0
    for y in range(height):
        for x in range(width):
            buf[i] = (x + shift) & 0xFF
            buf[i + 1] = (y + shift) & 0xFF
            buf[i + 2] = (x + y + shift) & 0xFF
            i += 3
    return bytes(buf)


def _wait_for(pred: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture
def fake_emitter(monkeypatch: pytest.MonkeyPatch) -> FakeEmitter:
    from neuracore.data_daemon import event_emitter as event_emitter_module
    from neuracore.data_daemon.recording_encoding_disk_manager import (
        recording_disk_manager as rdm_module,
    )
    from neuracore.data_daemon.recording_encoding_disk_manager.lifecycle import (
        encoder_manager as encoder_manager_module,
    )
    from neuracore.data_daemon.recording_encoding_disk_manager.lifecycle import (
        trace_controller as trace_controller_module,
    )
    from neuracore.data_daemon.recording_encoding_disk_manager.workers import (
        batch_encoder_worker as batch_encoder_worker_module,
    )
    from neuracore.data_daemon.recording_encoding_disk_manager.workers import (
        raw_batch_writer as raw_batch_writer_module,
    )

    fe = FakeEmitter()

    monkeypatch.setattr(event_emitter_module, "emitter", fe, raising=False)
    monkeypatch.setattr(rdm_module, "emitter", fe, raising=False)
    monkeypatch.setattr(raw_batch_writer_module, "emitter", fe, raising=False)
    monkeypatch.setattr(trace_controller_module, "emitter", fe, raising=False)
    monkeypatch.setattr(batch_encoder_worker_module, "emitter", fe, raising=False)
    monkeypatch.setattr(encoder_manager_module, "emitter", fe, raising=False)

    return fe


@pytest.fixture
def rdm_module(monkeypatch: pytest.MonkeyPatch):
    from neuracore.data_daemon.recording_encoding_disk_manager import (
        recording_disk_manager as rdm_module,
    )
    from neuracore.data_daemon.recording_encoding_disk_manager.lifecycle import (
        encoder_manager as encoder_manager_module,
    )

    # Encoder creation is owned by EncoderManager now, so patch it there.
    monkeypatch.setattr(
        encoder_manager_module,
        "VideoTrace",
        FakeVideoTrace,
        raising=True,
    )

    return rdm_module


@pytest.fixture
def rdm_factory(
    tmp_path: Path,
    fake_emitter: FakeEmitter,
    rdm_module,
    request: pytest.FixtureRequest,
):
    def _make(*, storage_limit: int | None, flush_bytes: int = 1):
        recordings_root = tmp_path / "recordings"

        rdm = rdm_module.RecordingDiskManager(
            flush_bytes=flush_bytes,
            storage_limit_bytes=storage_limit,
            recordings_root=str(recordings_root),
        )
        request.addfinalizer(rdm.shutdown)

        return rdm, recordings_root

    return _make


def test_rdm_stop_recording_drops_future_messages(
    fake_emitter: FakeEmitter,
    rdm_module,
    rdm_factory,
) -> None:
    Emitter = rdm_module.Emitter

    rdm, recordings_root = rdm_factory(storage_limit=None, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "elbow_joint"

    written: list[tuple[str, int]] = []
    done = threading.Event()

    @fake_emitter.on(Emitter.TRACE_WRITTEN)
    def on_written(tid: str, rid: str, bytes_written: int) -> None:
        if tid == trace_id:
            written.append((tid, bytes_written))
            done.set()

    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=DataType.JOINT_POSITIONS,
            data_type_name="joint_position",
            robot_instance=0,
            data=_json_bytes({"x": 1}),
            final_chunk=False,
        )
    )

    rdm.trace_message_queue.join()
    fake_emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

    assert done.wait(timeout=10.0) is True

    trace_dir = recordings_root / recording_id / "JOINT_POSITIONS" / trace_id
    assert _wait_for(lambda: (trace_dir / "trace.json").is_file(), timeout=5.0) is True

    before_files = sorted(p.name for p in trace_dir.rglob("*") if p.is_file())

    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=DataType.JOINT_POSITIONS,
            data_type_name="joint_position",
            robot_instance=0,
            data=_json_bytes({"x": 2}),
            final_chunk=False,
        )
    )
    rdm.trace_message_queue.join()
    time.sleep(0.05)

    after_files = sorted(p.name for p in trace_dir.rglob("*") if p.is_file())
    assert after_files == before_files
    assert len(written) == 1


def test_rdm_delete_trace_event_deletes_trace_dir(
    fake_emitter: FakeEmitter,
    rdm_module,
    rdm_factory,
) -> None:
    Emitter = rdm_module.Emitter

    rdm, recordings_root = rdm_factory(storage_limit=None, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "elbow_joint"

    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=DataType.JOINT_POSITIONS,
            data_type_name="joint_position",
            robot_instance=0,
            data=_json_bytes({"x": 1}),
            final_chunk=False,
        )
    )

    rdm.trace_message_queue.join()
    fake_emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

    trace_dir = recordings_root / recording_id / "JOINT_POSITIONS" / trace_id
    assert _wait_for(lambda: trace_dir.exists(), timeout=5.0) is True

    fake_emitter.emit(
        Emitter.DELETE_TRACE,
        recording_id,
        trace_id,
        DataType.JOINT_POSITIONS.value,
    )

    assert _wait_for(lambda: not trace_dir.exists(), timeout=5.0) is True


def test_rdm_storage_limit_aborts_trace_and_emits_trace_written_zero(
    fake_emitter: FakeEmitter,
    rdm_module,
    rdm_factory,
) -> None:
    Emitter = rdm_module.Emitter

    rdm, recordings_root = rdm_factory(storage_limit=1, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "too_big"

    written: list[tuple[str, int]] = []
    done = threading.Event()

    @fake_emitter.on(Emitter.TRACE_WRITTEN)
    def on_written(tid: str, rid: str, bytes_written: int) -> None:
        if tid == trace_id:
            written.append((tid, bytes_written))
            done.set()

    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=DataType.JOINT_POSITIONS,
            data_type_name="joint_position",
            robot_instance=0,
            data=b"x" * 2048,
            final_chunk=False,
        )
    )

    rdm.trace_message_queue.join()
    assert done.wait(timeout=5.0) is True

    assert written[0] == (trace_id, 0)

    trace_dir = recordings_root / recording_id / "JOINT_POSITIONS" / trace_id
    assert trace_dir.exists() is False


def test_rdm_encoder_creation_failure_aborts_one_trace_but_other_completes(
    fake_emitter: FakeEmitter,
    monkeypatch: pytest.MonkeyPatch,
    rdm_module,
    rdm_factory,
) -> None:
    Emitter = rdm_module.Emitter

    bad_trace_id = "bad_trace"
    good_trace_id = "good_trace"

    from neuracore.data_daemon.recording_encoding_disk_manager.lifecycle import (
        encoder_manager as encoder_manager_module,
    )

    real_json_trace = encoder_manager_module.JsonTrace

    class FailingJsonTrace(real_json_trace):  # type: ignore[misc]
        def __init__(self, output_dir: Path, *args: Any, **kwargs: Any) -> None:
            if bad_trace_id in str(output_dir):
                raise RuntimeError("boom")
            super().__init__(output_dir=output_dir, *args, **kwargs)

    monkeypatch.setattr(
        encoder_manager_module,
        "JsonTrace",
        FailingJsonTrace,
        raising=True,
    )

    rdm, _recordings_root = rdm_factory(storage_limit=None, flush_bytes=1)
    recording_id = str(uuid.uuid4())

    written: list[tuple[str, int]] = []
    done = threading.Event()

    @fake_emitter.on(Emitter.TRACE_WRITTEN)
    def on_written(tid: str, rid: str, bytes_written: int) -> None:
        if tid in {bad_trace_id, good_trace_id}:
            written.append((tid, bytes_written))

        seen = {t for t, _ in written}
        if seen == {bad_trace_id, good_trace_id}:
            done.set()

    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=bad_trace_id,
            data_type=DataType.JOINT_POSITIONS,
            data_type_name="joint_position",
            robot_instance=0,
            data=_json_bytes({"x": 1}),
            final_chunk=False,
        )
    )
    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=good_trace_id,
            data_type=DataType.JOINT_POSITIONS,
            data_type_name="joint_position",
            robot_instance=0,
            data=_json_bytes({"y": 2}),
            final_chunk=False,
        )
    )

    rdm.trace_message_queue.join()
    fake_emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

    assert done.wait(timeout=10.0) is True

    by_trace: dict[str, list[int]] = {}
    for tid, nbytes in written:
        by_trace.setdefault(tid, []).append(nbytes)

    assert set(by_trace) == {bad_trace_id, good_trace_id}
    assert max(by_trace[bad_trace_id]) == 0
    assert max(by_trace[good_trace_id]) > 0

    rdm.shutdown()
