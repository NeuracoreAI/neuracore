from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.models import CompleteMessage


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

    def add_rgb_batch(self, batch_path: Path) -> None:
        meta_path = batch_path.with_suffix(".meta.jsonl")
        with meta_path.open("rb") as meta_f, batch_path.open("rb") as rgb_f:
            for raw_line in meta_f:
                meta_line = raw_line.strip()
                if not meta_line:
                    continue
                parsed = json.loads(meta_line.decode("utf-8"))
                frame_nbytes = 0
                frame_only = False
                if isinstance(parsed, dict):
                    frame_only = bool(parsed.get("__raw_frame_only"))
                    raw_frame_nbytes = parsed.get("frame_nbytes")
                    if isinstance(raw_frame_nbytes, int) and raw_frame_nbytes >= 0:
                        frame_nbytes = raw_frame_nbytes
                if not frame_only:
                    self.add_payload(
                        json.dumps(
                            parsed,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    )
                if frame_nbytes > 0:
                    self.add_payload(rgb_f.read(frame_nbytes))

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


def _make_combined_video_payload(
    *, width: int, height: int, timestamp: float, frame: bytes
) -> bytes:
    metadata = {
        "width": width,
        "height": height,
        "timestamp": timestamp,
        "frame_nbytes": len(frame),
    }
    metadata_json = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(metadata_json)) + metadata_json + frame


def _wait_for(pred: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture
def loop_manager_with_emitter() -> tuple[EventLoopManager, Emitter]:
    """EventLoopManager instance for tests, together with its Emitter."""
    manager = EventLoopManager()
    rdm_emitter = manager.start()
    yield manager, rdm_emitter

    if manager.is_running():
        try:
            manager.stop()
        except RuntimeError:
            pass


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
    rdm_module,
    loop_manager_with_emitter: tuple[EventLoopManager, Emitter],
    request: pytest.FixtureRequest,
):
    loop_manager, rdm_emitter = loop_manager_with_emitter
    rdm_instances = []

    def _make(
        *,
        storage_limit: int | None,
        flush_bytes: int = 1,
        rgb_flush_bytes: int | None = None,
        trace_message_queue_maxsize: int = 1,
    ):
        recordings_root = tmp_path / "recordings"

        rdm = rdm_module.RecordingDiskManager(
            loop_manager=loop_manager,
            emitter=rdm_emitter,
            flush_bytes=flush_bytes,
            rgb_flush_bytes=(
                flush_bytes if rgb_flush_bytes is None else rgb_flush_bytes
            ),
            trace_message_queue_maxsize=trace_message_queue_maxsize,
            storage_limit_bytes=storage_limit,
            recordings_root=str(recordings_root),
        )
        rdm_instances.append(rdm)

        return rdm, recordings_root

    yield _make

    for rdm in rdm_instances:
        try:
            future = asyncio.run_coroutine_threadsafe(
                rdm.shutdown(), loop_manager.general_loop
            )
            future.result(timeout=5.0)
        except Exception:
            pass


def test_rdm_stop_recording_drops_future_messages(
    rdm_module,
    rdm_factory,
    loop_manager_with_emitter: tuple[EventLoopManager, Emitter],
) -> None:
    _loop_manager, emitter = loop_manager_with_emitter
    RdmEmitter = rdm_module.Emitter

    rdm, recordings_root = rdm_factory(storage_limit=None, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "elbow_joint"

    written: list[tuple[str, int]] = []
    done = threading.Event()

    @emitter.on(RdmEmitter.TRACE_WRITTEN)
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

    # Wait for message to be processed
    time.sleep(0.1)
    emitter.emit(RdmEmitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

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
    time.sleep(0.15)

    after_files = sorted(p.name for p in trace_dir.rglob("*") if p.is_file())
    assert after_files == before_files
    assert len(written) == 1


def test_rdm_delete_trace_event_deletes_trace_dir(
    rdm_module,
    rdm_factory,
    loop_manager_with_emitter: tuple[EventLoopManager, Emitter],
) -> None:
    _loop_manager, emitter = loop_manager_with_emitter
    RdmEmitter = rdm_module.Emitter

    rdm, recordings_root = rdm_factory(storage_limit=None, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "elbow_joint"

    trace_written = threading.Event()

    @emitter.on(RdmEmitter.TRACE_WRITTEN)
    def _on_trace_written(tid: str, _rid: str, _bytes: int) -> None:
        if tid == trace_id:
            trace_written.set()

    try:
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

        # Wait for message to be processed
        time.sleep(0.1)
        emitter.emit(RdmEmitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

        trace_dir = recordings_root / recording_id / "JOINT_POSITIONS" / trace_id
        assert _wait_for(lambda: trace_dir.exists(), timeout=5.0) is True

        # Wait for encoder to finish (TRACE_WRITTEN) so no files are open when we delete
        assert trace_written.wait(timeout=5.0) is True

        emitter.emit(
            RdmEmitter.DELETE_TRACE,
            recording_id,
            trace_id,
            DataType.JOINT_POSITIONS.value,
        )

        assert _wait_for(lambda: not trace_dir.exists(), timeout=5.0) is True
        data_type_dir = recordings_root / recording_id / "JOINT_POSITIONS"
        recording_dir = recordings_root / recording_id
        assert _wait_for(lambda: not data_type_dir.exists(), timeout=5.0) is True
        assert _wait_for(lambda: not recording_dir.exists(), timeout=5.0) is True
    finally:
        emitter.remove_listener(RdmEmitter.TRACE_WRITTEN, _on_trace_written)


def test_rdm_storage_limit_aborts_trace_and_emits_trace_written_zero(
    rdm_module,
    rdm_factory,
    loop_manager_with_emitter: tuple[EventLoopManager, Emitter],
) -> None:
    _loop_manager, emitter = loop_manager_with_emitter
    RdmEmitter = rdm_module.Emitter

    rdm, recordings_root = rdm_factory(storage_limit=1, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "too_big"

    written: list[tuple[str, int]] = []
    done = threading.Event()

    @emitter.on(RdmEmitter.TRACE_WRITTEN)
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

    # Wait for message to be processed
    assert done.wait(timeout=5.0) is True

    assert written[0] == (trace_id, 0)

    trace_dir = recordings_root / recording_id / "JOINT_POSITIONS" / trace_id
    assert trace_dir.exists() is False


def test_rdm_encoder_creation_failure_aborts_one_trace_but_other_completes(
    monkeypatch: pytest.MonkeyPatch,
    rdm_module,
    rdm_factory,
    loop_manager_with_emitter: tuple[EventLoopManager, Emitter],
) -> None:
    _loop_manager, emitter = loop_manager_with_emitter
    RdmEmitter = rdm_module.Emitter

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

    @emitter.on(RdmEmitter.TRACE_WRITTEN)
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

    # Wait for messages to be processed
    time.sleep(0.1)
    emitter.emit(RdmEmitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

    assert done.wait(timeout=10.0) is True

    by_trace: dict[str, list[int]] = {}
    for tid, nbytes in written:
        by_trace.setdefault(tid, []).append(nbytes)

    assert set(by_trace) == {bad_trace_id, good_trace_id}
    assert max(by_trace[bad_trace_id]) == 0
    assert max(by_trace[good_trace_id]) > 0


def test_rdm_rgb_trace_uses_rgb_batch_flow(
    rdm_factory,
) -> None:
    rdm, recordings_root = rdm_factory(storage_limit=None, flush_bytes=1)

    recording_id = str(uuid.uuid4())
    trace_id = "camera_front"
    width, height = 4, 3
    frame = _make_rgb24_frame_bytes(0, width, height)
    payload = _make_combined_video_payload(
        width=width,
        height=height,
        timestamp=1.0,
        frame=frame,
    )

    rdm.enqueue(
        CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=DataType.RGB_IMAGES,
            data_type_name="camera_front",
            robot_instance=0,
            data=payload,
            final_chunk=True,
        )
    )

    trace_dir = recordings_root / recording_id / "RGB_IMAGES" / trace_id
    assert _wait_for(lambda: (trace_dir / "trace.json").is_file(), timeout=10.0) is True
    assert (trace_dir / "lossy.mp4").is_file()
    assert (trace_dir / "lossless.mp4").is_file()


def test_rdm_enqueue_blocks_when_rgb_writer_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
    rdm_factory,
) -> None:
    from neuracore.data_daemon.recording_encoding_disk_manager.workers import (
        raw_batch_writer as raw_batch_writer_module,
    )

    started = threading.Event()
    allow_continue = threading.Event()

    real_append_rgb_payload = raw_batch_writer_module._RawBatchWriter._append_rgb_payload

    async def blocked_append_rgb_payload(self, writer_state, payload_bytes):
        if not started.is_set():
            started.set()
            await asyncio.to_thread(allow_continue.wait)
        return await real_append_rgb_payload(self, writer_state, payload_bytes)

    monkeypatch.setattr(
        raw_batch_writer_module._RawBatchWriter,
        "_append_rgb_payload",
        blocked_append_rgb_payload,
        raising=True,
    )

    rdm, _recordings_root = rdm_factory(
        storage_limit=None,
        flush_bytes=1024 * 1024,
        trace_message_queue_maxsize=1,
    )

    recording_id = str(uuid.uuid4())
    trace_id = "camera_front"
    width, height = 4, 3

    def _message(frame_index: int) -> CompleteMessage:
        return CompleteMessage.from_bytes(
            producer_id="p",
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=DataType.RGB_IMAGES,
            data_type_name="camera_front",
            robot_instance=0,
            data=_make_combined_video_payload(
                width=width,
                height=height,
                timestamp=float(frame_index),
                frame=_make_rgb24_frame_bytes(frame_index, width, height),
            ),
            final_chunk=False,
        )

    rdm.enqueue(_message(0))
    assert started.wait(timeout=5.0) is True

    rdm.enqueue(_message(1))

    third_enqueue_finished = threading.Event()
    third_enqueue_errors: list[BaseException] = []

    def _enqueue_third_message() -> None:
        try:
            rdm.enqueue(_message(2))
        except BaseException as exc:  # pragma: no cover - surfaced in assertion
            third_enqueue_errors.append(exc)
        finally:
            third_enqueue_finished.set()

    enqueue_thread = threading.Thread(target=_enqueue_third_message, daemon=True)
    enqueue_thread.start()

    time.sleep(0.15)
    assert third_enqueue_finished.is_set() is False

    allow_continue.set()

    assert third_enqueue_finished.wait(timeout=5.0) is True
    enqueue_thread.join(timeout=1.0)
    assert third_enqueue_errors == []
