"""Shared fixtures for the native-producer adaptor tests.

Centralises the recording stub for the PyO3 ``_native_producer`` module so the
adaptor tests and the data-stream routing tests stay in lockstep on call
signatures. The previous duplicated copies drifted into a 3-arg vs 4-arg
mismatch on ``send_data`` that masked a real production-code TypeError.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from neuracore.data_daemon.communications_management.producer import (
    native_producer_channel as adaptor,
)


class NativeProducerStub:
    """Records every native entry-point call for inspection by tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def start_recording(self, recording_id: str) -> None:
        self.calls.append(("start_recording", (recording_id,)))

    def start_trace(
        self,
        recording_id: str,
        trace_id: str,
        data_type: str,
        data_type_name: str | None = None,
    ) -> None:
        self.calls.append(
            ("start_trace", (recording_id, trace_id, data_type, data_type_name))
        )

    def open_frame_stream(self, trace_id: str, width: int, height: int) -> None:
        self.calls.append(("open_frame_stream", (trace_id, width, height)))

    def send_data(
        self,
        trace_id: str,
        payload: bytes,
        timestamp_ns: int,
        timestamp_s: float | None = None,
    ) -> None:
        self.calls.append(
            ("send_data", (trace_id, bytes(payload), timestamp_ns, timestamp_s))
        )

    def end_trace(self, trace_id: str) -> None:
        self.calls.append(("end_trace", (trace_id,)))

    def stop_recording(self, recording_id: str) -> None:
        self.calls.append(("stop_recording", (recording_id,)))


def install_native_stub(
    monkeypatch: pytest.MonkeyPatch, stub: NativeProducerStub
) -> None:
    """Register ``stub`` as the active ``_native_producer`` module.

    Resets the adaptor module's cached handle on both setup and teardown so
    repeated installations within a session do not see stale references.
    """
    fake_module = types.ModuleType("neuracore.data_daemon._native_producer")
    for method_name in (
        "start_recording",
        "start_trace",
        "open_frame_stream",
        "send_data",
        "end_trace",
        "stop_recording",
    ):
        setattr(fake_module, method_name, getattr(stub, method_name))

    monkeypatch.setitem(
        sys.modules, "neuracore.data_daemon._native_producer", fake_module
    )
    monkeypatch.setattr(adaptor, "_NATIVE_MODULE", None)


@pytest.fixture
def native_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[NativeProducerStub]:
    """Install a fake ``_native_producer`` for the duration of one test."""
    stub = NativeProducerStub()
    install_native_stub(monkeypatch, stub)
    try:
        yield stub
    finally:
        monkeypatch.setattr(adaptor, "_NATIVE_MODULE", None)
