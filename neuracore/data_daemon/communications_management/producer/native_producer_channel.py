"""Thin Python adaptor over the Rust ``_native_producer`` PyO3 module.

Used by [data_stream.py](neuracore/core/streaming/data_stream.py) when
[rust_daemon_enabled()](neuracore/data_daemon/rust_selection.py) is true. The
adaptor publishes envelopes straight onto the iceoryx2 commands service that
the Rust daemon listens on — there is no sequence allocator, no background
sender thread, no chunking, and no shared-slot transport here. iceoryx2's
publish/subscribe ports handle delivery; we only translate per-stream
lifecycle calls into the matching ``StartRecording`` / ``StartTrace`` /
``OpenFrameStream`` / ``Frame`` / ``EndTrace`` envelopes.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only typing hints
    from neuracore_types import DataType

    from neuracore.data_daemon.models import BatchedJointDataPayload

logger = logging.getLogger(__name__)

_NATIVE_MODULE: ModuleType | None = None
_NATIVE_IMPORT_HINT = (
    "neuracore.data_daemon._native_producer is not available. Build the Rust "
    "data_daemon_producer crate with maturin and ensure the resulting "
    "extension is on sys.path, or unset NCD_RUST_DAEMON to fall back to the "
    "legacy Python producer."
)

# Per-process record of which shim methods have already logged a warning.
# Keeps the log noise to one line per missing semantic per process, while
# still surfacing to the next caller who depends on real backpressure or
# sequence numbers that the Rust daemon is not yet wired for that contract.
_WARNED_NATIVE_SHIMS: set[str] = set()


def _warn_once_native_shim(shim_name: str, detail: str) -> None:
    if shim_name in _WARNED_NATIVE_SHIMS:
        return
    _WARNED_NATIVE_SHIMS.add(shim_name)
    logger.warning(
        "NativeProducerChannel.%s is a no-op shim under the rust daemon "
        "(%s). Sub-phase 4h will wire this through iceoryx2 commands-"
        "response acknowledgements.",
        shim_name,
        detail,
    )


def _load_native() -> ModuleType:
    """Lazily import the PyO3 module and cache it for the process lifetime."""
    global _NATIVE_MODULE
    if _NATIVE_MODULE is None:
        try:
            _NATIVE_MODULE = import_module("neuracore.data_daemon._native_producer")
        except ImportError as error:
            raise RuntimeError(_NATIVE_IMPORT_HINT) from error
    return _NATIVE_MODULE


class NativeProducerChannel:
    """Per-stream lifecycle wrapper around the native producer entry points.

    Mirrors the slice of
    [ProducerChannel](neuracore/data_daemon/communications_management/producer/producer_channel.py)
    that [data_stream.py](neuracore/core/streaming/data_stream.py) actually
    relies on. Methods that don't translate to the Rust daemon's wire format
    (sequence numbers, batched joint data, shared-slot transport) degrade
    gracefully — explicit no-ops with debug logging — so the SDK does not have
    to branch on backend at every call site.
    """

    def __init__(
        self,
        data_type: DataType,
        recording_id: str | None = None,
        id: str | None = None,
        data_type_name: str | None = None,
    ) -> None:
        """Initialise the channel; no IPC happens until the first envelope.

        Args:
            data_type: SDK-side data type. Its ``.value`` becomes the wire
                ``data_type`` field on ``StartTrace`` and the on-disk path
                segment under ``recordings/<recording>/``.
            recording_id: Optional recording identifier. Can be supplied here
                or via :meth:`start_recording_session`.
            id: Optional channel identifier kept for parity with
                :class:`ProducerChannel`; not used by the native pipeline.
            data_type_name: Optional per-stream label (joint name, camera id,
                custom label). Propagated to the daemon on ``StartTrace`` so
                multiple traces sharing a ``data_type`` are distinguishable in
                the database — required for tests/tooling that key on
                ``"<data_type>/<data_type_name>"``.
        """
        if data_type is None:
            raise ValueError("data_type is required")
        self._data_type = data_type
        self.channel_id = id or str(uuid.uuid4())
        self.recording_id = recording_id
        self.trace_id: str | None = None
        self._data_type_name = data_type_name
        self._frame_resolution: tuple[int, int] | None = None

    def start_recording_session(
        self,
        recording_id: str | None = None,
        shared_memory_size: int | None = None,  # noqa: ARG002 — legacy parity
    ) -> None:
        """Publish ``StartRecording`` and ``StartTrace`` for a fresh trace."""
        native = _load_native()
        if recording_id is not None:
            self.recording_id = recording_id
        if not self.recording_id:
            raise ValueError(
                "recording_id is required; set on NativeProducerChannel init."
            )
        if self.trace_id is not None:
            raise RuntimeError(
                "Cannot start a new recording session while a trace is active."
            )

        native.start_recording(self.recording_id)
        self.trace_id = str(uuid.uuid4())
        native.start_trace(
            self.recording_id,
            self.trace_id,
            self._data_type.value,
            self._data_type_name,
        )

        # If a frame resolution was announced before the trace existed, replay
        # it now so the daemon-side actor opens the video writer in time.
        if self._frame_resolution is not None:
            width, height = self._frame_resolution
            native.open_frame_stream(self.trace_id, width, height)

    def announce_frame_resolution(self, width: int, height: int) -> None:
        """Publish ``OpenFrameStream`` so the daemon opens the video writer.

        Idempotent — re-announcing the same resolution within a trace is a
        no-op. Calling this before :meth:`start_recording_session` is allowed;
        the resolution is buffered and emitted as soon as the trace exists.
        """
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if self._frame_resolution == (width, height) and self.trace_id is not None:
            return
        self._frame_resolution = (width, height)
        if self.trace_id is None:
            return
        native = _load_native()
        native.open_frame_stream(self.trace_id, width, height)

    def send_frame(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        timestamp_ns: int | None = None,
        timestamp_s: float | None = None,
    ) -> None:
        """Publish one ``Frame`` envelope for the active trace.

        ``timestamp_s`` (the SDK-supplied float capture time in seconds) is
        forwarded alongside ``timestamp_ns`` so the daemon-side video sidecar
        can record bit-exact floats — the integer-ns round-trip would clip
        manual-test timestamps like ``7/60`` past microsecond precision.
        """
        if self.trace_id is None:
            raise RuntimeError(
                "send_frame called before start_recording_session opened a trace"
            )
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        elif isinstance(payload, bytearray):
            payload = bytes(payload)
        if timestamp_ns is None and timestamp_s is not None:
            timestamp_ns = int(timestamp_s * 1_000_000_000)
        timestamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        native = _load_native()
        native.send_data(self.trace_id, payload, timestamp, timestamp_s)

    def mark_recording_stop_requested(self) -> int:
        """Return a no-op cutoff — the Rust daemon does not yet use sequences."""
        return 0

    def cleanup_producer_channel(
        self,
        stop_cutoff_sequence_number: int,  # noqa: ARG002 — legacy parity
        wait_for_slot_drain: bool = True,  # noqa: ARG002 — legacy parity
    ) -> None:
        """Publish ``EndTrace`` and clear the per-trace state."""
        if self.trace_id is None:
            return
        native = _load_native()
        try:
            native.end_trace(self.trace_id)
        finally:
            self.trace_id = None
            self.recording_id = None
            self._frame_resolution = None

    def stop_producer_channel(
        self,
        wait_for_slot_drain: bool = True,  # noqa: ARG002 — legacy parity
    ) -> None:
        """No-op: the recording-scoped ``StopRecording`` is sent elsewhere.

        The legacy ProducerChannel shuts down its background sender thread and
        shared-slot transport here. Neither exists in this adaptor, so there
        is nothing to drain. Recording-level shutdown lives in the SDK's
        recording state manager, not in the per-stream channel.
        """

    # Compatibility shims for the few call sites in
    # [api/logging.py](neuracore/api/logging.py) that reach into the producer
    # for unsupported features.

    def send_batched_joint_data(self, payload: BatchedJointDataPayload) -> None:
        """Fan a batched joint payload out to one ``Frame`` envelope per item.

        The legacy producer sends a single ``BATCHED_JOINT_DATA`` zmq message
        and the Python daemon's data-bridge splits it per item. iceoryx2's
        commands service has no equivalent fan-out, so the shim mirrors the
        bridge inline: per item, publish a ``Frame`` envelope addressed to
        that item's ``trace_id`` carrying ``{"timestamp": ..., "value": ...}``
        — matching the on-disk JSON the legacy daemon wrote
        ([data_bridge.py:508](neuracore/data_daemon/communications_management/consumer/data_bridge.py#L508)).
        """
        if not payload.items:
            return
        _warn_once_native_shim(
            "send_batched_joint_data",
            "approximate inline fan-out replaces the legacy daemon's single "
            "BATCHED_JOINT_DATA zmq message; per-item ordering and timing are "
            "best-effort under iceoryx2",
        )
        native = _load_native()
        timestamp_ns = int(payload.timestamp * 1_000_000_000)
        for item in payload.items:
            entry = json.dumps(
                {"timestamp": payload.timestamp, "value": item.value},
                separators=(",", ":"),
            ).encode("utf-8")
            native.send_data(item.trace_id, entry, timestamp_ns, payload.timestamp)

    def get_last_accepted_sequence_number(self) -> int:
        """Return 0; the Rust daemon does not expose sequence numbers."""
        _warn_once_native_shim(
            "get_last_accepted_sequence_number",
            "always returns 0 under the rust daemon",
        )
        return 0

    def get_last_enqueued_sequence_number(self) -> int:
        """Return 0; the Rust daemon does not expose sequence numbers."""
        _warn_once_native_shim(
            "get_last_enqueued_sequence_number",
            "always returns 0 under the rust daemon",
        )
        return 0

    def get_last_sent_sequence_number(self) -> int:
        """Return 0; the Rust daemon does not expose sequence numbers."""
        _warn_once_native_shim(
            "get_last_sent_sequence_number",
            "always returns 0 under the rust daemon",
        )
        return 0

    def wait_until_sequence_sent(self, sequence_number: int) -> bool:
        """Return True; native publishes block until iceoryx2 accepts the sample."""
        if sequence_number > 0:
            _warn_once_native_shim(
                "wait_until_sequence_sent",
                f"called with sequence_number={sequence_number} but the rust "
                "daemon publishes are synchronous; True is returned without a "
                "real acknowledgement",
            )
        return True
