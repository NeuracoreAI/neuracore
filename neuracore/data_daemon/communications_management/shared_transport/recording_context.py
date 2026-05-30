"""Recording-scoped context for driving the data daemon."""

from __future__ import annotations

import logging
import time
from importlib import import_module
from types import ModuleType

from neuracore.data_daemon.models import CommandType
from neuracore.data_daemon.rust_selection import rust_daemon_enabled

from .communications_manager import CommunicationsManager, MessageEnvelope

logger = logging.getLogger(__name__)

_NATIVE_MODULE: ModuleType | None = None

_NATIVE_IMPORT_HINT = (
    "neuracore.data_daemon._native_producer is not available. Build the Rust "
    "data_daemon_producer crate with maturin and ensure the resulting "
    "extension is on sys.path, or unset NCD_RUST_DAEMON to fall back to the "
    "legacy Python producer."
)


def _load_native() -> ModuleType:
    """Lazily import and cache the PyO3 producer module for the process."""
    global _NATIVE_MODULE
    if _NATIVE_MODULE is None:
        try:
            _NATIVE_MODULE = import_module("neuracore.data_daemon._native_producer")
        except ImportError as error:
            raise RuntimeError(_NATIVE_IMPORT_HINT) from error
    return _NATIVE_MODULE


class RecordingContext:
    """Recording-scoped interface to the data daemon.

    Under the Rust daemon this is a *thin shipper* bridge: ``start_recording``
    / ``log_joints`` / ``log_frame`` / ``log_scalar`` / ``stop_recording`` /
    ``cancel_recording`` forward straight through to ``_native_producer`` over
    iceoryx2, tagged only with the **source** ``(robot_id, robot_instance)``.
    The daemon owns all recording identity — there is no recording id on the
    wire. Routing is by a producer-stamped *publish* timestamp (wall clock),
    decoupled from the data's own capture timestamp, so the daemon partitions
    recordings by when data was published rather than what clock it carries.

    Under the legacy daemon it keeps its original role: sending recording
    control messages over the ZMQ producer socket. That path is unchanged.
    """

    def __init__(
        self,
        recording_id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initialize the recording context.

        Under the Rust daemon the ZMQ producer socket is unused — every
        envelope flows through ``_native_producer`` over iceoryx2 — so we skip
        creating it.
        """
        self.recording_id = recording_id
        self._rust_mode = rust_daemon_enabled()
        # Rust-mode source state. Set on ``start_recording``; the data path
        # reads it to tag every envelope. ``_recording_marker_ns`` is a
        # wall-clock instant inside the recording window used later to resolve
        # the daemon-owned cloud recording id (see ``get_recording_id``).
        self._robot_id: str | None = None
        self._robot_instance: int = 0
        self._recording_marker_ns: int = 0
        if self._rust_mode:
            self._comm = None
        else:
            self._comm = comm_manager or CommunicationsManager()
            self._comm.create_producer_socket()

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set or clear the recording identifier for this context."""
        self.recording_id = recording_id

    # -- Native (Rust daemon) interface -------------------------------------

    def start_recording(
        self,
        robot_id: str,
        robot_instance: int = 0,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Announce a recording to the Rust daemon for a source.

        Publishes exactly one ``StartRecording`` envelope tagged with the
        source ``(robot_id, robot_instance)``. No recording id is on the wire —
        the daemon allocates and owns recording identity.
        """
        if not self._rust_mode:
            return
        if not robot_id:
            raise ValueError("robot_id is required to start a recording.")
        self._robot_id = robot_id
        self._robot_instance = robot_instance
        # The producer owns the publish clock: it stamps the window's
        # ``started_at_ns`` and returns it. That value is the daemon's
        # ``start_timestamp_ns`` for this recording, so we keep it verbatim as
        # the marker used to resolve the daemon-assigned cloud recording id
        # later (``get_recording_id`` matches it exactly).
        self._recording_marker_ns = _load_native().start_recording(
            robot_id,
            robot_instance,
            robot_name,
            dataset_id,
            dataset_name,
        )

    def log_joints(
        self,
        data_type: str,
        timestamp: float,
        items: list[tuple[str, float]],
    ) -> None:
        """Forward a batch of ``(joint_name, value)`` samples to the daemon.

        The daemon lazily creates a trace per sensor and routes the whole batch
        into the source's active recording window by ``timestamp``.
        """
        if not items:
            return
        robot_id = self._require_source("log_joints")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_joints(
            robot_id, self._robot_instance, data_type, items, timestamp_ns, timestamp
        )

    def log_frame(
        self,
        data_type: str,
        name: str,
        width: int,
        height: int,
        payload: memoryview,
        timestamp: float,
    ) -> None:
        """Forward one video frame to the daemon.

        ``payload`` may be either a ``bytes`` object or a flat ``memoryview``
        (e.g. ``memoryview(numpy_array).cast("B")``); the native side reads
        the buffer via the Python buffer protocol and copies straight into
        the NUT writer's destination, so there's no benefit to materialising
        a ``bytes`` first.
        """
        robot_id = self._require_source("log_frame")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_frame(
            robot_id,
            self._robot_instance,
            data_type,
            name,
            int(width),
            int(height),
            payload,
            timestamp_ns,
            timestamp,
        )

    def log_scalar(
        self,
        data_type: str,
        name: str,
        payload: bytes,
        timestamp: float,
    ) -> None:
        """Forward one scalar/custom sample to the daemon."""
        robot_id = self._require_source("log_scalar")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_scalar(
            robot_id,
            self._robot_instance,
            data_type,
            name,
            payload,
            timestamp_ns,
            timestamp,
        )

    def cancel_recording(self, recording_id: str | None = None) -> None:
        """Cancel the source's active recording — the daemon discards it."""
        if not self._rust_mode:
            return
        if not self._robot_id:
            return
        _load_native().cancel_recording(self._robot_id, self._robot_instance)

    def _require_source(self, operation: str) -> str:
        """Return the active source's robot id or raise if logging before start."""
        if not self._rust_mode:
            raise RuntimeError(f"{operation} is only available under the rust daemon.")
        if not self._robot_id:
            raise RuntimeError(
                f"{operation} called before start_recording set a source."
            )
        return self._robot_id

    # -- Lifecycle ----------------------------------------------------------

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        """Send a recording-stopped control message.

        Under the Rust daemon this publishes one ``StopRecording`` tagged with
        the source and the publish-clock stop boundary (wall-clock now), which
        the daemon uses to close the recording window. ``recording_id`` /
        ``producer_stop_sequence_numbers`` are only used by the legacy path.
        """
        if self._rust_mode:
            if not self._robot_id:
                return
            _load_native().stop_recording(self._robot_id, self._robot_instance)
            return

        effective_recording_id = recording_id or self.recording_id
        if not effective_recording_id:
            raise ValueError("recording_id is required to stop a recording.")
        recording_stopped_payload: dict[str, object] = {
            "recording_id": effective_recording_id
        }
        if producer_stop_sequence_numbers:
            recording_stopped_payload["producer_stop_sequence_numbers"] = (
                producer_stop_sequence_numbers
            )
        self._send(
            CommandType.RECORDING_STOPPED,
            {"recording_stopped": recording_stopped_payload},
        )
        self.recording_id = effective_recording_id

    def get_recording_id(
        self,
        timestamp_ns: int | None = None,
        timeout_s: float = 30.0,
    ) -> str | None:
        """Resolve the daemon-owned cloud recording id for this source.

        The thin-shipper producer never sees the cloud recording id — the
        daemon allocates it and POSTs ``/recording/start`` asynchronously. This
        method asks the native daemon for the id of the recording whose window
        brackets ``timestamp_ns`` (defaulting to the marker captured at
        ``start_recording``) for this source, polling the daemon's state until
        the id has been minted or ``timeout_s`` elapses.

        It MAY block and is for non-performance-critical paths only (tests,
        ``nc.stop_recording(wait=True)``). Returns ``None`` on timeout or in the
        legacy daemon mode.
        """
        if not self._rust_mode or not self._robot_id:
            return None
        marker_ns = (
            timestamp_ns if timestamp_ns is not None else self._recording_marker_ns
        )

        # Imported here to avoid a module-load dependency on the daemon
        # lifecycle helpers for callers that never resolve a cloud id.
        import sqlite3

        from neuracore.data_daemon.helpers import get_daemon_db_path

        db_uri = f"file:{get_daemon_db_path()}?mode=ro"
        deadline = time.monotonic() + timeout_s
        # Match the recording whose start *equals* the marker rather than
        # ``<=``. The marker is the producer's ``started_at_ns`` for a specific
        # ``start_recording`` call, stored verbatim as the row's
        # ``start_timestamp_ns``, so an exact match resolves precisely that
        # recording. ``<=`` could otherwise fall back to an earlier recording
        # for the same source during the window where this recording's row /
        # cloud id has not landed yet (e.g. a just-cancelled prior recording
        # whose ``cancelled_at`` the daemon has not stamped yet) — which would
        # resolve a stale, soon-to-be-discarded cloud id.
        query = (
            "SELECT recording_id FROM recordings "
            "WHERE robot_id = ? AND robot_instance = ? AND start_timestamp_ns = ? "
            "AND cancelled_at IS NULL "
            "ORDER BY recording_index DESC LIMIT 1"
        )
        while True:
            try:
                connection = sqlite3.connect(db_uri, uri=True, timeout=1.0)
                try:
                    row = connection.execute(
                        query, (self._robot_id, self._robot_instance, marker_ns)
                    ).fetchone()
                finally:
                    connection.close()
                if row is not None and row[0] is not None:
                    return str(row[0])
            except sqlite3.Error as error:
                logger.debug("recording-id lookup query failed: %s", error)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)

    def close(self) -> None:
        """Close sockets and cleanup context resources owned by this instance."""
        if self._comm is not None:
            self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> None:
        """Send a management message to the daemon.

        Args:
            command: The CommandType to send to the daemon.
            payload: A dictionary containing any additional data required by the daemon
                to process the message.

        Returns:
            None
        """
        if self._comm is None:
            raise RuntimeError(
                "Cannot send a control message: no CommunicationsManager is "
                "available (rust daemon mode bypasses the ZMQ producer socket)."
            )
        envelope = MessageEnvelope(
            producer_id=None,
            command=command,
            payload=payload or {},
        )
        self._comm.send_message(envelope)
