"""Recording-scoped context for driving the data daemon.

The SDK's only interface to the daemon. Every method forwards straight through
to the ``_data_bridge`` PyO3 extension over iceoryx2; see
[rust/data_daemon_bridge/src/lib.rs](../../rust/data_daemon_bridge/src/lib.rs)
for the native side.
"""

from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType

from neuracore.data_daemon.daemon_control import ensure_daemon_running

logger = logging.getLogger(__name__)

_DATA_BRIDGE_MODULE: ModuleType | None = None

_DATA_BRIDGE_IMPORT_HINT = (
    "neuracore.data_daemon._data_bridge is not available. The Rust extension "
    "is bundled in the neuracore wheel (prebuilt for Linux x86_64 and "
    "Apple-Silicon macOS); reinstall with `pip install neuracore` on one of "
    "those platforms, or build it into a source checkout with "
    "`bash rust/scripts/build_wheel_artefacts.sh && pip install -e .`."
)


def _load_native() -> ModuleType:
    """Lazily import and cache the PyO3 data daemon bridge module for the process.

    The compiled extension ships inside the ``neuracore`` wheel (prebuilt for
    Linux x86_64 and Apple-Silicon macOS), so this raises a helpful error when
    it is absent (e.g. a source install without the Rust build).
    """
    global _DATA_BRIDGE_MODULE
    if _DATA_BRIDGE_MODULE is None:
        try:
            _DATA_BRIDGE_MODULE = import_module("neuracore.data_daemon._data_bridge")
        except ImportError as error:
            raise RuntimeError(_DATA_BRIDGE_IMPORT_HINT) from error
    return _DATA_BRIDGE_MODULE


class LoggingStalledError(RuntimeError):
    """The daemon is not draining its spool backlog fast enough to admit a frame."""


def notify_daemon_config_changed() -> None:
    """Try ask a running daemon to reload its profile immediately."""
    try:
        _load_native().refresh_config()
    except Exception as error:  # noqa: BLE001 - best-effort, never fatal
        logger.debug("Could not notify the daemon of a config change: %s", error)


class RecordingContext:
    """Recording-scoped interface to the data daemon.

    A *thin shipper* bridge: ``start_recording`` / ``log_joints`` / ``log_frame``
    / ``log_json`` / ``stop_recording`` / ``cancel_recording`` forward straight
    through to ``_data_bridge`` over iceoryx2, tagged only with the **source**
    ``(robot_id, robot_instance)``. The daemon owns all recording identity —
    there is no recording id on the wire, so this class carries none. Routing
    is by a producer-stamped *publish* timestamp (wall clock), decoupled from
    the data's own capture timestamp, so the daemon partitions recordings by
    when data was published rather than what clock it carries.
    """

    def __init__(self) -> None:
        """Initialize the recording context."""
        self._robot_id: str | None = None
        self._robot_instance: int = 0
        self._recording_marker_ns: int = 0

    def bind_source(self, robot_id: str, robot_instance: int = 0) -> None:
        """Bind messages to a robot source without starting a recording.

        A process may learn that a recording is active through SSE even when a
        different process published ``StartRecording``. Those producer
        processes still need the source identity in order to contribute data to
        the daemon-owned recording window, but must not publish another start.

        Args:
            robot_id: Robot identifier used as the daemon source key.
            robot_instance: Robot instance used as the daemon source key.
        """
        if not robot_id:
            raise ValueError("robot_id is required to bind a recording source.")
        self._robot_id = robot_id
        self._robot_instance = robot_instance

    def start_recording(
        self,
        robot_id: str,
        robot_instance: int = 0,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Announce a recording to the daemon for a source.

        Publishes exactly one ``StartRecording`` envelope tagged with the
        source ``(robot_id, robot_instance)``. No recording id is on the wire —
        the daemon allocates and owns recording identity.

        ``timestamp`` is the recording's *capture* start time (Unix seconds),
        stored and reported as such, and returned as the marker that resolves the
        cloud id (:meth:`get_recording_id`). It does **not** bound the recording
        window, which the daemon takes from a publish stamp inside this call.
        """
        ensure_daemon_running()
        self.bind_source(robot_id, robot_instance)
        timestamp_ns = int(timestamp * 1_000_000_000) if timestamp is not None else None

        self._recording_marker_ns = _load_native().start_recording(
            robot_id,
            robot_instance,
            robot_name,
            dataset_id,
            dataset_name,
            timestamp_ns,
        )

    def log_joints(
        self,
        data_type: str,
        timestamp: float,
        joined_names: str,
        values: list[float],
    ) -> None:
        r"""Forward a batch of joint scalar samples to the daemon.

        Args:
            data_type:  Type of joint data e.g. DataType.JOINT_POSITIONS.
            timestamp: the Unix timestamp of the sample.
            joined_names: a single ``\0``-joined string of joint names.
            values: a flat list of joint values.
        """
        if not values:
            return
        robot_id = self._require_source("log_joints")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_joints(
            robot_id,
            self._robot_instance,
            data_type,
            joined_names,
            values,
            timestamp_ns,
            timestamp,
        )

    def log_frame(
        self,
        data_type: str,
        name: str,
        width: int,
        height: int,
        dtype: str,
        payload: bytes | memoryview,
        timestamp: float,
    ) -> None:
        """Forward one video frame to the daemon.

        Args:
            data_type: Type of video data e.g. DataType.RGB_IMAGES.
            name: the camera/sensor name e.g. left_wrist_camera.
            width: Video frame width.
            height: Video frame height.
            dtype: The frame's original numpy dtype name (``image.dtype.name``),
                e.g. ``"uint8"`` for RGB or ``"float16"`` / ``"float32"`` for
                depth. Parsed once at the native boundary so every internal
                Rust component works with a strongly typed representation.
            payload: Raw video frame bytes.
            timestamp: the Unix timestamp of the sample.
        """
        robot_id = self._require_source("log_frame")
        timestamp_ns = int(timestamp * 1_000_000_000)
        native = _load_native()
        try:
            native.log_frame(
                robot_id,
                self._robot_instance,
                data_type,
                name,
                int(width),
                int(height),
                dtype,
                payload,
                timestamp_ns,
                timestamp,
            )
        except native.LoggingStalledError as error:
            raise LoggingStalledError(str(error)) from error

    def log_json(
        self,
        data_type: str,
        name: str,
        payload: bytes,
        timestamp: float,
    ) -> None:
        """Forward one JSON sample to the daemon.

        The generic single-sample path for any non-joint, non-video data type
        (scalars, poses, gripper amounts, language, point clouds, ...). The
        ``data_type`` is an opaque wire label and ``payload`` is already
        serialized, so the daemon stores it verbatim as a per-trace JSON sample.
        """
        robot_id = self._require_source("log_json")
        timestamp_ns = int(timestamp * 1_000_000_000)
        _load_native().log_json(
            robot_id,
            self._robot_instance,
            data_type,
            name,
            payload,
            timestamp_ns,
            timestamp,
        )

    def cancel_recording(self, timestamp: float | None = None) -> None:
        """Cancel the source's active recording — the daemon discards it.

        A cancel is a recording stop that discards data, so ``timestamp``
        behaves exactly like ``stop_recording``'s: it optionally pins the
        recording's capture stop time (Unix seconds); when ``None`` the producer
        stamps wall-clock now.
        """
        if not self._robot_id:
            return
        timestamp_ns = int(timestamp * 1_000_000_000) if timestamp is not None else None
        _load_native().cancel_recording(
            self._robot_id, self._robot_instance, timestamp_ns
        )

    def _require_source(self, operation: str) -> str:
        """Return the active source's robot id or raise if logging before start."""
        if not self._robot_id:
            raise RuntimeError(
                f"{operation} called before start_recording set a source."
            )
        return self._robot_id

    def stop_recording(self, timestamp: float | None = None) -> None:
        """Publish one ``StopRecording`` tagged with the source.

        The daemon uses the publish-clock stop boundary to close the recording
        window. ``timestamp`` is the recording's *capture* stop time and is
        separate from that boundary, never used for window membership.

        This publishes the stop only; :meth:`flush_source` seals the writer's
        tail chunks and must be called straight after, so the caller can close
        its own logging gate in between.
        """
        if not self._robot_id:
            return
        timestamp_ns = int(timestamp * 1_000_000_000) if timestamp is not None else None
        _load_native().stop_recording(
            self._robot_id, self._robot_instance, timestamp_ns
        )

    def flush_source(self) -> None:
        """Run the writer's deferred tail-chunk barrier for the bound source.

        Mandatory after :meth:`stop_recording` — tail chunks are sealed nowhere
        else. No-op before a source is bound.
        """
        if not self._robot_id:
            return
        _load_native().flush_source(self._robot_id, self._robot_instance)

    def get_recording_id(
        self,
        timestamp_ns: int | None = None,
        timeout_s: float = 30.0,
    ) -> str | None:
        """Resolve the daemon-owned cloud recording id for this source.

        The producer never sees the cloud recording id — the
        daemon allocates it and POSTs ``/recording/start`` asynchronously. This
        asks the daemon over the native ``queries`` request-response service for
        the id of the recording identified by this source and the capture
        ``timestamp_ns`` marker (defaulting to the marker captured at
        ``start_recording``). The daemon answers authoritatively from its own
        state; the native call blocks (with the GIL released) until the id is
        minted or ``timeout_s`` elapses.

        It MAY block and is for non-performance-critical paths only (tests,
        ``nc.stop_recording(wait=True)``). Returns ``None`` on timeout.
        """
        if not self._robot_id:
            return None
        marker_ns = (
            timestamp_ns if timestamp_ns is not None else self._recording_marker_ns
        )
        return _load_native().get_recording_id(
            self._robot_id, self._robot_instance, marker_ns, timeout_s
        )

    def close(self) -> None:
        """Release resources owned by this instance.

        The native producer is process-global and outlives any single context,
        so there is nothing to tear down here.
        """
