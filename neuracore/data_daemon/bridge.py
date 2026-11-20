"""Recording-scoped context for driving the data daemon.

The SDK's only interface to the daemon. Every method forwards straight through
to the ``_data_bridge`` PyO3 extension over iceoryx2; see
[rust/data_daemon_bridge/src/lib.rs](../../rust/data_daemon_bridge/src/lib.rs)
for the native side.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Literal, NamedTuple, Protocol, cast

from neuracore.data_daemon.daemon_control import ensure_daemon_running

logger = logging.getLogger(__name__)

_DATA_BRIDGE_MODULE: ModuleType | None = None

_DATA_BRIDGE_IMPORT_HINT = (
    "neuracore.data_daemon._data_bridge is not available. The Rust extension "
    "is bundled in the neuracore wheel (prebuilt for Linux x86_64/aarch64 and "
    "macOS); reinstall with `pip install neuracore` on one of "
    "those platforms, or build it into a source checkout with "
    "`bash rust/scripts/build_wheel_artefacts.sh && pip install -e .`."
)


def _load_native() -> ModuleType:
    """Lazily import and cache the PyO3 data daemon bridge module for the process.

    The compiled extension ships inside the ``neuracore`` wheel (prebuilt for
    Linux x86_64/aarch64 and macOS), so this raises a helpful error when
    it is absent (e.g. a source install without the Rust build).
    """
    global _DATA_BRIDGE_MODULE
    if _DATA_BRIDGE_MODULE is None:
        try:
            _DATA_BRIDGE_MODULE = import_module("neuracore.data_daemon._data_bridge")
        except ImportError as error:
            raise RuntimeError(_DATA_BRIDGE_IMPORT_HINT) from error
    return _DATA_BRIDGE_MODULE


class LiveRecording(NamedTuple):
    """The recording a source has open, as the daemon reports it.

    Attributes:
        recording_index: The daemon's own primary key, once data has opened the
            recording's window. It exists before the cloud id does, so it is
            what tells two consecutive recordings of one source apart. ``None``
            for a recording the backend started that this host has not yet
            published for.
        cloud_recording_id: The cloud handle, once ``/recording/start`` has been
            notified. ``None`` while the recording is still local-only.
        start_timestamp_us: The recording's start marker in microseconds, as
            returned by :meth:`RecordingContext.start_recording`.
    """

    recording_index: int | None
    cloud_recording_id: str | None
    start_timestamp_us: int | None


class RecordingStateUnavailableError(RuntimeError):
    """The daemon did not answer a recording-state query in time."""


def query_recording_state(
    robot_id: str, robot_instance: int, timeout_s: float = 0.0
) -> LiveRecording | None:
    """Which recording, if any, a source currently has open.

    The daemon owns recording state — including the recordings the backend
    started, which it alone subscribes to — so this is how a process learns
    about a recording it did not itself start or stop.

    Reads the cache the bridge keeps, the same one
    :meth:`RecordingContext.recording_epoch` reads, so a call costs a memory
    read. Only the first read for a source can block, and only briefly: an entry
    nothing has answered for is queried on the calling thread. The cache is
    refreshed off-thread within 50 ms, so a recording started elsewhere shows up
    that late; a recording this process bracketed is exact immediately.

    Args:
        robot_id: Robot identifier of the source.
        robot_instance: Robot instance of the source.
        timeout_s: How long to wait for the daemon to mint the open recording's
            cloud id, which it does asynchronously. The wait happens in the
            bridge with the GIL released, and only while an id is missing.

    Returns:
        The open recording, or ``None`` when the source has none.

    Raises:
        RecordingStateUnavailableError: nothing has ever answered for this
            source. Silence is not "not recording", and callers must not read it
            as such.
    """
    reply = _load_native().recording_state(robot_id, robot_instance, timeout_s)
    if reply is None:
        raise RecordingStateUnavailableError(
            f"the daemon has not answered for {robot_id} instance {robot_instance}"
        )
    live = reply["recording"]
    if live is None:
        return None
    return LiveRecording(
        recording_index=live["recording_index"],
        cloud_recording_id=live["cloud_recording_id"],
        start_timestamp_us=live["start_timestamp_us"],
    )


# How long one robot or dataset request waits for the daemon to answer. The
# daemon answers from its own state store, so only a stalled daemon is slow.
REGISTRY_TIMEOUT_S = 5.0


class RegistryUnavailableError(RuntimeError):
    """The daemon did not answer a robot or dataset request in time."""


class RegistryRefusedError(RuntimeError):
    """The daemon refused to connect a robot or select a dataset."""


RowStatus = Literal["pending", "done", "failed"]


class RowState(NamedTuple):
    """A robot or dataset row as the daemon holds it.

    Attributes:
        status: ``"pending"`` until the server confirms the row, ``"done"``
            once it has, ``"failed"`` once the server refused it for good.
        cloud_id: The server's id, once the row is ``"done"``.
        reason: Why a pending row is held, or why the row failed, as a reason
            code such as ``"robot_archived"``.
    """

    status: RowStatus
    cloud_id: str | None
    reason: str | None


class _NativeRowState(Protocol):
    """A row's state as the native bridge returns it."""

    @property
    def status(self) -> str: ...

    @property
    def cloud_id(self) -> str | None: ...

    @property
    def reason(self) -> str | None: ...


class _NativeRegistry(Protocol):
    """The robot and dataset registry calls of the ``_data_bridge`` extension.

    Every argument is keyword-only on the native side, so a call cannot pass
    one in the wrong place.
    """

    RegistryUnavailableError: type[Exception]
    RegistryRefusedError: type[Exception]

    def connect_robot(
        self,
        *,
        org_id: str,
        name: str,
        shared: bool,
        instance: int,
        overwrite_urdf: bool,
        package_path: str | None,
        timeout_s: float,
    ) -> int: ...

    def select_dataset(
        self,
        *,
        org_id: str,
        name: str,
        shared: bool,
        description: str | None,
        tags: list[str],
        timeout_s: float,
    ) -> int: ...

    def dataset_for_cloud_id(
        self, *, org_id: str, cloud_dataset_id: str, timeout_s: float
    ) -> int: ...

    def robot_state(
        self, *, robot_index: int, timeout_s: float
    ) -> _NativeRowState | None: ...

    def dataset_state(
        self, *, dataset_index: int, timeout_s: float
    ) -> _NativeRowState | None: ...


def _registry() -> _NativeRegistry:
    """The native bridge's registry calls."""
    return cast(_NativeRegistry, _load_native())


@contextmanager
def _registry_errors(native: _NativeRegistry) -> Iterator[None]:
    """Raise the native bridge's registry errors as the SDK's own."""
    try:
        yield
    except native.RegistryUnavailableError as error:
        raise RegistryUnavailableError(str(error)) from error
    except native.RegistryRefusedError as error:
        raise RegistryRefusedError(str(error)) from error


def _row_state(state: _NativeRowState | None) -> RowState | None:
    if state is None:
        return None
    return RowState(
        status=cast(RowStatus, state.status),
        cloud_id=state.cloud_id,
        reason=state.reason,
    )


def connect_robot_row(
    *,
    org_id: str,
    name: str,
    shared: bool,
    instance: int,
    overwrite_urdf: bool,
    package_path: str | None,
) -> int:
    """Connect a robot instance through the daemon and return its row's index.

    Starts the daemon if it is not running. The daemon finds or creates the
    row and has the server confirm it in the background, so this returns
    without waiting on the network.

    Args:
        org_id: The organization to connect the robot in.
        name: The robot's name.
        shared: Whether the robot is shared.
        instance: The robot instance.
        overwrite_urdf: Whether the saved package replaces the server's URDF.
        package_path: The URDF package saved for the server, if any. The
            daemon deletes it once the server has it.

    Returns:
        The robot row's index.

    Raises:
        RegistryUnavailableError: the daemon did not answer.
        RegistryRefusedError: the daemon refused the request.
    """
    ensure_daemon_running()
    native = _registry()
    with _registry_errors(native):
        return native.connect_robot(
            org_id=org_id,
            name=name,
            shared=shared,
            instance=instance,
            overwrite_urdf=overwrite_urdf,
            package_path=package_path,
            timeout_s=REGISTRY_TIMEOUT_S,
        )


def select_dataset_row(
    *,
    org_id: str,
    name: str,
    shared: bool,
    description: str | None,
    tags: list[str],
) -> int:
    """Select a dataset through the daemon and return its row's index.

    Starts the daemon if it is not running. The daemon finds or creates the
    row and has the server find the dataset, or create it if it has none of
    that name, in the background, so this returns without waiting on the
    network.

    Args:
        org_id: The organization to find or create the dataset in.
        name: The dataset's name.
        shared: Whether the dataset is shared.
        description: The description to create the dataset with.
        tags: The tags to create the dataset with.

    Returns:
        The dataset row's index.

    Raises:
        RegistryUnavailableError: the daemon did not answer.
        RegistryRefusedError: the daemon refused the request.
    """
    ensure_daemon_running()
    native = _registry()
    with _registry_errors(native):
        return native.select_dataset(
            org_id=org_id,
            name=name,
            shared=shared,
            description=description,
            tags=tags,
            timeout_s=REGISTRY_TIMEOUT_S,
        )


def dataset_row_for_cloud_id(*, org_id: str, cloud_dataset_id: str) -> int:
    """The daemon's row for a dataset read from the server, by its server id.

    Starts the daemon if it is not running. The daemon finds the row that has
    this id, or makes one the server has confirmed already, so a recording
    into the dataset points at a row like any other.

    Args:
        org_id: The organization the dataset is recorded in.
        cloud_dataset_id: The server's id for the dataset.

    Returns:
        The dataset row's index.

    Raises:
        RegistryUnavailableError: the daemon did not answer.
        RegistryRefusedError: the daemon refused the request.
    """
    ensure_daemon_running()
    native = _registry()
    with _registry_errors(native):
        return native.dataset_for_cloud_id(
            org_id=org_id,
            cloud_dataset_id=cloud_dataset_id,
            timeout_s=REGISTRY_TIMEOUT_S,
        )


def robot_row_state(robot_index: int) -> RowState | None:
    """The state of a robot row, or ``None`` when the daemon has no such row.

    Never starts the daemon: a caller may poll this, and the row only exists
    if the daemon ran.

    Raises:
        RegistryUnavailableError: the daemon did not answer.
        RegistryRefusedError: the daemon could not read the row.
    """
    native = _registry()
    with _registry_errors(native):
        state = native.robot_state(
            robot_index=robot_index, timeout_s=REGISTRY_TIMEOUT_S
        )
    return _row_state(state)


def dataset_row_state(dataset_index: int) -> RowState | None:
    """The state of a dataset row, as :func:`robot_row_state`.

    Raises:
        RegistryUnavailableError: the daemon did not answer.
        RegistryRefusedError: the daemon could not read the row.
    """
    native = _registry()
    with _registry_errors(native):
        state = native.dataset_state(
            dataset_index=dataset_index, timeout_s=REGISTRY_TIMEOUT_S
        )
    return _row_state(state)


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
        dataset_index: int | None = None,
        dataset_name: str | None = None,
        *,
        timestamp_us: int,
    ) -> int:
        """Announce a recording to the daemon for a source.

        Publishes one ``StartRecording`` envelope tagged with the source
        ``(robot_id, robot_instance)``. The daemon gets the cloud id from the backend.

        ``timestamp_us`` is the recording's start in microseconds on the
        caller's capture clock, stored and returned as this recording's start
        marker. It does **not** bound the recording window,
        which the daemon takes from a publish stamp inside this call.

        Returns:
            The start in microseconds the daemon stored as this recording's
            ``start_timestamp_us``, which tells it apart from its predecessor
            before either has a cloud id.
        """
        ensure_daemon_running()
        self.bind_source(robot_id, robot_instance)

        return _load_native().start_recording(
            robot_id,
            robot_instance,
            robot_name,
            dataset_index,
            dataset_name,
            timestamp_us,
        )

    def log_joints(
        self,
        data_type: str,
        timestamp_us: int,
        joined_names: str,
        values: list[float],
    ) -> None:
        r"""Forward a batch of joint scalar samples to the daemon.

        Args:
            data_type:  Type of joint data e.g. DataType.JOINT_POSITIONS.
            timestamp_us: the sample's capture time in microseconds.
            joined_names: a single ``\0``-joined string of joint names.
            values: a flat list of joint values.
        """
        if not values:
            return
        robot_id = self._require_source("log_joints")
        _load_native().log_joints(
            robot_id,
            self._robot_instance,
            data_type,
            joined_names,
            values,
            timestamp_us,
        )

    def log_frame(
        self,
        data_type: str,
        name: str,
        width: int,
        height: int,
        dtype: str,
        payload: bytes | memoryview,
        timestamp_us: int,
        depth_scale_m: float | None = None,
        intrinsics: list[float] | None = None,
        extrinsics: list[float] | None = None,
    ) -> None:
        """Forward one video frame to the daemon.

        Args:
            data_type: Type of video data e.g. DataType.RGB_IMAGES.
            name: the camera/sensor name e.g. left_wrist_camera.
            width: Video frame width.
            height: Video frame height.
            dtype: The frame's original numpy dtype name (``image.dtype.name``),
                ``"uint8"`` for RGB or ``"uint16"`` for depth in sensor units.
                Parsed once at the native boundary so every internal Rust
                component works with a strongly typed representation.
            payload: Raw video frame bytes.
            timestamp_us: the frame's capture time in microseconds.
            depth_scale_m: Meters per sensor unit, required for uint16 depth.
            intrinsics: Optional row-major 3x3 camera intrinsics, flattened to
                9 values.
            extrinsics: Optional row-major 4x4 camera extrinsics, flattened to
                16 values.
        """
        robot_id = self._require_source("log_frame")
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
                timestamp_us,
                depth_scale_m,
                intrinsics,
                extrinsics,
            )
        except native.LoggingStalledError as error:
            raise LoggingStalledError(str(error)) from error

    def log_json(
        self,
        data_type: str,
        name: str,
        payload: bytes,
        timestamp_us: int,
    ) -> None:
        """Forward one JSON sample to the daemon.

        The generic single-sample path for any non-joint, non-video data type
        (scalars, poses, gripper amounts, language, point clouds, ...). The
        ``data_type`` is an opaque wire label and ``payload`` is already
        serialized, so the daemon stores it verbatim as a per-trace JSON sample.
        """
        robot_id = self._require_source("log_json")
        _load_native().log_json(
            robot_id,
            self._robot_instance,
            data_type,
            name,
            payload,
            timestamp_us,
        )

    def cancel_recording(self, timestamp_us: int) -> None:
        """Cancel the source's active recording — the daemon discards it.

        A cancel is a recording stop that discards data, so ``timestamp_us``
        behaves exactly like ``stop_recording``'s: it is the recording's
        capture stop time in microseconds.
        """
        if not self._robot_id:
            return
        _load_native().cancel_recording(
            self._robot_id, self._robot_instance, timestamp_us
        )

    def recording_epoch(self) -> int | None:
        """The recording this source has open, as this process last saw it.

        A counter, not a property of the recording — compare it, don't read
        it, and don't carry it between processes. It changes when the source
        crosses a recording boundary, whoever opened it; ``None`` means no open
        recording, or that nothing has answered yet, which callers must treat
        the same.

        A cache read, cheap enough for the log path: the daemon is asked off
        the caller's thread, and a recording this process bracketed is known
        without asking at all.
        """
        if self._robot_id is None:
            return None
        return _load_native().recording_epoch(self._robot_id, self._robot_instance)

    def _require_source(self, operation: str) -> str:
        """Return the active source's robot id or raise if logging before start."""
        if not self._robot_id:
            raise RuntimeError(
                f"{operation} called before start_recording set a source."
            )
        return self._robot_id

    def stop_recording(self, timestamp_us: int, tags: list[str] | None = None) -> None:
        """Publish one ``StopRecording`` tagged with the source.

        The daemon uses the publish-clock stop boundary to close the recording
        window. ``timestamp_us`` is the recording's *capture* stop time in
        microseconds and is separate from that boundary, never used for window
        membership. ``tags`` are normalized user tag names that the daemon sends
        to the backend with the stop.

        This publishes the stop only; :meth:`flush_source` seals the writer's
        tail chunks and must be called straight after, so the caller can close
        its own logging gate in between.
        """
        if not self._robot_id:
            return
        _load_native().stop_recording(
            self._robot_id, self._robot_instance, timestamp_us, tags or []
        )

    def flush_source(self) -> None:
        """Run the writer's deferred tail-chunk barrier for the bound source.

        Mandatory after :meth:`stop_recording` — tail chunks are sealed nowhere
        else. No-op before a source is bound.
        """
        if not self._robot_id:
            return
        _load_native().flush_source(self._robot_id, self._robot_instance)

    def close(self) -> None:
        """Release resources owned by this instance.

        The native producer is process-global and outlives any single context,
        so there is nothing to tear down here.
        """
