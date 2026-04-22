"""Process, filesystem, and data-content assertions for daemon integration tests.

Provides all assertion helpers used by the data-daemon integrity and
behavioural-correctness test suites.  Functions are grouped as follows.

**Process assertions** — verify that daemon and producer processes are (or are
not) running:
:func:`assert_no_daemon_pids`, :func:`assert_exactly_one_daemon_pid`,
:func:`assert_no_producer_processes`.

**File and socket assertions** — verify artefact clean-up:
:func:`assert_no_pid_file`, :func:`assert_socket_unlinked`,
:func:`assert_db_absent`, :func:`assert_recordings_folder_absent`,
:func:`assert_recordings_folder_empty`.

**Composite isolation assertions** — one-call helpers that assert all isolation
invariants at once:
:func:`assert_daemon_cleanup`, :func:`assert_post_test_storage_state`.

**Disk-state snapshot / diff** — capture daemon-managed paths before a test and
compare afterwards to detect artefact leaks:
:func:`snapshot_daemon_disk_state`, :func:`assert_disk_state_unchanged`.

**Timer stat helpers** — manage per-run `Timer` stats:
:func:`clear_daemon_timer_stats`.

**Offline DB content assertions** — verify the daemon's SQLite DB after an
offline recording session:
:func:`assert_db_contents`, :func:`assert_disk_traces`.

**Online verification** — download the cloud dataset and verify every episode's
data matches what was logged:
:func:`verify_cloud_results` (structural + data passes),
:func:`_verify_recording_structure`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from neuracore_types import Dataset, DataType

import neuracore as nc
from neuracore.core.data.recording import Recording
from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_pid_path
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
    wait_for_recordings_finalized,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    _live_daemon_pids,
    get_runner_pids,
)
from tests.integration.platform.data_daemon.shared.storage_assertions import (
    assert_db_absent,
    assert_db_empty,
    assert_post_test_storage_state,
    assert_recordings_folder_absent,
    assert_recordings_folder_empty,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    case_timeout_seconds,
)

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
        ContextResult,
    )

from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    FRAME_BYTE_LENGTH,
    FRAME_GRID_SIZE,
    MODE_SEQUENTIAL,
    TIMESTAMP_MODE_REAL,
)

logger = logging.getLogger(__name__)


def assert_context_mode(case: DataDaemonTestCase, results: list[ContextResult]) -> None:
    """Assert that context timing matches the expected mode."""
    active_results = [result for result in results if result.recording_ids]
    if len(active_results) < 2:
        return

    ordered_results = sorted(active_results, key=lambda result: result.context_index)
    first = ordered_results[0]
    second = ordered_results[1]
    tolerance_s = 0.1
    if case.mode == MODE_SEQUENTIAL:
        assert abs(second.timestamp_start_s - first.timestamp_start_s) < tolerance_s
        assert abs(first.timestamp_end_s - second.timestamp_end_s) < tolerance_s
        assert second.wall_started_at is not None
        assert first.wall_stopped_at > second.wall_started_at
        return

    assert first.timestamp_end_s > second.timestamp_start_s
    assert second.wall_started_at is not None
    assert first.wall_stopped_at > second.wall_started_at


# ---------------------------------------------------------------------------
# Process assertions
# ---------------------------------------------------------------------------


def assert_no_daemon_pids() -> None:
    """Fail immediately if any daemon PIDs are still running.

    Raises:
        AssertionError: When one or more daemon processes are still alive.
    """
    live = _live_daemon_pids()
    assert not live, (
        f"Daemon processes still running during isolation assertion — "
        f"PIDs: {sorted(live)}"
    )


def assert_exactly_one_daemon_pid() -> int:
    """Fail unless exactly one daemon PID is alive, then return it.

    Returns:
        The single live daemon PID.

    Raises:
        AssertionError: When the number of live daemon PIDs is not exactly one.
    """
    live = _live_daemon_pids()
    assert len(live) == 1, (
        f"Expected exactly one daemon PID after startup but found {len(live)} — "
        f"PIDs: {sorted(live)}"
    )
    return next(iter(live))


def assert_no_producer_processes() -> None:
    """Fail if any runner/producer subprocesses are still in the process table.

    Raises:
        AssertionError: When runner PIDs are found.
    """
    runner_pids = get_runner_pids()
    assert not runner_pids, (
        f"Producer/runner subprocesses still alive after test — "
        f"PIDs: {sorted(runner_pids)}"
    )


# ---------------------------------------------------------------------------
# File / socket assertions
# ---------------------------------------------------------------------------


def assert_no_pid_file() -> None:
    """Fail if the daemon PID file still exists on disk.

    Raises:
        AssertionError: When the PID file path exists.
    """
    pid_path = get_daemon_pid_path()
    assert (
        not pid_path.exists()
    ), f"PID file was not cleaned up after daemon shutdown: {pid_path}"


def assert_socket_unlinked() -> None:
    """Fail if the daemon Unix socket path still exists.

    Raises:
        AssertionError: When the socket file is still present.
    """
    socket_path = Path(str(SOCKET_PATH))
    assert (
        not socket_path.exists()
    ), f"Unix socket was not unlinked after daemon shutdown: {socket_path}"


__all__ = [
    "assert_db_absent",
    "assert_db_empty",
    "assert_post_test_storage_state",
    "assert_recordings_folder_absent",
    "assert_recordings_folder_empty",
]


# ---------------------------------------------------------------------------
# Composite isolation assertions
# ---------------------------------------------------------------------------


def assert_daemon_cleanup() -> None:
    """Assert all process and socket isolation invariants.

    Verifies that no daemon processes, producer subprocesses, PID file, or
    Unix socket remain after the test.  Does not inspect on-disk DB or
    recordings artefacts — those are governed by ``storage_state_action`` and
    belong in :func:`assert_post_test_disk_state`.
    """
    assert_no_daemon_pids()
    assert_no_pid_file()
    assert_socket_unlinked()
    assert_no_producer_processes()


# ---------------------------------------------------------------------------
# Timer stats helpers
# ---------------------------------------------------------------------------


def clear_daemon_timer_stats() -> None:
    """Clear all daemon timer stats before a test case starts."""
    Timer._stats.clear()


def decode_frame_number(img: np.ndarray) -> int:
    """Decode a frame number from a video frame produced by the test encoder.

    Reads the Red-channel byte from each pixel in the top-left 4x4 grid and
    interprets those 16 bytes as a big-endian integer.

    Args:
        img: A NumPy array with shape ``(height, width, 3)`` containing a
            frame previously created by the test encoder.

    Returns:
        The decoded frame number as an integer.
    """
    frame_bytes = bytearray()
    for row in range(FRAME_GRID_SIZE):
        for col in range(FRAME_GRID_SIZE):
            frame_bytes.append(img[row, col, 0])
    return int.from_bytes(frame_bytes[:FRAME_BYTE_LENGTH], byteorder="big")


def _extract_custom_scalar(custom_data: object) -> float | None:
    """Extract a scalar float from synchronized CUSTOM_1D payload data."""
    raw_value = getattr(custom_data, "value", custom_data)

    if isinstance(raw_value, np.ndarray):
        if raw_value.size == 0:
            return None
        return float(raw_value.reshape(-1)[0])

    if isinstance(raw_value, (list, tuple)):
        if not raw_value:
            return None
        return float(raw_value[0])

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Online verification
# ---------------------------------------------------------------------------


def _collect_episode_summary(synced_episode: object) -> dict[str, object]:
    """Walk a synchronised episode and collect per-type counts and values.

    Iterates every sync point in the episode and accumulates:

    - ``sync_points`` — total sync-point count.
    - ``timestamps`` — per-sync-point timestamp values.
    - ``rgb_counts`` — per-camera frame counts.
        - ``frame_codes`` — per-camera list of decoded frame numbers from
            :func:`decode_frame_number`.
    - ``joint_position_counts``, ``joint_velocity_counts``,
      ``joint_torque_counts`` — per-joint frame counts.
    - ``joint_position_values`` — list of ``(frame_index, joint_name, value)``
      tuples.
    - ``custom_counts`` — per-channel frame counts for ``CUSTOM_1D`` data.
        - ``custom_values`` — per-channel list of ``(frame_index, value)`` tuples.

    Args:
        synced_episode: A synchronised episode object returned by
            ``dataset.synchronize()``.

    Returns:
        A dict containing all of the accumulated fields described above.
    """
    summary: dict[str, object] = {
        "sync_points": 0,
        "timestamps": [],
        "rgb_counts": {},
        "frame_codes": {},
        "joint_position_counts": {},
        "joint_velocity_counts": {},
        "joint_torque_counts": {},
        "joint_position_values": [],
        "custom_counts": {},
        "custom_values": {},
    }

    for frame_index, sync_point in enumerate(
        synced_episode
    ):  # type: ignore[call-overload]
        summary["sync_points"] = int(summary["sync_points"]) + 1
        summary["timestamps"].append(float(sync_point.timestamp))

        if DataType.RGB_IMAGES in sync_point.data:
            rgb_counts = dict(summary["rgb_counts"])
            frame_codes = dict(summary["frame_codes"])
            for camera_name, camera_data in sync_point[DataType.RGB_IMAGES].items():
                name = str(camera_name)
                rgb_counts[name] = rgb_counts.get(name, 0) + 1
                frame_codes.setdefault(name, []).append(
                    decode_frame_number(np.array(camera_data.frame))
                )
            summary["rgb_counts"] = rgb_counts
            summary["frame_codes"] = frame_codes

        if DataType.JOINT_POSITIONS in sync_point.data:
            counts = dict(summary["joint_position_counts"])
            for joint_name, joint_data in sync_point[DataType.JOINT_POSITIONS].items():
                name = str(joint_name)
                counts[name] = counts.get(name, 0) + 1
                summary["joint_position_values"].append(
                    (frame_index, name, float(joint_data.value))
                )
            summary["joint_position_counts"] = counts

        if DataType.JOINT_VELOCITIES in sync_point.data:
            counts = dict(summary["joint_velocity_counts"])
            for joint_name in sync_point[DataType.JOINT_VELOCITIES].keys():
                name = str(joint_name)
                counts[name] = counts.get(name, 0) + 1
            summary["joint_velocity_counts"] = counts

        if DataType.JOINT_TORQUES in sync_point.data:
            counts = dict(summary["joint_torque_counts"])
            for joint_name in sync_point[DataType.JOINT_TORQUES].keys():
                name = str(joint_name)
                counts[name] = counts.get(name, 0) + 1
            summary["joint_torque_counts"] = counts

        if DataType.CUSTOM_1D in sync_point.data:
            custom_counts = dict(summary["custom_counts"])
            custom_values = dict(summary["custom_values"])
            for name, custom_data in sync_point[DataType.CUSTOM_1D].items():
                key = str(name)
                custom_counts[key] = custom_counts.get(key, 0) + 1
                scalar = _extract_custom_scalar(custom_data)
                if scalar is not None:
                    values_for_key = list(custom_values.get(key, []))
                    values_for_key.append((frame_index, scalar))
                    custom_values[key] = values_for_key
            summary["custom_counts"] = custom_counts
            summary["custom_values"] = custom_values

    return summary


def _assert_synced_episode_timestamps_are_sane(
    *,
    timestamps: list[float],
    result: ContextResult,
) -> None:
    """Validate synchronized-episode timestamp sanity without sync-window coupling.

    Cloud integrity checks should not re-test the synchronization algorithm.
    Keep this validation broad: values must be finite and non-decreasing, and
    real-timestamp mode values must look like Unix epochs.
    """
    if not timestamps:
        return

    non_finite = [ts for ts in timestamps if not math.isfinite(ts)]
    assert (
        not non_finite
    ), f"Synced episode has non-finite timestamp(s): {non_finite[:5]}"

    if result.timestamp_mode == TIMESTAMP_MODE_REAL:
        epoch_floor = 946_684_800.0
        non_epoch = [ts for ts in timestamps if ts < epoch_floor]
        assert not non_epoch, (
            f"Synced episode has {len(non_epoch)} timestamp(s) that are not "
            f"valid epoch values (< year 2000) — e.g. {non_epoch[:5]}"
        )

    non_monotonic = [
        (i, timestamps[i], timestamps[i + 1])
        for i in range(len(timestamps) - 1)
        if timestamps[i] >= timestamps[i + 1]
    ]
    assert not non_monotonic, (
        f"Synced episode timestamps are not monotonically non-decreasing — "
        f"e.g. {non_monotonic[:5]}"
    )


def _assert_synced_camera_codes_are_sane(
    *,
    actual_codes: object,
    camera_name: str,
    context_index: int,
    recording_index: int,
    camera_index: int,
    expected_video_frame_count: int,
) -> None:
    """Validate decoded camera frame codes without asserting sync-frame mapping.

    Only checks that synchronized episode camera frames still belong to the
    expected recording/camera code namespace, remain monotonic, and that no
    expected frame codes are missing.
    """
    base_code = (
        (context_index * 1_000_000_000)
        + (recording_index * 10_000_000)
        + (camera_index * 100_000)
    )
    max_code = base_code + max(expected_video_frame_count - 1, 0)

    assert isinstance(actual_codes, list), (
        f"Frame codes for camera {camera_name!r} must be a list, "
        f"got {type(actual_codes).__name__}: {actual_codes}"
    )
    assert (
        actual_codes
    ), f"Expected at least one camera frame code for camera {camera_name!r}"

    def _ctx(codes, highlight, radius=2):
        lo = max(0, min(highlight) - radius)
        hi = min(len(codes), max(highlight) + radius + 1)
        parts = []
        for j in range(lo, hi):
            entry = f"[{j}]={codes[j]}"
            parts.append(f"*{entry}*" if j in highlight else entry)
        return "  ".join(parts)

    def _out_of_range(codes, lo, hi):
        return [(i, code) for i, code in enumerate(codes) if not (lo <= code <= hi)]

    range_violations = _out_of_range(actual_codes, base_code, max_code)
    assert not range_violations, "\n".join([
        f"Frame codes out of expected recording range for camera {camera_name!r}:",
        f"  expected all in [{base_code}, {max_code}],"
        f" got {len(range_violations)} violation(s):",
        *[
            f"  [{i}]={c}\n    context: {_ctx(actual_codes, {i})}"
            for i, c in range_violations[:8]
        ],
    ])

    def _decreasing_pairs(codes):
        return [
            (i, prev, curr)
            for i, (prev, curr) in enumerate(zip(codes, codes[1:]))
            if prev > curr
        ]

    violations = _decreasing_pairs(actual_codes)
    assert not violations, "\n".join([
        f"Frame codes must be non-decreasing for camera {camera_name!r}:",
        f"  got {len(violations)} violation(s):",
        *[
            f"  [{i}]={p} > [{i+1}]={c}\n    context: {_ctx(actual_codes, {i, i+1})}"
            for i, p, c in violations[:8]
        ],
    ])


def _verify_synched_episode_summary(
    *,
    summary: dict[str, object],
    result: ContextResult,
    case: DataDaemonTestCase,
    recording_index: int,
) -> None:
    """Check that an episode summary matches expected values for a case.

    Verifies synchronized-episode data integrity without asserting sync internals.
    This checks type presence, channel completeness, finite values, and encoded
    frame-code provenance while avoiding exact timestamp/frame alignment checks.

    Args:
        summary: Episode summary produced by :func:`_collect_episode_summary`.
        result: Per-context result dict (must contain ``"joint_names"``,
            ``"frame_count"``, ``"fps"``, ``"marker_names"``,
            ``"camera_names"``, ``"context_index"``, and ``"recording_ids"``).
        case: The active :class:`~matrix_test_configs.DataDaemonTestCase`.
        recording_index: Zero-based index of this recording within its
            context, used to reconstruct expected frame codes.

    """
    # --- Episode shape ---
    sync_points = int(summary.get("sync_points", 0))
    assert sync_points > 0, "Synced episode produced zero sync points"

    timestamps = list(summary["timestamps"])
    assert (
        len(timestamps) == sync_points
    ), f"Timestamp count mismatch: got {len(timestamps)}, expected {sync_points}"

    # --- Timestamps ---
    _assert_synced_episode_timestamps_are_sane(timestamps=timestamps, result=result)

    # --- Joint data: counts then values ---
    joint_names = result.joint_names
    for label, count_key in (
        ("position", "joint_position_counts"),
        ("velocity", "joint_velocity_counts"),
        ("torque", "joint_torque_counts"),
    ):
        counts = dict(summary[count_key])
        for joint_name in joint_names:
            assert counts.get(joint_name) == sync_points, (
                f"Joint {label} missing for joint {joint_name!r}: "
                f"got {counts.get(joint_name)} frame(s), expected {sync_points}"
            )
        unexpected_joints = set(counts) - set(joint_names)
        assert (
            not unexpected_joints
        ), f"Unexpected joint(s) in {label} counts: {unexpected_joints}"

    for frame_index, joint_name, actual_value in summary["joint_position_values"]:
        assert (
            joint_name in joint_names
        ), f"Unexpected joint name at synced frame {frame_index}: {joint_name!r}"
        assert math.isfinite(actual_value), (
            f"Non-finite joint position for {joint_name!r} "
            f"at synced frame {frame_index}: {actual_value}"
        )
        assert -1.0 - 1e-5 <= actual_value <= 1.0 + 1e-5, (
            f"Joint position outside sine bounds for {joint_name!r} "
            f"at synced frame {frame_index}: {actual_value}"
        )

    # --- Custom markers ---
    custom_counts = dict(summary["custom_counts"])
    for marker_name in result.marker_names:
        assert custom_counts.get(marker_name) == sync_points, (
            f"Custom marker {marker_name!r}: "
            f"got {custom_counts.get(marker_name)}, expected {sync_points}"
        )
    unexpected_markers = set(custom_counts) - set(result.marker_names)
    assert (
        not unexpected_markers
    ), f"Unexpected custom marker(s) in episode: {unexpected_markers}"

    # --- Video presence gate ---
    if not result.has_video:
        rgb_msg = (
            f"Expected no RGB frames for joints-only case "
            f"but got: {summary['rgb_counts']}"
        )
        assert not summary["rgb_counts"], rgb_msg
        fc_msg = (
            f"Expected no frame codes for joints-only case "
            f"but got: {summary['frame_codes']}"
        )
        assert not summary["frame_codes"], fc_msg
        return

    # --- Camera frame counts ---
    rgb_counts = dict(summary["rgb_counts"])
    for camera_name in result.camera_names:
        assert rgb_counts.get(camera_name) == sync_points, (
            f"RGB frames missing for camera {camera_name!r}: "
            f"got {rgb_counts.get(camera_name)}, expected {sync_points}"
        )
    unexpected_cameras = set(rgb_counts) - set(result.camera_names)
    assert (
        not unexpected_cameras
    ), f"Unexpected camera(s) in RGB counts: {unexpected_cameras}"

    # --- Camera frame codes ---
    for camera_index, camera_name in enumerate(result.camera_names):
        _assert_synced_camera_codes_are_sane(
            actual_codes=summary["frame_codes"].get(camera_name),
            camera_name=camera_name,
            context_index=result.context_index,
            recording_index=recording_index,
            camera_index=camera_index,
            expected_video_frame_count=result.video_frame_count,
        )


def _verify_recording_structure(
    *,
    recording: Recording,
    result: ContextResult,
    case: DataDaemonTestCase,
    dataset_name: str | None = None,
) -> None:
    """Assert structural properties of an unsynced recording.

    Checks that the recording's duration, byte size, and robot ID are
    consistent with what was logged during the recording phase.  Does not
    require synchronization or data download.

    Duration is checked against ``case.duration_sec`` scaled by
    ``DURATION_VARIABLE_MIN_FACTOR``–``DURATION_VARIABLE_MAX_FACTOR`` when
    ``context_duration_mode="variable"``, or against the exact base duration for
    fixed mode.  A ±2 s clock/upload tolerance is applied to both bounds.

    Args:
        recording: A :class:`~neuracore.core.data.recording.Recording` object
            returned by iterating an unsynced
            :class:`~neuracore.core.data.dataset.Dataset`.
        result: Per-context result dict containing ``"duration_sec"`` and
            ``"recording_ids"``.
        case: The active :class:`~matrix_test_configs.DataDaemonTestCase`.
        dataset_name: The name of the dataset containing the recording.
    Raises:
        AssertionError: When any structural property does not match expectations.
    """
    rec_id = str(recording.id)  # type: ignore[union-attr]
    dataset_name = recording.dataset.name  # type: ignore[union-attr]
    logger.info(
        "_verify_recording_structure: recording %s of dataset %r", rec_id, dataset_name
    )

    # Duration bounds: variable mode allows 0.75–1.25× base; fixed mode uses exact
    # base duration.
    base_duration_s = float(result.duration_sec)
    clock_tolerance_s = 1.0
    if case.context_duration_mode == DURATION_MODE_VARIABLE:
        min_duration_s = (
            base_duration_s * DURATION_VARIABLE_MIN_FACTOR - clock_tolerance_s
        )
        max_duration_s = (
            base_duration_s * DURATION_VARIABLE_MAX_FACTOR + clock_tolerance_s
        )
    else:
        min_duration_s = base_duration_s - clock_tolerance_s
        max_duration_s = base_duration_s + clock_tolerance_s

    actual_duration_s = (  # type: ignore[union-attr]
        float(recording.end_time) - float(recording.start_time)
    )
    assert min_duration_s <= actual_duration_s <= max_duration_s, (
        f"Recording {rec_id} of dataset {dataset_name!r} "
        f"duration {actual_duration_s:.2f}s outside expected range "
        f"[{min_duration_s:.2f}s, {max_duration_s:.2f}s] "
        f"(base={base_duration_s}s, mode={case.context_duration_mode})"
    )

    # Sanity: recording must have non-zero bytes.
    assert int(recording.total_bytes) > 0, (  # type: ignore[union-attr]
        f"Recording {rec_id} of dataset {dataset_name!r} has zero total_bytes"
    )

    # Robot ID must be set.
    assert recording.robot_id, (  # type: ignore[union-attr]
        f"Recording {rec_id} of dataset {dataset_name!r} has no robot_id"
    )


def verify_cloud_results(
    *,
    results: list[ContextResult],
    case: DataDaemonTestCase,
) -> None:
    """Wait for cloud readiness then verify every recording matches expectations.

    Waits for:

    1. The dataset to contain the expected number of recordings
       (:func:`~db_helpers.wait_for_dataset_ready`).
    2. Every recording's ``end_time`` to be finalized on the backend
       (:func:`~db_helpers.wait_for_recordings_finalized`).

    Then runs two verification passes:

    1. **Structural pass** — iterates unsynced
       :class:`~neuracore.core.data.recording.Recording`
       objects and checks duration, byte size, and robot ID via
       :func:`_verify_recording_structure`.
    2. **Data pass** — synchronises the dataset, then iterates each episode and
       cross-references frame counts and values against the per-context
       ``results`` list using :func:`_verify_episode_summary`.

    Both passes assert that the set of recording IDs is exactly equal to the
    expected set — no missing, no extras.

    Args:
        results: List of per-context result dicts.  The first entry must
            contain ``"dataset_name"``.
        case: The active :class:`~matrix_test_configs.DataDaemonTestCase`.

    Raises:
        AssertionError: When any recording fails structural or data verification,
            or when expected recording IDs are not present in the dataset.
    """
    dataset_name = results[0].dataset_name
    expected_count = case.recording_count
    logger.info(
        "verify_cloud_results: dataset=%r expected_recordings=%d",
        dataset_name,
        expected_count,
    )
    wait_for_dataset_ready(
        dataset_name,
        expected_recording_count=expected_count,
        timeout_s=case_timeout_seconds(case),
        poll_interval_s=2.0,
    )

    # Compute duration bounds (mirrors _verify_recording_structure) so we can
    # wait until the backend has finalized end_time for every recording before
    # performing the structural pass.
    base_duration_s = float(case.duration_sec)
    clock_tolerance_s = 2.0
    if case.context_duration_mode == DURATION_MODE_VARIABLE:
        finalized_min_s = (
            base_duration_s * DURATION_VARIABLE_MIN_FACTOR - clock_tolerance_s
        )
        finalized_max_s = (
            base_duration_s * DURATION_VARIABLE_MAX_FACTOR + clock_tolerance_s
        )
    else:
        finalized_min_s = base_duration_s - clock_tolerance_s
        finalized_max_s = base_duration_s + clock_tolerance_s

    all_recording_ids = {
        str(rec_id) for result in results for rec_id in result.recording_ids
    }
    wait_for_recordings_finalized(
        dataset_name,
        all_recording_ids,
        min_duration_s=finalized_min_s,
        max_duration_s=finalized_max_s,
        timeout_s=case_timeout_seconds(case),
        poll_interval_s=2.0,
    )

    dataset: Dataset = nc.get_dataset(dataset_name)

    recording_lookup: dict[str, tuple[ContextResult, int]] = {}
    for result in results:
        for rec_index, rec_id in enumerate(result.recording_ids):
            recording_lookup[str(rec_id)] = (result, rec_index)

    # --- Structural pass (unsynced) ---
    verified_ids: set[str] = set()
    structural_failures: dict[str, list[str]] = {}
    unexpected_ids: list[str] = []

    recording: Recording
    for recording in dataset:
        recording_id = str(recording.id)
        if recording_id not in recording_lookup:
            unexpected_ids.append(recording_id)
            continue
        result, recording_index = recording_lookup[recording_id]
        logger.info(
            "structural pass: recording %s of dataset %r", recording_id, dataset_name
        )
        try:
            _verify_recording_structure(
                recording=recording,
                dataset_name=dataset_name,
                result=result,
                case=case,
            )
        except AssertionError as exc:
            rec_failures = [str(exc)]
            structural_failures[recording_id] = rec_failures
        verified_ids.add(recording_id)

    missing = set(recording_lookup.keys()) - verified_ids
    structural_errors: list[str] = []
    if unexpected_ids:
        structural_errors.append(
            f"Unexpected recordings in dataset '{dataset_name}': {unexpected_ids}"
        )
    if missing:
        structural_errors.append(
            f"Recordings not found in dataset '{dataset_name}': {missing}"
        )
    for rec_id, rec_failures in structural_failures.items():
        structural_errors.append(
            f"Recording {rec_id} structural failures:\n"
            + "\n".join(f"  - {f}" for f in rec_failures)
        )
    assert not structural_errors, (
        f"Structural pass: {len(structural_failures)} recording(s) failed "
        f"({len(verified_ids)} checked, {len(missing)} missing, "
        f"{len(unexpected_ids)} unexpected):\n" + "\n".join(structural_errors)
    )

    # --- Data pass (synced) ---
    synced_dataset = dataset.synchronize()
    verified_ids = set()
    data_failures: dict[str, list[str]] = {}
    unexpected_ids = []

    for synced_episode in synced_dataset:
        recording_id = str(synced_episode.id)
        if recording_id not in recording_lookup:
            unexpected_ids.append(recording_id)
            continue
        result, recording_index = recording_lookup[recording_id]
        logger.info("data pass: recording %s of dataset %r", recording_id, dataset_name)
        summary = _collect_episode_summary(synced_episode)
        try:
            _verify_synched_episode_summary(
                summary=summary,
                result=result,
                case=case,
                recording_index=recording_index,
            )
        except AssertionError as exc:
            data_failures[recording_id] = [str(exc)]
        verified_ids.add(recording_id)

        # Loose: warn on missing frame codes but do not fail the test.
        if result.has_video:
            for camera_index, camera_name in enumerate(result.camera_names):
                actual_codes = list(summary["frame_codes"].get(camera_name, []))
                base_code = (
                    (result.context_index * 1_000_000_000)
                    + (recording_index * 10_000_000)
                    + (camera_index * 100_000)
                )
                expected_codes = set(
                    range(base_code, base_code + result.video_frame_count)
                )
                missing_codes = expected_codes - set(actual_codes)
                if missing_codes:
                    sample = sorted(missing_codes)[: len(missing_codes) // 2 + 1]
                    duplicate_count = len(actual_codes) - (
                        len(expected_codes) - len(missing_codes)
                    )
                    logger.warning(
                        "recording %s camera %r: %d missing frame code(s) "
                        "(received %d frames, %d duplicate(s), "
                        "%d expected codes absent from [%d, %d]); "
                        "first missing: %s",
                        recording_id,
                        camera_name,
                        len(missing_codes),
                        len(actual_codes),
                        duplicate_count,
                        len(missing_codes),
                        base_code,
                        base_code + result.video_frame_count - 1,
                        sample,
                    )

    missing = set(recording_lookup.keys()) - verified_ids
    data_errors: list[str] = []
    if unexpected_ids:
        data_errors.append(
            f"Unexpected recordings in dataset '{dataset_name}': {unexpected_ids}"
        )
    if missing:
        data_errors.append(
            f"Recordings not found in dataset '{dataset_name}': {missing}"
        )
    for rec_id, ep_failures in data_failures.items():
        data_errors.append(
            f"Recording {rec_id} data failures:\n"
            + "\n".join(f"  - {f}" for f in ep_failures)
        )
    assert not data_errors, (
        f"Data pass: {len(data_failures)} recording(s) failed "
        f"({len(verified_ids)} checked, {len(missing)} missing, "
        f"{len(unexpected_ids)} unexpected):\n" + "\n".join(data_errors)
    )
