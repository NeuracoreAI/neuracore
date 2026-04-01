"""Online recording-and-playback matrix tests.

Each parametrized case records data, uploads it, waits for the dataset to
become ready on the platform, then downloads and verifies every frame.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from neuracore_types import DataType
from recording_playback_matrix_config import (
    MATRIX_CASES,
    RecordingPlaybackMatrixCase,
    assert_context_mode,
    build_context_specs,
    case_id,
    case_timeout_seconds,
    has_configured_org,
    run_case_contexts,
)
from recording_playback_shared import decode_frame_number, wait_for_dataset_ready

import neuracore as nc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode verification helpers
# ---------------------------------------------------------------------------


def _collect_episode_summary(synced_episode: object) -> dict[str, object]:
    """Walk a synchronised episode and collect per-type counts and values."""
    summary: dict[str, object] = {
        "sync_points": 0,
        "rgb_counts": {},
        "frame_codes": {},
        "joint_position_counts": {},
        "joint_velocity_counts": {},
        "joint_torque_counts": {},
        "joint_position_values": [],
        "custom_counts": {},
    }

    for frame_index, sync_point in enumerate(
        synced_episode
    ):  # type: ignore[call-overload]
        summary["sync_points"] = int(summary["sync_points"]) + 1

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
            for name in sync_point[DataType.CUSTOM_1D].keys():
                key = str(name)
                custom_counts[key] = custom_counts.get(key, 0) + 1
            summary["custom_counts"] = custom_counts

    return summary


def _verify_episode_summary(
    *,
    summary: dict[str, object],
    result: dict[str, object],
    case: RecordingPlaybackMatrixCase,
    recording_index: int,
) -> None:
    """Assert that an episode summary matches expected values for a case."""
    joint_names = list(result["joint_names"])
    frame_count = int(result["frame_count"])
    fps = int(result["fps"])

    for joint_name in joint_names:
        assert summary["joint_position_counts"].get(joint_name) == frame_count
        assert summary["joint_velocity_counts"].get(joint_name) == frame_count
        assert summary["joint_torque_counts"].get(joint_name) == frame_count

    for frame_index, joint_name, actual_value in summary["joint_position_values"]:
        joint_index = joint_names.index(joint_name)
        expected_value = math.sin((frame_index / fps) * (0.5 + (joint_index * 0.25)))
        assert abs(actual_value - expected_value) <= 1e-5

    expected_marker_names = list(result["marker_names"])
    for marker_name in expected_marker_names:
        assert summary["custom_counts"].get(marker_name) == frame_count

    if case.data_type == "joints_only":
        assert summary["rgb_counts"] == {}
        assert summary["frame_codes"] == {}
        return

    camera_name_list = list(result["camera_names"])
    for camera_index, camera_name in enumerate(camera_name_list):
        assert summary["rgb_counts"].get(camera_name) == frame_count
        expected_codes = [
            (int(result["context_index"]) * 1_000_000_000)
            + (recording_index * 10_000_000)
            + (camera_index * 100_000)
            + frame_index
            for frame_index in range(frame_count)
        ]
        assert summary["frame_codes"].get(camera_name) == expected_codes


def _verify_all_results(
    *,
    results: list[dict[str, object]],
    case: RecordingPlaybackMatrixCase,
) -> None:
    """Download the dataset and verify every recording matches expectations."""
    dataset_name = str(results[0]["dataset_name"])
    wait_for_dataset_ready(
        dataset_name,
        expected_recording_count=case.recording_count,
        timeout_s=case_timeout_seconds(case),
        poll_interval_s=2.0,
    )

    dataset = nc.get_dataset(dataset_name)
    synced_dataset = dataset.synchronize()

    recording_lookup: dict[str, tuple[dict[str, object], int]] = {}
    for result in results:
        for rec_index, rec_id in enumerate(list(result["recording_ids"])):
            recording_lookup[str(rec_id)] = (result, rec_index)

    verified_ids: set[str] = set()

    for synced_episode in synced_dataset:
        recording_id = str(synced_episode.id)
        assert (
            recording_id in recording_lookup
        ), f"Unexpected recording {recording_id} in dataset '{dataset_name}'"
        result, recording_index = recording_lookup[recording_id]
        summary = _collect_episode_summary(synced_episode)
        _verify_episode_summary(
            summary=summary,
            result=result,
            case=case,
            recording_index=recording_index,
        )
        verified_ids.add(recording_id)

    missing = set(recording_lookup.keys()) - verified_ids
    assert not missing, f"Recordings not found in dataset '{dataset_name}': {missing}"


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", MATRIX_CASES, ids=case_id)
@pytest.mark.parametrize("wait_for_upload", [True, False], ids=["wait", "nowait"])
def test_recording_and_playback_matrix_online(
    case: RecordingPlaybackMatrixCase,
    wait_for_upload: bool,
    dataset_cleanup,
    reset_matrix_timer_stats,
    log_run_analysis_on_teardown,
) -> None:
    """Record, upload, and verify playback for a matrix case."""
    nc.login()
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )
    specs = build_context_specs(case, wait=wait_for_upload)
    dataset_cleanup(str(specs[0]["dataset_name"]))
    results = run_case_contexts(case, wait=wait_for_upload, specs=specs)
    log_run_analysis_on_teardown(case, results)
    assert_context_mode(case, results)
    nc.login()
    _verify_all_results(results=results, case=case)
