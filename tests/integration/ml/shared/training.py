"""Training-job helpers for ML integration tests."""

import logging
import time
from typing import TYPE_CHECKING

from neuracore_types import DataType, Metrics

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.utils.http_session import thread_local_session

if TYPE_CHECKING:
    from neuracore.core.data.synced_dataset import SynchronizedDataset
    from neuracore.core.endpoint import Policy

logger = logging.getLogger(__name__)

MSE_THRESHOLD = 1e-4
L1_LOSS_METRIC_KEY = "train/epoch/loss/l1_loss"

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}


def cancel_incomplete_training_jobs(
    job_ids: list[str],
    completed_statuses: dict[str, str] | None = None,
) -> None:
    """Cancel training jobs that have not reached a terminal state."""
    completed_statuses = completed_statuses or {}
    for job_id in job_ids:
        if job_id in completed_statuses:
            continue
        try:
            status = nc.get_training_job_status(job_id=job_id)
            if status in TERMINAL_STATES:
                continue
            logger.warning(
                f"Cancelling incomplete training job {job_id} (status={status})"
            )
            nc.cancel_training_job(job_id)
        except Exception:
            logger.warning(f"Failed to cancel training job {job_id}", exc_info=True)


def wait_for_training(
    job_id: str,
    timeout_minutes: int = 120,
    poll_seconds: int = 20,
) -> str:
    deadline = time.time() + timeout_minutes * 60
    while True:
        status = nc.get_training_job_status(job_id=job_id)
        logger.info(f"Training job {job_id}: {status}")
        if status in TERMINAL_STATES:
            return status
        if time.time() >= deadline:
            cancel_incomplete_training_jobs([job_id])
            assert (
                False
            ), f"Training job {job_id} did not finish within {timeout_minutes} minutes"
        time.sleep(poll_seconds)


def wait_for_all_training(
    job_ids: list[str],
    timeout_minutes: int = 120,
    poll_seconds: int = 20,
) -> dict[str, str]:
    assert job_ids, "Expected at least one training job id"
    deadline = time.time() + timeout_minutes * 60
    final_statuses: dict[str, str] = {}

    while True:
        for job_id in job_ids:
            if job_id in final_statuses:
                continue
            status = nc.get_training_job_status(job_id=job_id)
            logger.info(f"Training job {job_id}: {status}")
            if status in TERMINAL_STATES:
                final_statuses[job_id] = status

        if len(final_statuses) == len(job_ids):
            return final_statuses

        if time.time() >= deadline:
            cancel_incomplete_training_jobs(job_ids, final_statuses)
            assert False, (
                f"Training job(s) {job_ids} did not finish within "
                f"{timeout_minutes} minutes"
            )
        time.sleep(poll_seconds)


def format_training_log_entries(entries: list[dict]) -> str:
    """Format VM log entries for pytest failure output."""
    formatted_entries = []
    for entry in entries:
        timestamp = entry.get("timestamp", "<missing timestamp>")
        severity = entry.get("severity", "<missing severity>")
        message = entry.get("message", "<missing message>")
        traceback = entry.get("traceback")
        formatted_entry = f"[{timestamp}] {severity}: {message}"
        if traceback:
            formatted_entry = f"{formatted_entry}\n{traceback}"
        formatted_entries.append(formatted_entry)
    return "\n\n".join(formatted_entries)


def assert_no_training_log_errors(
    job_id: str,
    context: str,
    max_entries: int = 10_000,
) -> None:
    """Fail if a training VM emitted ERROR or CRITICAL log entries."""
    offending_entries: list[dict] = []
    for severity in ("ERROR", "CRITICAL"):
        logs = nc.get_training_job_logs(
            job_id,
            max_entries=max_entries,
            severity_filter=severity,
        )
        entries = logs.get("logs", [])
        assert isinstance(entries, list), (
            f"{context}: expected '{severity}' training logs response to contain "
            f"a list under 'logs', got: {logs!r}"
        )
        offending_entries.extend(entries)

    assert not offending_entries, (
        f"{context}: training job {job_id} emitted "
        f"{len(offending_entries)} ERROR/CRITICAL log entries:\n\n"
        f"{format_training_log_entries(offending_entries)}"
    )


def get_training_job_metrics(job_id: str) -> Metrics:
    """Fetch training metrics for a completed or in-progress job."""
    org_id = get_current_org()
    session = thread_local_session()
    response = session.get(
        f"{API_URL}/org/{org_id}/training/jobs/{job_id}/metrics",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()
    return Metrics.model_validate(response.json())


def get_final_metric_value(job_id: str, metric_key: str) -> float | None:
    """Return the value at the highest logged step for *metric_key*, if present."""
    metrics = get_training_job_metrics(job_id)
    metric_data = metrics.metrics.get(metric_key)
    if metric_data is None or not metric_data.data:
        return None
    final_step = max(metric_data.data.keys())
    return float(metric_data.data[final_step])


def assert_training_loss_below(
    job_id: str,
    metric_key: str,
    threshold: float,
    *,
    context: str = "training loss",
) -> float:
    """Assert the final epoch loss metric is below *threshold*."""
    final_loss = get_final_metric_value(job_id, metric_key)
    assert (
        final_loss is not None
    ), f"{context}: metric {metric_key!r} not found for job {job_id}"
    assert (
        final_loss < threshold
    ), f"{context}: {metric_key}={final_loss:.6g} is not below {threshold:.6g}"
    return final_loss


def evaluate_training_mse(
    policy: "Policy",
    synced_dataset: "SynchronizedDataset",
    ground_truth_by_recording: dict[str, list[float]],
    target_joint_name: str,
    *,
    threshold: float = MSE_THRESHOLD,
    excluded_recording_ids: set[str] | None = None,
) -> float:
    """Evaluate per-sample MSE between predictions and stored ground truth.

    Returns the mean MSE across all evaluated frames.
    """
    excluded_recording_ids = excluded_recording_ids or set()
    total_squared_error = 0.0
    total_frames = 0

    for recording in synced_dataset:
        recording_id = str(recording.id)
        assert (
            recording_id not in excluded_recording_ids
        ), f"Deleted recording {recording_id} appeared in synced dataset"
        expected_targets = ground_truth_by_recording[recording_id]
        for frame_idx, sync_point in enumerate(recording):
            predictions = policy.predict(sync_point=sync_point, timeout=60)
            joint_targets = predictions[DataType.JOINT_TARGET_POSITIONS]
            predicted = float(joint_targets[target_joint_name].value[0, 0, 0].item())
            expected = expected_targets[frame_idx]
            squared_error = (predicted - expected) ** 2
            total_squared_error += squared_error
            total_frames += 1
            assert squared_error < threshold, (
                f"Recording {recording_id} frame {frame_idx}: "
                f"MSE={squared_error:.6g} exceeds {threshold:.6g} "
                f"(predicted={predicted:.6g}, expected={expected:.6g})"
            )

    assert total_frames > 0, "No frames evaluated"
    mean_mse = total_squared_error / total_frames
    logger.info(
        f"Evaluated {total_frames} frames across "
        f"{len(synced_dataset)} recordings; mean MSE={mean_mse:.6g}"
    )
    return mean_mse


def build_cross_embodiment_descriptions(
    dataset: Dataset,
    input_types: list[DataType],
    output_types: list[DataType],
) -> tuple[dict, dict]:
    input_desc: dict = {}
    output_desc: dict = {}
    for robot_id in dataset.robot_ids:
        embodiment_description = dataset.get_full_embodiment_description(robot_id)
        for data_type in input_types:
            assert (
                data_type in embodiment_description
            ), f"{data_type.value} missing from robot {robot_id} embodiment description"
        input_desc[robot_id] = {dt: embodiment_description[dt] for dt in input_types}
        for data_type in output_types:
            assert (
                data_type in embodiment_description
            ), f"{data_type.value} missing from robot {robot_id} embodiment description"
        output_desc[robot_id] = {dt: embodiment_description[dt] for dt in output_types}
    return input_desc, output_desc
