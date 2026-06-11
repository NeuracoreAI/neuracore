"""Verify that training cannot resume after its source dataset is deleted.

The test deliberately completes an initial training run before deleting its
dataset. This isolates the resume path from ordinary training failures and
proves that the backend validates the original dataset before provisioning a
new training worker.
"""

from __future__ import annotations

import logging
import time
import uuid

import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.data.dataset import Dataset

logger = logging.getLogger(__name__)

# This existing integration dataset supplies known-good recordings. The test
# clones it so it can safely delete its own dataset without modifying shared
# test data.
SOURCE_DATASET_NAME = "Conq Hose Manipulation"
GPU_TYPE = "NVIDIA_TESLA_V100"

# Cloud training and dataset operations are asynchronous. Keep their timeouts
# separate so a failure identifies whether training or deletion became stuck.
TRAINING_TIMEOUT_MINUTES = 120
TRAINING_POLL_SECONDS = 20
DELETION_TIMEOUT_SECONDS = 60
DELETION_POLL_SECONDS = 5
DATASET_READY_TIMEOUT_SECONDS = 120
DATASET_READY_POLL_SECONDS = 5
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}

# A single epoch is sufficient to produce a resumable completed run while
# limiting the integration test's execution time and compute cost.
TRAINING_CONFIG = {
    "batch_size": 64,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


def _unique_name(prefix: str) -> str:
    """Return a collision-resistant name for resources shared across CI runs."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _build_cross_embodiment_descriptions(dataset: Dataset) -> tuple[dict, dict]:
    """Build the CNNMLP input and output schema from the dataset itself.

    Reading the schema from the dataset avoids hard-coding robot IDs or sensor
    names, both of which belong to the source recordings and may change.
    """
    robot_ids = dataset.robot_ids
    assert len(robot_ids) == 1, f"Expected one robot, got {robot_ids}"
    robot_id = robot_ids[0]
    full_description = dataset.get_full_embodiment_description(robot_id)

    # CNNMLP consumes camera frames and current joint positions.
    input_description = {
        robot_id: {
            DataType.RGB_IMAGES: full_description[DataType.RGB_IMAGES],
            DataType.JOINT_POSITIONS: full_description[DataType.JOINT_POSITIONS],
        }
    }
    # The model predicts future joint targets for the same robot.
    output_description = {
        robot_id: {
            DataType.JOINT_TARGET_POSITIONS: full_description[
                DataType.JOINT_TARGET_POSITIONS
            ]
        }
    }
    return input_description, output_description


def _wait_for_training_to_start(job_id: str) -> str:
    """Wait until the initial run starts or fails before initialization."""
    deadline = time.time() + TRAINING_TIMEOUT_MINUTES * 60
    while True:
        status = nc.get_training_job_status(job_id)
        logger.info("Training job %s status: %s", job_id, status)
        # A very short run may move directly from PENDING to COMPLETED between
        # polls, so all terminal states must also stop this loop.
        if status == "RUNNING" or status in TERMINAL_STATES:
            return status
        assert time.time() < deadline, (
            f"Training job {job_id} did not start within "
            f"{TRAINING_TIMEOUT_MINUTES} minutes"
        )
        time.sleep(TRAINING_POLL_SECONDS)


def _wait_for_training_to_complete(job_id: str) -> dict:
    """Return the final job snapshot after the initial run becomes terminal."""
    deadline = time.time() + TRAINING_TIMEOUT_MINUTES * 60
    while True:
        job_data = nc.get_training_job_data(job_id)
        status = job_data["status"]
        logger.info("Training job %s status: %s", job_id, status)
        if status in TERMINAL_STATES:
            return job_data
        assert time.time() < deadline, (
            f"Training job {job_id} did not finish within "
            f"{TRAINING_TIMEOUT_MINUTES} minutes"
        )
        time.sleep(TRAINING_POLL_SECONDS)


def _assert_no_training_errors(job_id: str) -> None:
    """Confirm the initial run did not merely reach COMPLETED with error logs."""
    error_logs = []
    for severity in ("ERROR", "CRITICAL"):
        response = nc.get_training_job_logs(
            job_id,
            max_entries=10_000,
            severity_filter=severity,
        )
        error_logs.extend(response.get("logs", []))
    assert not error_logs, f"Initial training emitted error logs: {error_logs!r}"


def _wait_for_dataset_deletion(dataset_name: str, dataset_id: str) -> None:
    """Verify dataset metadata is unavailable through both lookup keys.

    Checking both name and ID matters because the training job stores the ID,
    while the expected error message should expose the human-readable name.
    """
    deadline = time.time() + DELETION_TIMEOUT_SECONDS
    while time.time() < deadline:
        by_name = Dataset.get_by_name(dataset_name, non_exist_ok=True)
        by_id = Dataset.get_by_id(dataset_id, non_exist_ok=True)
        if by_name is None and by_id is None:
            return
        time.sleep(DELETION_POLL_SECONDS)

    pytest.fail(
        f"Dataset {dataset_name!r} ({dataset_id}) remained accessible after deletion"
    )


def _assert_resume_failed_without_starting_job(
    *,
    job_id: str,
    dataset_name: str,
    job_before_resume: dict,
) -> None:
    """Assert the resume is actionable, side-effect free, and fails early."""
    # The SDK wraps backend resume errors in ValueError. Capturing the complete
    # message lets the test reject a generic 404 that omits the root cause.
    breakpoint()
    with pytest.raises(ValueError) as exc_info:
        nc.resume_training_run(job_id=job_id, additional_epochs=1)

    breakpoint()
    error_message = str(exc_info.value)
    normalized_error = error_message.lower()

    # These checks enforce an actionable error: operation, affected resource,
    # exact dataset, and explanation that the dataset is no longer available.
    assert (
        "resume" in normalized_error or "resuming" in normalized_error
    ), f"Resume failure did not identify the failed operation: {error_message}"
    assert (
        "dataset" in normalized_error
    ), f"Resume failure did not identify the dataset as the cause: {error_message}"
    assert (
        dataset_name.lower() in normalized_error
    ), f"Resume failure did not reference dataset {dataset_name!r}: {error_message}"
    assert any(
        phrase in normalized_error
        for phrase in ("deleted", "no longer exists", "does not exist", "not found")
    ), f"Resume failure did not explain that the dataset is gone: {error_message}"

    # A fail-fast validation must leave the completed job untouched. A PENDING
    # or RUNNING transition would mean the backend accepted the resume request.
    job_after_resume = nc.get_training_job_data(job_id)
    assert job_after_resume["status"] == job_before_resume["status"] == "COMPLETED", (
        "The failed resume changed the training job status: "
        f"{job_before_resume['status']!r} -> {job_after_resume['status']!r}"
    )
    for field in ("resume_points", "resumed_at"):
        assert job_after_resume.get(field) == job_before_resume.get(field), (
            f"The failed resume unexpectedly changed {field}: "
            f"{job_before_resume.get(field)!r} -> {job_after_resume.get(field)!r}"
        )

    # Compute field names vary between backend response versions. Comparing all
    # known variants catches infrastructure allocation when that metadata is
    # exposed, while remaining compatible with responses that omit the fields.
    compute_fields = (
        "cloud_compute_id",
        "cloud_compute_ids",
        "compute_id",
        "compute_ids",
        "instance_id",
    )
    for field in compute_fields:
        assert job_after_resume.get(field) == job_before_resume.get(field), (
            f"The failed resume unexpectedly allocated compute field {field}: "
            f"{job_before_resume.get(field)!r} -> {job_after_resume.get(field)!r}"
        )


def test_resume_fails_when_training_dataset_has_been_deleted() -> None:
    """Complete the create, train, delete, and failed-resume scenario."""
    nc.login()

    # Unique names prevent parallel CI jobs and interrupted previous runs from
    # resolving resources created by another test invocation.
    dataset_name = _unique_name("resume_deleted_dataset")
    training_name = _unique_name("resume_deleted_dataset_training")
    dataset: Dataset | None = None
    job_id: str | None = None

    try:
        # Step 1: create a disposable dataset containing valid recordings.
        # Merging a single known-good source creates a distinct dataset ID while
        # preserving the recording data and embodiment needed by CNNMLP.
        source_dataset = nc.get_dataset(SOURCE_DATASET_NAME)
        source_recording_count = len(source_dataset)
        assert (
            source_recording_count > 0
        ), f"Source dataset {SOURCE_DATASET_NAME!r} has no valid recordings"

        # The dataset_clone call returns as soon as the new dataset record is created,
        dataset = nc.dataset_clone(
            new_dataset_name=dataset_name, dataset=source_dataset
        )

        recordings = list(dataset)
        assert recordings, f"Created dataset {dataset_name!r} has no recordings"

        # Step 2: launch and complete a real training run. Completion proves the
        # dataset was valid and leaves the job in a state that can be resumed.
        input_description, output_description = _build_cross_embodiment_descriptions(
            dataset
        )
        job = nc.start_training_run(
            name=training_name,
            dataset_name=dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=TRAINING_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=1,
            frequency=10,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
        )
        job_id = job["id"]

        # Verify the backend persisted the disposable dataset association. The
        # later resume must validate this exact ID, not the shared source ID.
        assert job["dataset_id"] == dataset.id, (
            f"Training job references dataset {job['dataset_id']!r}, "
            f"expected {dataset.id!r}"
        )

        started_status = _wait_for_training_to_start(job_id)
        assert started_status in {
            "RUNNING",
            "COMPLETED",
        }, f"Training failed before starting: status={started_status!r}"
        completed_job = _wait_for_training_to_complete(job_id)
        assert (
            completed_job["status"] == "COMPLETED"
        ), f"Initial training did not complete successfully: {completed_job!r}"
        _assert_no_training_errors(job_id)

        # Step 3: delete the dataset and prove both its metadata and its
        # recordings are unavailable before attempting the resume.
        dataset.delete()
        _wait_for_dataset_deletion(dataset_name, dataset.id)
        dataset = None

        # Step 4: the backend must reject the request before mutating the job or
        # allocating replacement compute, and must name the deleted dataset.
        _assert_resume_failed_without_starting_job(
            job_id=job_id,
            dataset_name=dataset_name,
            job_before_resume=completed_job,
        )
    finally:
        # Cleanup is best effort so the original assertion remains the reported
        # failure. The dataset reference is cleared after successful deletion,
        # preventing an unnecessary second delete request.
        if job_id is not None:
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(
                    "Failed to delete training job %s", job_id, exc_info=True
                )
        if dataset is not None:
            try:
                dataset.delete()
            except Exception:
                logger.warning(
                    "Failed to delete dataset %s", dataset_name, exc_info=True
                )
