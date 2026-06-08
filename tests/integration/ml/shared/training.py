"""Training-job helpers for ML integration tests."""

import logging
import time

from neuracore_types import DataType

import neuracore as nc
from neuracore.core.data.dataset import Dataset

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}


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
        assert (
            time.time() < deadline
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

        assert (
            time.time() < deadline
        ), f"Training job(s) {job_ids} did not finish within {timeout_minutes} minutes"
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
