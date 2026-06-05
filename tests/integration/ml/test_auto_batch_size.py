"""Integration tests for automatic batch-size selection during training.

This test submits training jobs with batch_size set to auto for representative
algorithm, GPU type, and GPU count combinations. It verifies that each job
reaches COMPLETED.
The test is split into two phases:

  test_start_training submits a training job and records the job ID.
  test_wait_for_training_terminal_status waits for that job and verifies that
  it completes successfully.

Case definitions live in auto_batch_size_cases.yaml.
"""

from __future__ import annotations

import atexit
import logging
import os
import pprint
import time
from pathlib import Path

import pytest
import yaml
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.data.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "Conq Hose Manipulation"
# IT-ABS = Integration Test Auto Batch Size
TRAINING_NAME = "IT-ABS"
DEFAULT_FREQUENCY = 10
TRAINING_TIMEOUT_MINUTES = 60
TRAINING_POLL_SECONDS = 60
ALGORITHM_CONFIGS_FILE = Path(__file__).with_name("algorithm_configs.yaml")
AUTO_BATCH_SIZE_CASES_FILE = Path(__file__).with_name("auto_batch_size_cases.yaml")
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}

# Key by test case, not algorithm. ACT runs on multiple GPUs, so keying by
# algorithm would let one ACT job overwrite another during local full-file runs.
TRAINING_JOB_IDS: dict[str, str] = {}


def load_auto_batch_size_cases() -> list[dict]:
    """Load the algorithm, GPU type, and GPU count combinations to test."""
    with open(AUTO_BATCH_SIZE_CASES_FILE) as f:
        return yaml.safe_load(f)["cases"]


def build_auto_batch_size_test_case(
    algorithm_name: str,
    gpu_type: str,
    num_gpus: int,
) -> str:
    """Build the pytest case id for one configured training setup."""
    return f"{algorithm_name}_{num_gpus}_{gpu_type}"


def build_training_name(
    algorithm_name: str,
    num_gpus: int,
    gpu_type: str,
    timestamp: str,
) -> str:
    """Build a readable name that stays safe for current backend VM naming."""
    short_gpu_type = gpu_type.removeprefix("NVIDIA_TESLA_").removeprefix("NVIDIA_")
    return (
        f"{TRAINING_NAME} - "
        f"{algorithm_name}-{num_gpus}-gpus-{short_gpu_type}-{timestamp}"
    )


AUTO_BATCH_SIZE_CASES = tuple(
    pytest.param(
        auto_batch_size_case["algorithm"],
        auto_batch_size_case["gpu_type"],
        auto_batch_size_case["num_gpus"],
        id=build_auto_batch_size_test_case(
            algorithm_name=auto_batch_size_case["algorithm"],
            gpu_type=auto_batch_size_case["gpu_type"],
            num_gpus=auto_batch_size_case["num_gpus"],
        ),
    )
    for auto_batch_size_case in load_auto_batch_size_cases()
)


def get_algorithm_config_entry(algorithm_name: str) -> dict:
    """Return the algorithm config used as the base for the training request."""
    with open(ALGORITHM_CONFIGS_FILE) as f:
        algorithm_configs = yaml.safe_load(f)["algorithms"]
    return next(
        algorithm_config
        for algorithm_config in algorithm_configs
        if algorithm_config["name"] == algorithm_name
    )


def build_cross_embodiment_descriptions(
    dataset: Dataset,
) -> tuple[dict, dict]:
    """Build the dataset embodiment descriptions required by training."""
    robot_ids = dataset.robot_ids
    assert len(robot_ids) == 1, f"Expected one robot, got {robot_ids}"
    robot_id = robot_ids[0]

    full_embodiment_description = dataset.get_full_embodiment_description(robot_id)
    input_cross_embodiment_description = {
        robot_id: {
            DataType.RGB_IMAGES: full_embodiment_description[DataType.RGB_IMAGES],
            DataType.JOINT_POSITIONS: full_embodiment_description[
                DataType.JOINT_POSITIONS
            ],
        }
    }
    output_cross_embodiment_description = {
        robot_id: {
            DataType.JOINT_TARGET_POSITIONS: full_embodiment_description[
                DataType.JOINT_TARGET_POSITIONS
            ]
        }
    }
    return input_cross_embodiment_description, output_cross_embodiment_description


def wait_for_terminal_status(test_case: str, training_job_id: str) -> str:
    """Wait for a training job to reach a terminal status."""
    training_wait_start = time.time()
    training_job_status = nc.get_training_job_status(training_job_id)

    while training_job_status not in TERMINAL_STATES:
        elapsed_minutes = (time.time() - training_wait_start) / 60
        if elapsed_minutes > TRAINING_TIMEOUT_MINUTES:
            pytest.fail(
                f"[{test_case}] Training job {training_job_id} did not reach "
                f"a terminal state after {TRAINING_TIMEOUT_MINUTES} minutes."
            )

        logger.info(
            f"[{test_case}] Waiting for training job {training_job_id}: "
            f"status={training_job_status}"
        )
        time.sleep(TRAINING_POLL_SECONDS)
        training_job_status = nc.get_training_job_status(training_job_id)

    logger.info(
        f"[{test_case}] Training job {training_job_id} finished with "
        f"status={training_job_status}"
    )
    return training_job_status


def cleanup_training_job(
    test_case: str,
    training_job_id: str | None = None,
) -> None:
    """Delete a training job created by this test."""
    training_job_id = training_job_id or TRAINING_JOB_IDS.get(test_case)
    if not training_job_id:
        return

    # Drop the local reference before the API call so a repeated cleanup path
    # does not retry the same job after another cleanup handler already ran.
    TRAINING_JOB_IDS.pop(test_case, None)
    try:
        nc.delete_training_job(training_job_id)
        logger.info(f"[{test_case}] Deleted training job {training_job_id}")
    except Exception as exc:  # pragma: no cover - cleanup best effort
        logger.warning(
            f"[{test_case}] Failed to delete training job {training_job_id}: " f"{exc}",
            exc_info=True,
        )


def cleanup_training_jobs() -> None:
    """Delete all training jobs still tracked by this test process."""
    for test_case, training_job_id in list(TRAINING_JOB_IDS.items()):
        cleanup_training_job(test_case, training_job_id)


atexit.register(cleanup_training_jobs)


def build_algorithm_config(algorithm_config_entry: dict) -> dict:
    """Only override the fields this test is responsible for."""
    algorithm_config = {
        "batch_size": "auto",
        "epochs": 1,
    }
    output_prediction_horizon = algorithm_config_entry["algorithm_config"].get(
        "output_prediction_horizon"
    )
    if output_prediction_horizon is not None:
        algorithm_config["output_prediction_horizon"] = output_prediction_horizon
    return algorithm_config


@pytest.mark.parametrize(
    ("algorithm_name", "gpu_type", "num_gpus"),
    AUTO_BATCH_SIZE_CASES,
)
class TestAutoBatchSize:
    def test_start_training(
        self,
        algorithm_name: str,
        gpu_type: str,
        num_gpus: int,
    ) -> None:
        """Phase 1: submit an auto batch-size training job and record its ID."""
        test_case = build_auto_batch_size_test_case(
            algorithm_name=algorithm_name,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
        )
        algorithm_config_entry = get_algorithm_config_entry(algorithm_name)
        nc.login()

        logger.info(
            f"[{test_case}] Starting auto batch-size training on "
            f"{num_gpus} {gpu_type} GPU(s)"
        )

        logger.info(f"[{test_case}] Fetching dataset {DATASET_NAME}")
        dataset = nc.get_dataset(DATASET_NAME)
        input_cross_embodiment_description, output_cross_embodiment_description = (
            build_cross_embodiment_descriptions(dataset)
        )

        timestamp = time.strftime("%y%m%d%H%M%S")
        logger.info(f"[{test_case}] Launching training job")
        algorithm_config = build_algorithm_config(algorithm_config_entry)
        logger.info(
            f"[{test_case}] Training config overrides: "
            f"{pprint.pformat(algorithm_config)}"
        )
        job_data = nc.start_training_run(
            name=build_training_name(
                algorithm_name=algorithm_name,
                num_gpus=num_gpus,
                gpu_type=gpu_type,
                timestamp=timestamp,
            ),
            dataset_name=DATASET_NAME,
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            frequency=DEFAULT_FREQUENCY,
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            name_auto_increment=True,
        )
        training_job_id = job_data["id"]
        TRAINING_JOB_IDS[test_case] = training_job_id
        logger.info(f"[{test_case}] Training job started: {training_job_id}")

        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as handle:
                handle.write(f"training_job_id={training_job_id}\n")
            # CI waits in a fresh pytest process with TRAINING_JOB_ID set from
            # this output. Removing the local cleanup reference keeps the start
            # process from deleting the job before the wait step can observe it.
            TRAINING_JOB_IDS.pop(test_case, None)

    def test_wait_for_training_terminal_status(
        self,
        algorithm_name: str,
        gpu_type: str,
        num_gpus: int,
    ) -> None:
        """Phase 2: verify that the submitted training job reaches COMPLETED."""
        test_case = build_auto_batch_size_test_case(
            algorithm_name=algorithm_name,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
        )
        nc.login()

        # CI supplies TRAINING_JOB_ID from the start step. Local full-file runs
        # use the in-process map populated by test_start_training.
        training_job_id = os.environ.get("TRAINING_JOB_ID")
        if not training_job_id:
            training_job_id = TRAINING_JOB_IDS.get(test_case)
        if not training_job_id:
            pytest.skip(
                "TRAINING_JOB_ID not set - run test_start_training first, then "
                "re-run with TRAINING_JOB_ID=<id>"
            )

        TRAINING_JOB_IDS[test_case] = training_job_id
        try:
            final_status = wait_for_terminal_status(
                test_case=test_case,
                training_job_id=training_job_id,
            )
            job_data = nc.get_training_job_data(training_job_id)

            if final_status != "COMPLETED":
                error_logs = None
                try:
                    error_logs = nc.get_training_job_logs(
                        training_job_id,
                        max_entries=200,
                        severity_filter="ERROR",
                    )
                except Exception as exc:  # pragma: no cover - best-effort logging
                    logger.warning(
                        f"[{test_case}] Failed to fetch training logs for "
                        f"{training_job_id}: {exc}"
                    )

                raise AssertionError(
                    f"[{test_case}] Training job {training_job_id} did not "
                    f"complete successfully.\n"
                    f"status={final_status}\n"
                    f"job_data={pprint.pformat(job_data)}\n"
                    f"error_logs={pprint.pformat(error_logs)}"
                )

            logger.info(
                f"[{test_case}] Training job {training_job_id} completed "
                f"successfully"
            )
        finally:
            cleanup_training_job(test_case, training_job_id)


def teardown_module(module: object) -> None:
    """Delete training jobs left behind by this test module."""
    cleanup_training_jobs()
