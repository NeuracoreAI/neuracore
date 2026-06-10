"""Integration test: per-recording trace validation before training run.

Each parametrized case collects a 3-episode dataset whose episodes log a
shrinking (or growing) subset of traces — joint positions joint_1..joint_3
or rgb cameras cam_1..cam_3 — then requests training on the dataset's
embodiment description and expects a ValueError naming the incomplete
recordings, independent of recording order.
"""

import logging
import time

import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.api.training import get_training_jobs
from neuracore.core.data.dataset import Dataset
from tests.integration.ml.shared.utils import unique_name
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint_1", "joint_2", "joint_3"]
CAMERA_NAMES = ["cam_1", "cam_2", "cam_3"]
TRACE_NAMES = {
    DataType.JOINT_POSITIONS: JOINT_NAMES,
    DataType.RGB_IMAGES: CAMERA_NAMES,
}

ROBOT_NAME = "trace_validation_test_robot"
ROBOT_INSTANCE = 0
FREQUENCY = 20
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
EPISODE_STEPS = 5
RECORDING_STOP_TIMEOUT_SECONDS = 300
RECORDING_POLL_SECONDS = 5

RGB_FRAME = np.full((84, 84, 3), 128, dtype=np.uint8)
JOINT_TARGET_SAMPLE = {name: 0.2 for name in JOINT_NAMES}

CNNMLP_CONFIG = {
    "batch_size": 16,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


def _record_episode(data_type: DataType, trace_names: list[str]) -> None:
    """Record one episode logging only the given traces of one data type."""
    t = time.time()
    nc.start_recording(robot_name=ROBOT_NAME, instance=ROBOT_INSTANCE)
    for _ in range(EPISODE_STEPS):
        t += 1.0 / FREQUENCY
        nc.log_joint_target_positions(
            target_positions=JOINT_TARGET_SAMPLE,
            robot_name=ROBOT_NAME,
            instance=ROBOT_INSTANCE,
            timestamp=t,
        )
        if data_type is DataType.JOINT_POSITIONS:
            nc.log_joint_positions(
                positions={name: 0.1 for name in trace_names},
                robot_name=ROBOT_NAME,
                instance=ROBOT_INSTANCE,
                timestamp=t,
            )
        elif data_type is DataType.RGB_IMAGES:
            for camera_name in trace_names:
                nc.log_rgb(
                    name=camera_name,
                    rgb=RGB_FRAME,
                    robot_name=ROBOT_NAME,
                    instance=ROBOT_INSTANCE,
                    timestamp=t,
                )
        else:
            raise NotImplementedError(f"Unsupported data type {data_type}")
    nc.stop_recording(wait=False, robot_name=ROBOT_NAME, instance=ROBOT_INSTANCE)


def _collect_dataset(
    dataset_name: str, data_type: DataType, episode_traces: list[list[str]]
) -> Dataset:
    """Collect one dataset with the given per-episode trace subsets."""
    with online_daemon_running():
        assert_exactly_one_daemon_pid()
        nc.connect_robot(
            robot_name=ROBOT_NAME,
            instance=ROBOT_INSTANCE,
            overwrite=False,
        )
        nc.create_dataset(name=dataset_name)
        for ep_idx, trace_names in enumerate(episode_traces):
            logger.info(
                f"Recording episode {ep_idx + 1}/{len(episode_traces)}: "
                f"{trace_names}"
            )
            _record_episode(data_type, trace_names)
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=ep_idx + 1,
                timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
                poll_interval_s=RECORDING_POLL_SECONDS,
            )
            logger.info(f"Episode {ep_idx + 1} ready.")
    dataset = nc.get_dataset(name=dataset_name)
    assert len(dataset) == len(episode_traces), (
        f"Expected {len(episode_traces)} recordings in {dataset_name!r}, "
        f"got {len(dataset)}"
    )
    logger.info(
        f"Dataset {dataset_name!r}: id={dataset.id} robot_ids={dataset.robot_ids} "
        f"recordings={len(dataset)}"
    )
    for recording in dataset:
        logger.info(
            f"Recording {recording.id}: "
            f"data_types={sorted(dt.value for dt in recording.data_types)} "
            f"robot_id={recording.robot_id} instance={recording.instance} "
            f"total_bytes={recording.total_bytes} "
            f"duration_s={recording.end_time - recording.start_time:.2f}"
        )
    return dataset


def _delete_training_job_by_name(job_name: str) -> None:
    """Delete the training job with the given name, if one exists."""
    for job in get_training_jobs():
        if job.get("name") == job_name and job.get("id"):
            logger.warning(
                f"Unexpected training job: id={job.get('id')} "
                f"name={job.get('name')!r} status={job.get('status')} "
                f"dataset_id={job.get('dataset_id')}"
            )
            try:
                nc.delete_training_job(job["id"])
                logger.warning(f"Deleted training job {job_name!r}")
            except Exception:
                logger.warning(
                    f"Failed to delete training job {job_name!r}", exc_info=True
                )


@pytest.fixture(scope="module", autouse=True)
def _login() -> None:
    nc.login()


@pytest.mark.parametrize(
    "data_type",
    [DataType.JOINT_POSITIONS, DataType.RGB_IMAGES],
    ids=["joint_positions", "rgb"],
)
@pytest.mark.parametrize("order", ["decreasing", "increasing"])
def test_training_rejects_recordings_with_missing_traces(
    data_type: DataType, order: str
) -> None:
    """Training on all traces must fail when some recordings lack traces.

    The outcome must not depend on the order the episodes were recorded in.
    """
    trace_names = TRACE_NAMES[data_type]
    # Suffix subsets: [t1,t2,t3], [t2,t3], [t3] — reversed for increasing.
    episode_traces = [trace_names[i:] for i in range(len(trace_names))]
    if order == "increasing":
        episode_traces.reverse()

    dataset_name = unique_name(prefix=f"trace_validation_{data_type.value}_{order}")
    dataset = _collect_dataset(dataset_name, data_type, episode_traces)

    robot_id = dataset.robot_ids[0]
    description = dataset.get_full_embodiment_description(robot_id)
    logger.info(f"[{data_type.value}/{order}] Embodiment description: {description}")
    input_desc = {robot_id: {data_type: description[data_type]}}
    output_desc = {
        robot_id: {
            DataType.JOINT_TARGET_POSITIONS: description[
                DataType.JOINT_TARGET_POSITIONS
            ]
        }
    }
    logger.info(f"[{data_type.value}/{order}] Input description: {input_desc}")
    logger.info(f"[{data_type.value}/{order}] Output description: {output_desc}")

    # TODO: update the expected behaviour once per-recording trace validation
    # lands (assert the error message format and which recordings it names).
    job_name = unique_name(prefix="trace_validation_job")
    logger.info(
        f"[{data_type.value}/{order}] Starting training attempt: "
        f"job_name={job_name!r} dataset={dataset_name!r} algorithm=CNNMLP "
        f"frequency={FREQUENCY}"
    )
    try:
        with pytest.raises(ValueError) as exc_info:
            nc.start_training_run(
                name=job_name,
                dataset_name=dataset_name,
                algorithm_name="CNNMLP",
                algorithm_config=CNNMLP_CONFIG,
                gpu_type=GPU_TYPE,
                num_gpus=NUM_GPUS,
                frequency=FREQUENCY,
                input_cross_embodiment_description=input_desc,
                output_cross_embodiment_description=output_desc,
            )
    except BaseException:
        # A job may have been submitted; delete it before failing.
        _delete_training_job_by_name(job_name)
        raise

    logger.info(
        f"[{data_type.value}/{order}] Training rejected with error:\n"
        f"{exc_info.value}"
    )

    job_names = {job.get("name") for job in get_training_jobs()}
    assert (
        job_name not in job_names
    ), f"Training job {job_name!r} was created despite failing validation"

    # Reached only on success; failures keep the dataset for debugging.
    dataset.delete()
