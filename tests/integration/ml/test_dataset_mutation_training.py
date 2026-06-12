"""Integration test: dataset mutation, statistics, datatype handling, training.

Validates synthetic multimodal datasets where the label is:

    target = joint_positions[0] + velocity[0] + mean(camera)

encoded as the first joint's JOINT_TARGET_POSITIONS. Covers initial collection,
statistics verification, baseline overfitting, dataset mutation (add 20 / delete 20),
retraining, and corrupted-camera ablation proving all modalities are used.
"""

import logging
import os
import sys
import time

import numpy as np
from neuracore_types import DataType, EmbodimentDescription

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from tests.integration.ml.shared.dataset import (
    assert_active_recordings,
    assert_dataset_metadata,
    assert_synced_statistics,
    delete_recording_from_dataset,
    wait_for_dataset_recording_count,
)
from tests.integration.ml.shared.training import (
    L1_LOSS_METRIC_KEY,
    assert_no_training_log_errors,
    assert_training_loss_below,
    build_cross_embodiment_descriptions,
    evaluate_training_mse,
    get_final_metric_value,
    wait_for_training,
)
from tests.integration.ml.shared.utils import unique_name
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.base_env import BimanualViperXTask
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ROBOT_NAME = "integration_test_robot"
ROBOT_INSTANCE = 5
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1

NC_CAM_NAME = "rgb_angle"
JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
)
TARGET_JOINT = JOINT_NAMES[0]

FREQUENCY = 20
EPISODE_STEPS = 8
RECORDING_STOP_TIMEOUT_SECONDS = 300
RECORDING_POLL_SECONDS = 5

INITIAL_RECORDINGS = 30
MUTATION_ADD = 20
MUTATION_DELETE = 20
CORRUPT_RECORDINGS = 30

INPUT_DATA_TYPES = [
    DataType.JOINT_POSITIONS,
    DataType.JOINT_VELOCITIES,
    DataType.RGB_IMAGES,
]
OUTPUT_DATA_TYPES = [DataType.JOINT_TARGET_POSITIONS]

EXPECTED_RECORDING_TYPES = {
    DataType.JOINT_POSITIONS,
    DataType.JOINT_VELOCITIES,
    DataType.RGB_IMAGES,
    DataType.JOINT_TARGET_POSITIONS,
}
EXPECTED_COMMON_TYPES = EXPECTED_RECORDING_TYPES

OVERFIT_CNNMLP_CONFIG = {
    "batch_size": 8,
    "epochs": 75,
    "output_prediction_horizon": 1,
}

L1_LOSS_THRESHOLD = 0.05
CORRUPT_MSE_MIN = 0.01
CORRUPT_MSE_FACTOR = 10


def _compute_target(joint_pos_0: float, velocity_0: float, camera_mean: float) -> float:
    """Scalar label: first joint position + first velocity + mean(camera)."""
    return joint_pos_0 + velocity_0 + camera_mean


def _record_episode(
    episode_idx: int,
    *,
    robot_name: str,
    instance_id: int,
    corrupt_camera: bool = False,
) -> list[float]:
    """Record one episode and return per-frame ground-truth targets."""
    frame_targets: list[float] = []
    t = time.time()
    rng = np.random.default_rng(episode_idx)
    nc.start_recording(robot_name=robot_name, instance=instance_id)

    for frame_idx in range(EPISODE_STEPS):
        t += 1.0 / FREQUENCY
        joint_positions = {
            name: float(value)
            for name, value in zip(
                JOINT_NAMES, rng.uniform(0.0, 1.0, size=len(JOINT_NAMES))
            )
        }
        joint_velocities = {
            name: float(value)
            for name, value in zip(
                JOINT_NAMES, rng.uniform(0.0, 1.0, size=len(JOINT_NAMES))
            )
        }
        camera_pixel = int(rng.integers(0, 256))
        target = _compute_target(
            joint_positions[TARGET_JOINT],
            joint_velocities[TARGET_JOINT],
            float(camera_pixel),
        )
        frame_targets.append(target)

        if corrupt_camera:
            rgb = np.zeros((84, 84, 3), dtype=np.uint8)
        else:
            rgb = np.full((84, 84, 3), camera_pixel, dtype=np.uint8)

        nc.log_joint_positions(
            positions=joint_positions,
            timestamp=t,
            robot_name=robot_name,
            instance=instance_id,
        )
        nc.log_joint_velocities(
            velocities=joint_velocities,
            timestamp=t,
            robot_name=robot_name,
            instance=instance_id,
        )
        nc.log_joint_target_positions(
            target_positions={TARGET_JOINT: target},
            timestamp=t,
            robot_name=robot_name,
            instance=instance_id,
        )
        nc.log_rgb(
            name=NC_CAM_NAME,
            rgb=rgb,
            timestamp=t,
            robot_name=robot_name,
            instance=instance_id,
        )

    nc.stop_recording(wait=True, robot_name=robot_name, instance=instance_id)
    return frame_targets


def _collect_recordings(
    dataset_name: str,
    robot_name: str,
    instance_id: int,
    num_episodes: int,
    *,
    start_episode_idx: int = 0,
    corrupt_camera: bool = False,
    ground_truth: dict[str, list[float]] | None = None,
    known_recording_ids: set[str] | None = None,
) -> tuple[Dataset, dict[str, list[float]], set[str]]:
    """Collect episodes into *dataset_name* and track ground-truth targets."""
    ground_truth = ground_truth if ground_truth is not None else {}
    known_recording_ids = (
        known_recording_ids if known_recording_ids is not None else set()
    )
    new_recording_ids: set[str] = set()

    with online_daemon_running():
        assert_exactly_one_daemon_pid()
        nc.connect_robot(
            robot_name=robot_name,
            instance=instance_id,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
        nc.create_dataset(name=dataset_name)
        expected_count = len(known_recording_ids)

        for offset in range(num_episodes):
            episode_idx = start_episode_idx + offset
            logger.info(
                f"Recording episode {offset + 1}/{num_episodes} "
                f"(episode_idx={episode_idx})"
            )
            frame_targets = _record_episode(
                episode_idx,
                robot_name=robot_name,
                instance_id=instance_id,
                corrupt_camera=corrupt_camera,
            )
            expected_count += 1
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=expected_count,
                timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
                poll_interval_s=RECORDING_POLL_SECONDS,
            )
            current_ids = {str(r.id) for r in nc.get_dataset(name=dataset_name)}
            added = current_ids - known_recording_ids - new_recording_ids
            assert len(added) == 1, (
                f"Expected exactly one new recording after episode "
                f"{offset + 1}, got {sorted(added)}"
            )
            recording_id = added.pop()
            new_recording_ids.add(recording_id)
            ground_truth[recording_id] = frame_targets

    dataset = nc.get_dataset(name=dataset_name)
    return dataset, ground_truth, new_recording_ids


def _single_joint_output_desc(_robot_id: str) -> EmbodimentDescription:
    return {DataType.JOINT_TARGET_POSITIONS: {0: TARGET_JOINT}}


def _build_training_descriptions(dataset: Dataset) -> tuple[dict, dict]:
    assert (
        len(dataset.robot_ids) == 1
    ), f"Expected single-robot dataset, got {len(dataset.robot_ids)} robots"
    input_desc, _ = build_cross_embodiment_descriptions(
        dataset=dataset,
        input_types=INPUT_DATA_TYPES,
        output_types=OUTPUT_DATA_TYPES,
    )
    robot_id = dataset.robot_ids[0]
    output_desc = {robot_id: _single_joint_output_desc(robot_id)}
    return input_desc, output_desc


def _input_output_embodiment_for_policy(
    input_desc: dict, output_desc: dict
) -> tuple[EmbodimentDescription, EmbodimentDescription]:
    robot_id = next(iter(input_desc))
    return input_desc[robot_id], output_desc[robot_id]


class TestDatasetMutationTraining:
    """End-to-end dataset mutation, statistics, and training correctness."""

    track_step_teardown = True
    all_steps_passed: bool = True

    dataset_name: str
    corrupt_dataset_name: str
    baseline_training_name: str
    retrain_training_name: str
    corrupt_training_name: str

    dataset: Dataset | None = None
    corrupt_dataset: Dataset | None = None
    ground_truth: dict[str, list[float]]
    corrupt_ground_truth: dict[str, list[float]]

    active_recording_ids: set[str]
    initial_recording_ids: set[str]
    deleted_recording_ids: set[str]
    added_recording_ids: set[str]

    input_desc: dict | None = None
    output_desc: dict | None = None
    stats_fingerprint_initial: str | None = None
    stats_fingerprint_mutated: str | None = None

    baseline_job_id: str | None = None
    retrain_job_id: str | None = None
    corrupt_job_id: str | None = None
    baseline_mse: float | None = None
    baseline_l1_loss: float | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="dataset_mutation")
        cls.corrupt_dataset_name = unique_name(prefix="dataset_corrupt")
        cls.baseline_training_name = unique_name(prefix="ml_dataset_mutation_baseline")
        cls.retrain_training_name = unique_name(prefix="ml_dataset_mutation_retrain")
        cls.corrupt_training_name = unique_name(prefix="ml_dataset_mutation_corrupt")
        cls.dataset = None
        cls.corrupt_dataset = None
        cls.ground_truth = {}
        cls.corrupt_ground_truth = {}
        cls.active_recording_ids = set()
        cls.initial_recording_ids = set()
        cls.deleted_recording_ids = set()
        cls.added_recording_ids = set()
        cls.input_desc = None
        cls.output_desc = None
        cls.stats_fingerprint_initial = None
        cls.stats_fingerprint_mutated = None
        cls.baseline_job_id = None
        cls.retrain_job_id = None
        cls.corrupt_job_id = None
        cls.baseline_mse = None
        cls.baseline_l1_loss = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestDatasetMutationTraining teardown: "
                "one or more steps failed"
            )
            return
        for job_id in (cls.baseline_job_id, cls.retrain_job_id, cls.corrupt_job_id):
            if job_id is None:
                continue
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete training job {job_id}", exc_info=True)
        for dataset in (cls.dataset, cls.corrupt_dataset):
            if dataset is None:
                continue
            try:
                dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete dataset {dataset.name}", exc_info=True
                )

    def test_step1_create_initial_dataset(self) -> None:
        dataset, ground_truth, new_ids = _collect_recordings(
            dataset_name=self.dataset_name,
            robot_name=ROBOT_NAME,
            instance_id=ROBOT_INSTANCE,
            num_episodes=INITIAL_RECORDINGS,
            start_episode_idx=0,
            ground_truth=self.ground_truth,
        )
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=INITIAL_RECORDINGS,
        )
        self.__class__.ground_truth = ground_truth
        self.__class__.initial_recording_ids = set(new_ids)
        self.__class__.active_recording_ids = set(new_ids)
        assert len(self.dataset) == INITIAL_RECORDINGS
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.dataset)} recordings "
            f"into '{self.dataset_name}'"
        )

    def test_step2_verify_initial_statistics(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        assert_dataset_metadata(
            self.dataset,
            expected_count=INITIAL_RECORDINGS,
            expected_common_types=EXPECTED_COMMON_TYPES,
        )
        assert_active_recordings(
            self.dataset,
            expected_count=INITIAL_RECORDINGS,
            expected_types=EXPECTED_RECORDING_TYPES,
            tracked_ids=self.initial_recording_ids,
        )
        input_desc, output_desc = _build_training_descriptions(self.dataset)
        self.__class__.input_desc = input_desc
        self.__class__.output_desc = output_desc
        _, fingerprint = assert_synced_statistics(
            self.dataset,
            input_desc,
            output_desc,
            expected_count=INITIAL_RECORDINGS,
            frequency=FREQUENCY,
        )
        self.__class__.stats_fingerprint_initial = fingerprint
        logger.info("[STEP 2] [PASSED] Initial dataset statistics verified")

    def test_step3_train_baseline_and_overfit(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        assert self.input_desc is not None, "[STEP 2] Did Not Complete"
        assert self.output_desc is not None, "[STEP 2] Did Not Complete"

        job_data = nc.start_training_run(
            name=self.baseline_training_name,
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=OVERFIT_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=self.input_desc,
            output_cross_embodiment_description=self.output_desc,
        )
        self.__class__.baseline_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.baseline_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Baseline training ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.baseline_job_id,
            context="Step 3 (baseline training)",
        )
        self.__class__.baseline_l1_loss = assert_training_loss_below(
            self.baseline_job_id,
            L1_LOSS_METRIC_KEY,
            L1_LOSS_THRESHOLD,
            context="Step 3 (baseline L1 loss)",
        )

        synced, _ = assert_synced_statistics(
            self.dataset,
            self.input_desc,
            self.output_desc,
            expected_count=INITIAL_RECORDINGS,
            frequency=FREQUENCY,
        )
        input_emb, output_emb = _input_output_embodiment_for_policy(
            self.input_desc, self.output_desc
        )
        policy = nc.policy(
            input_embodiment_description=input_emb,
            output_embodiment_description=output_emb,
            train_run_name=self.baseline_training_name,
        )
        try:
            self.__class__.baseline_mse = evaluate_training_mse(
                policy=policy,
                synced_dataset=synced,
                ground_truth_by_recording=self.ground_truth,
                target_joint_name=TARGET_JOINT,
            )
        finally:
            policy.disconnect()

        logger.info(
            f"[STEP 3] [PASSED] Baseline training overfit: "
            f"L1={self.baseline_l1_loss:.6g}, MSE={self.baseline_mse:.6g}"
        )

    def test_step4_mutate_dataset(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        dataset = nc.get_dataset(name=self.dataset_name)
        assert len(dataset) == INITIAL_RECORDINGS

        recordings_to_delete = [dataset[index] for index in range(MUTATION_DELETE)]
        self.__class__.deleted_recording_ids = {
            str(recording.id) for recording in recordings_to_delete
        }
        for recording in recordings_to_delete:
            logger.info(f"Deleting recording {recording.id!r}")
            delete_recording_from_dataset(dataset=dataset, recording=recording)

        remaining = INITIAL_RECORDINGS - MUTATION_DELETE
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=remaining,
        )
        surviving_ids = {str(r.id) for r in self.dataset}
        assert not self.deleted_recording_ids & surviving_ids

        known_ids = surviving_ids
        _, ground_truth, new_ids = _collect_recordings(
            dataset_name=self.dataset_name,
            robot_name=ROBOT_NAME,
            instance_id=ROBOT_INSTANCE,
            num_episodes=MUTATION_ADD,
            start_episode_idx=INITIAL_RECORDINGS,
            ground_truth=self.ground_truth,
            known_recording_ids=known_ids,
        )
        self.__class__.ground_truth = ground_truth
        self.__class__.added_recording_ids = new_ids
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=INITIAL_RECORDINGS,
        )
        self.__class__.active_recording_ids = surviving_ids | new_ids
        assert len(self.added_recording_ids) == MUTATION_ADD
        logger.info(
            f"[STEP 4] [PASSED] Mutated dataset: deleted {MUTATION_DELETE}, "
            f"added {MUTATION_ADD}, total {len(self.dataset)}"
        )

    def test_step5_verify_updated_statistics(self) -> None:
        assert self.dataset is not None, "[STEP 4] Did Not Complete"
        assert self.input_desc is not None, "[STEP 2] Did Not Complete"
        assert self.output_desc is not None, "[STEP 2] Did Not Complete"

        assert_dataset_metadata(
            self.dataset,
            expected_count=INITIAL_RECORDINGS,
            expected_common_types=EXPECTED_COMMON_TYPES,
        )
        active_ids = assert_active_recordings(
            self.dataset,
            expected_count=INITIAL_RECORDINGS,
            expected_types=EXPECTED_RECORDING_TYPES,
            tracked_ids=self.active_recording_ids,
        )
        assert self.added_recording_ids <= active_ids
        assert not self.deleted_recording_ids & active_ids

        _, fingerprint = assert_synced_statistics(
            self.dataset,
            self.input_desc,
            self.output_desc,
            expected_count=INITIAL_RECORDINGS,
            frequency=FREQUENCY,
        )
        self.__class__.stats_fingerprint_mutated = fingerprint
        assert (
            fingerprint != self.stats_fingerprint_initial
        ), "Expected dataset statistics to change after mutation"
        logger.info("[STEP 5] [PASSED] Updated dataset statistics verified")

    def test_step6_retrain_and_overfit(self) -> None:
        assert self.dataset is not None, "[STEP 4] Did Not Complete"
        assert self.input_desc is not None, "[STEP 2] Did Not Complete"
        assert self.output_desc is not None, "[STEP 2] Did Not Complete"

        job_data = nc.start_training_run(
            name=self.retrain_training_name,
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=OVERFIT_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=self.input_desc,
            output_cross_embodiment_description=self.output_desc,
            name_auto_increment=True,
        )
        self.__class__.retrain_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.retrain_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Retrain ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.retrain_job_id,
            context="Step 6 (retrain)",
        )
        assert_training_loss_below(
            self.retrain_job_id,
            L1_LOSS_METRIC_KEY,
            L1_LOSS_THRESHOLD,
            context="Step 6 (retrain L1 loss)",
        )

        synced, _ = assert_synced_statistics(
            self.dataset,
            self.input_desc,
            self.output_desc,
            expected_count=INITIAL_RECORDINGS,
            frequency=FREQUENCY,
        )
        synced_ids = {str(recording.id) for recording in synced.dataset}
        assert not self.deleted_recording_ids & synced_ids

        input_emb, output_emb = _input_output_embodiment_for_policy(
            self.input_desc, self.output_desc
        )
        policy = nc.policy(
            input_embodiment_description=input_emb,
            output_embodiment_description=output_emb,
            train_run_name=self.retrain_training_name,
        )
        try:
            evaluate_training_mse(
                policy=policy,
                synced_dataset=synced,
                ground_truth_by_recording=self.ground_truth,
                target_joint_name=TARGET_JOINT,
                excluded_recording_ids=self.deleted_recording_ids,
            )
        finally:
            policy.disconnect()

        logger.info("[STEP 6] [PASSED] Retrain overfit on mutated dataset")

    def test_step7_corrupted_modality_fails_to_overfit(self) -> None:
        corrupt_dataset, corrupt_ground_truth, _ = _collect_recordings(
            dataset_name=self.corrupt_dataset_name,
            robot_name=ROBOT_NAME,
            instance_id=ROBOT_INSTANCE,
            num_episodes=CORRUPT_RECORDINGS,
            start_episode_idx=0,
            corrupt_camera=True,
            ground_truth={},
        )
        self.__class__.corrupt_dataset = wait_for_dataset_recording_count(
            dataset_name=self.corrupt_dataset_name,
            expected_recordings=CORRUPT_RECORDINGS,
        )
        self.__class__.corrupt_ground_truth = corrupt_ground_truth

        corrupt_input_desc, corrupt_output_desc = _build_training_descriptions(
            self.corrupt_dataset
        )
        job_data = nc.start_training_run(
            name=self.corrupt_training_name,
            dataset_name=self.corrupt_dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=OVERFIT_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=corrupt_input_desc,
            output_cross_embodiment_description=corrupt_output_desc,
        )
        self.__class__.corrupt_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.corrupt_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Corrupt training ended with non-COMPLETED status: {final_status}"

        corrupt_l1 = get_final_metric_value(self.corrupt_job_id, L1_LOSS_METRIC_KEY)
        assert corrupt_l1 is not None, "Corrupt run missing L1 loss metric"
        if self.baseline_l1_loss is not None:
            assert corrupt_l1 >= self.baseline_l1_loss, (
                f"Corrupt L1 loss {corrupt_l1:.6g} should be >= baseline "
                f"{self.baseline_l1_loss:.6g}"
            )

        synced, _ = assert_synced_statistics(
            self.corrupt_dataset,
            corrupt_input_desc,
            corrupt_output_desc,
            expected_count=CORRUPT_RECORDINGS,
            frequency=FREQUENCY,
        )
        input_emb, output_emb = _input_output_embodiment_for_policy(
            corrupt_input_desc, corrupt_output_desc
        )
        policy = nc.policy(
            input_embodiment_description=input_emb,
            output_embodiment_description=output_emb,
            train_run_name=self.corrupt_training_name,
        )
        try:
            corrupt_mse = _evaluate_mean_mse_without_threshold(
                policy=policy,
                synced_dataset=synced,
                ground_truth_by_recording=self.corrupt_ground_truth,
                target_joint_name=TARGET_JOINT,
            )
        finally:
            policy.disconnect()

        assert self.baseline_mse is not None, "[STEP 3] baseline MSE missing"
        assert corrupt_mse >= max(
            CORRUPT_MSE_MIN, self.baseline_mse * CORRUPT_MSE_FACTOR
        ), (
            f"Corrupt MSE {corrupt_mse:.6g} should be much worse than baseline "
            f"{self.baseline_mse:.6g}"
        )
        logger.info(
            f"[STEP 7] [PASSED] Corrupted camera ablation: "
            f"baseline MSE={self.baseline_mse:.6g}, corrupt MSE={corrupt_mse:.6g}, "
            f"baseline L1={self.baseline_l1_loss:.6g}, corrupt L1={corrupt_l1:.6g}"
        )


def _evaluate_mean_mse_without_threshold(
    *,
    policy,
    synced_dataset,
    ground_truth_by_recording: dict[str, list[float]],
    target_joint_name: str,
) -> float:
    """Compute mean MSE without per-frame threshold (for ablation comparison)."""
    total_squared_error = 0.0
    total_frames = 0
    for recording in synced_dataset:
        recording_id = str(recording.id)
        expected_targets = ground_truth_by_recording[recording_id]
        for frame_idx, sync_point in enumerate(recording):
            predictions = policy.predict(sync_point=sync_point, timeout=60)
            joint_targets = predictions[DataType.JOINT_TARGET_POSITIONS]
            predicted = float(joint_targets[target_joint_name].value[0, 0, 0].item())
            expected = expected_targets[frame_idx]
            total_squared_error += (predicted - expected) ** 2
            total_frames += 1
    assert total_frames > 0
    return total_squared_error / total_frames
