"""Integration tests verifying per-algorithm success rates on the Transfer Cube task.

The suite is split into two phases so CI runners are not held idle during training:

  test_start_training  — submits a training job and records the job ID.
  test_evaluate        — waits for the job, runs every evaluation variant listed
                         under that algorithm in algorithm_configs.yaml, then
                         deletes the training job only if all variants pass.

Evaluation modes (per item in ``evaluations``):

  omit / endpoint      — remote endpoint + open-loop chunking (default)
  realtime_chunking    — in-process RTC via :func:`neuracore.policy_realtime`
  temporal_ensemble    — in-process temporal ensemble via ``policy_realtime``

When ``evaluations`` is omitted, a single endpoint eval uses top-level
``min_success_rate``. Variants that share one training job stay under one
config so cleanup is deferred until every variant meets its threshold.

Running locally
---------------
    # Phase 1 — kick off training (job ID is printed in the log output)
    ALGORITHM_NAME=ACT pytest -k test_start_training -v <this file>

    # Phase 2 — evaluate once training is complete
    ALGORITHM_NAME=ACT TRAINING_JOB_ID=<id> pytest -k test_evaluate -v <this file>

    # Running the full file without TRAINING_JOB_ID: test_start_training runs for
    # all algorithms, test_evaluate skips cleanly for each.
"""

import json
import logging
import os
import sys
import time
from typing import Any, cast

import numpy as np
import pytest
import torch
from neuracore_types import (
    BatchedJointData,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)
from neuracore_types.training.training import GPUType

import neuracore as nc
from neuracore.core.endpoint import Policy, RealTimePolicy
from neuracore.ml.utils.real_time_chunking import RTCConfig
from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig
from tests.integration.ml.shared.training import (
    get_training_failure_context,
    wait_for_training,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.transfer_cube import (
    BOX_POSE,
    BimanualViperXTask,
    TransferCubeTask,
    make_sim_env,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPISODE_LENGTH: int = 400
NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
MAX_REWARD = 4.0
ENDPOINT_NAME = "Integration Test Endpoint"
TRAINING_NAME = "Integration Test"
DATASET_NAME = "Transfer Cube VX300s Dataset"
DEFAULT_GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 50
NUM_ROLLOUTS = 20
# Training should be complete by the time Phase 2 runs; this is a safety buffer
# for jobs that finish slightly after the Phase 2 workflow starts.
TRAINING_WAIT_TIMEOUT_MINUTES = 60

JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES
    + BimanualViperXTask.LEFT_GRIPPER_JOINT_NAMES
    + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    + BimanualViperXTask.RIGHT_GRIPPER_JOINT_NAMES
)


def _indexed_names(names: list[str] | tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


INPUT_EMBODIMENT_DESCRIPTION = {
    DataType.RGB_IMAGES: _indexed_names([NC_CAM_NAME]),
    DataType.JOINT_POSITIONS: _indexed_names(JOINT_NAMES),
}
OUTPUT_EMBODIMENT_DESCRIPTION = {
    DataType.JOINT_TARGET_POSITIONS: _indexed_names(BimanualViperXTask.ACTION_KEYS),
}


# Distinguishes "key absent" (keep the dataclass default) from an explicit
# ``num_inference_steps: null`` (keep whatever the model was trained with).
_UNSET = object()


def _evaluation_label(evaluation: dict[str, Any]) -> str:
    """Human-readable name for an evaluation variant."""
    mode = evaluation.get("mode") or "endpoint"
    return str(evaluation.get("name") or mode)


def _evaluations(algorithm_config_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evaluation variants for one training job.

    Explicit ``evaluations`` lists win. Otherwise a single endpoint eval is
    synthesised from top-level ``min_success_rate``.
    """
    evaluations = algorithm_config_entry.get("evaluations")
    if evaluations:
        return list(evaluations)
    return [{"min_success_rate": algorithm_config_entry["min_success_rate"]}]


def _min_success_rate(
    evaluation: dict[str, Any], algorithm_config_entry: dict[str, Any]
) -> float:
    """Threshold for one variant (per-eval override, else top-level)."""
    if "min_success_rate" in evaluation:
        return float(evaluation["min_success_rate"])
    return float(algorithm_config_entry["min_success_rate"])


def _make_sync_point(obs: Any) -> SynchronizedPoint:
    """Build a model observation from a MuJoCo env observation."""
    return SynchronizedPoint(
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=obs.qpos[name]) for name in JOINT_NAMES
            },
            DataType.RGB_IMAGES: {
                NC_CAM_NAME: RGBCameraData(frame=obs.cameras[MJ_CAM_NAME].rgb),
            },
        },
    )


def eval_model(
    policy: Policy,
    env: TransferCubeTask,
    num_rollouts: int,
) -> float:
    success = 0
    for episode_idx in range(num_rollouts):
        logger.info(f"Starting rollout {episode_idx + 1} / {num_rollouts}")
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()
        episode_max = 0
        horizon = 1
        actions = []
        for i in range(EPISODE_LENGTH):
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                obs = env.get_observation()
                sync_point = _make_sync_point(obs)
                predictions = policy.predict(sync_point=sync_point, timeout=10)
                joint_target_positions = cast(
                    dict[str, BatchedJointData],
                    predictions[DataType.JOINT_TARGET_POSITIONS],
                )

                left_arm = torch.cat(
                    [
                        joint_target_positions[n].value
                        for n in BimanualViperXTask.LEFT_ARM_JOINT_NAMES
                    ],
                    dim=2,
                )
                right_arm = torch.cat(
                    [
                        joint_target_positions[n].value
                        for n in BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
                    ],
                    dim=2,
                )
                left_gripper = joint_target_positions[
                    BimanualViperXTask.LEFT_GRIPPER_OPEN
                ].value
                right_gripper = joint_target_positions[
                    BimanualViperXTask.RIGHT_GRIPPER_OPEN
                ].value

                batched_actions = (
                    torch.cat([left_arm, left_gripper, right_arm, right_gripper], dim=2)
                    .cpu()
                    .numpy()
                )
                actions = batched_actions[0]
                horizon = len(actions)

            a = actions[idx_in_horizon]
            # To save on rendering time during action sequences,
            # we do an explicit get_observation() every prediction step
            obs, reward, done = env.step(a, no_obs=True)
            episode_max = max(episode_max, reward)

        if episode_max >= MAX_REWARD:
            success += 1

    return success / num_rollouts


def eval_model_realtime(
    policy: RealTimePolicy,
    env: TransferCubeTask,
    num_rollouts: int,
    *,
    control_hz: float,
) -> float:
    """Evaluate a policy under the unified async chunking controller.

    Inference runs in a background thread while the control loop consumes one
    action per tick. Observations are pushed via ``get_action(obs)`` each tick.
    """
    tick_period = 1.0 / control_hz
    success = 0
    for episode_idx in range(num_rollouts):
        logger.info(
            f"Starting {policy.mode} rollout {episode_idx + 1} / {num_rollouts}"
        )
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()
        sync = _make_sync_point(obs)
        episode_max = 0.0

        policy.start()
        try:
            # Seed the controller with the first observation, then wait.
            policy.get_action(sync)
            if not policy.wait_for_first_chunk(timeout=120.0):
                raise TimeoutError(
                    "Timed out waiting for the first realtime action chunk"
                )
            for _ in range(EPISODE_LENGTH):
                tick_started = time.monotonic()
                action = policy.get_action(sync)
                if action is None:
                    raise RuntimeError("Realtime policy returned no action")
                obs, reward, done = env.step(np.asarray(action, dtype=np.float64))
                sync = _make_sync_point(obs)
                episode_max = max(episode_max, reward)
                remaining = tick_period - (time.monotonic() - tick_started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            policy.stop(timeout=60.0)

        stats = policy.stats()
        logger.info(
            "%s stats: chunks=%d d=%d s=%d deadline_misses=%d stalled_ticks=%d "
            "median_latency_ms=%.1f",
            policy.mode,
            stats.chunks,
            stats.inference_delay,
            stats.execution_horizon,
            stats.deadline_misses,
            stats.stalled_ticks,
            stats.median_latency_s * 1e3,
        )
        if episode_max >= MAX_REWARD:
            success += 1

    return success / num_rollouts


def _evaluate_remote_endpoint(
    algorithm_name: str,
    training_job_id: str,
    algorithm_config_entry: dict[str, Any],
) -> float:
    """Deploy a remote endpoint and evaluate with ordinary open-loop chunking."""
    timestamp = int(time.time())
    endpoint_name = f"{ENDPOINT_NAME} - {algorithm_name} - {timestamp}"
    endpoint_id = None
    try:
        endpoint_data = nc.deploy_model(
            job_id=training_job_id,
            name=endpoint_name,
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            ttl=60 * 30,
            gpu_type=GPUType(algorithm_config_entry.get("gpu_type", DEFAULT_GPU_TYPE)),
        )
        endpoint_id = endpoint_data["id"]

        endpoint_status = nc.get_endpoint_status(endpoint_id=endpoint_id)
        while endpoint_status == "creating":
            logger.info(
                f"[{algorithm_name}] Waiting for endpoint: status={endpoint_status}"
            )
            time.sleep(60)
            endpoint_status = nc.get_endpoint_status(endpoint_id=endpoint_id)

        if endpoint_status != "active":
            raise ValueError(
                f"[{algorithm_name}] Endpoint did not become active: {endpoint_status}"
            )

        policy = nc.policy_remote_server(endpoint_name)
        env = make_sim_env(seed=42)
        success_rate = eval_model(
            policy=policy,
            env=env,
            num_rollouts=NUM_ROLLOUTS,
        )
        policy.disconnect()
    except Exception:
        if endpoint_id is not None:
            nc.delete_endpoint(endpoint_id)
        raise

    nc.delete_endpoint(endpoint_id)
    return success_rate


def _evaluate_realtime(
    algorithm_name: str,
    training_job_id: str,
    evaluation: dict[str, Any],
) -> float:
    """Load the trained model in-process and evaluate under RTC or TE."""
    job_data = nc.get_training_job_data(training_job_id)
    train_run_name = job_data["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = evaluation["mode"]
    label = _evaluation_label(evaluation)
    if mode == "realtime_chunking":
        api_mode = "rtc"
        config: RTCConfig | TemporalEnsembleConfig = RTCConfig(
            inference_delay=int(evaluation["inference_delay"]),
            execution_horizon=int(evaluation["execution_horizon"]),
        )
        adapt = bool(evaluation.get("adapt_inference_delay", True))
    elif mode == "temporal_ensemble":
        api_mode = "temporal_ensemble"
        steps = evaluation.get("num_inference_steps", _UNSET)
        config = TemporalEnsembleConfig(
            execution_horizon=int(evaluation["execution_horizon"]),
            m=float(evaluation.get("m", 0.01)),
            blend_steps=int(evaluation.get("blend_steps", 0)),
            **(
                {}
                if steps is _UNSET
                else {"num_inference_steps": None if steps is None else int(steps)}
            ),
        )
        adapt = False
    else:
        raise ValueError(f"Unknown realtime evaluation mode {mode!r}")

    logger.info(
        f"[{algorithm_name}/{label}] Loading realtime policy ({api_mode}) from "
        f"{train_run_name!r} on {device}"
    )
    policy = nc.policy_realtime(
        train_run_name=train_run_name,
        input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
        output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        device=device,
        mode=api_mode,
        config=config,
        control_hz=float(evaluation.get("control_hz", FREQUENCY)),
        adapt_inference_delay=adapt,
    )
    try:
        env = make_sim_env(seed=42)
        return eval_model_realtime(
            policy=policy,
            env=env,
            num_rollouts=NUM_ROLLOUTS,
            control_hz=float(evaluation.get("control_hz", FREQUENCY)),
        )
    finally:
        policy.disconnect()


def _run_evaluation(
    algorithm_name: str,
    training_job_id: str,
    algorithm_config_entry: dict[str, Any],
    evaluation: dict[str, Any],
) -> float:
    """Dispatch one evaluation variant and return its success rate."""
    mode = evaluation.get("mode") or "endpoint"
    if mode in ("realtime_chunking", "temporal_ensemble"):
        return _evaluate_realtime(algorithm_name, training_job_id, evaluation)
    if mode in ("endpoint", "remote"):
        return _evaluate_remote_endpoint(
            algorithm_name, training_job_id, algorithm_config_entry
        )
    raise ValueError(
        f"[{algorithm_name}] Unknown evaluation mode {mode!r}; "
        "expected endpoint, realtime_chunking, or temporal_ensemble"
    )


class TestAlgorithmPerformance:
    def test_start_training(self, algorithm_config_entry: dict) -> None:
        """Phase 1: start a training job and record its ID.

        In CI this writes the job ID to $GITHUB_OUTPUT so the Phase 2 workflow
        can cache and forward it. Locally the ID is logged at INFO level.
        """
        algorithm_name = algorithm_config_entry["name"]

        nc.login()

        dataset = nc.get_dataset(DATASET_NAME)
        robot_ids = dataset.robot_ids
        assert len(robot_ids) == 1, f"Expected one robot in dataset, got {robot_ids}"
        robot_id = robot_ids[0]

        timestamp = int(time.time())
        gpu_type = algorithm_config_entry.get("gpu_type", DEFAULT_GPU_TYPE)
        logger.info(f"[{algorithm_name}] Starting training job...")
        job_data = nc.start_training_run(
            name=f"{TRAINING_NAME} - {algorithm_name} - {timestamp}",
            gpu_type=gpu_type,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            algorithm_name=algorithm_name,
            dataset_name=DATASET_NAME,
            algorithm_config=algorithm_config_entry["algorithm_config"],
            input_cross_embodiment_description={robot_id: INPUT_EMBODIMENT_DESCRIPTION},
            output_cross_embodiment_description={
                robot_id: OUTPUT_EMBODIMENT_DESCRIPTION
            },
        )
        training_job_id = job_data["id"]
        logger.info(f"[{algorithm_name}] Training job started: {training_job_id}")

        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"training_job_id={training_job_id}\n")

    def test_evaluate(self, algorithm_config_entry: dict) -> None:
        """Phase 2: run every evaluation variant for one training job.

        Expects TRAINING_JOB_ID from Phase 1. All variants listed under
        ``evaluations`` (or a synthesised endpoint eval) run in this test;
        the training job is deleted only when every variant meets its
        ``min_success_rate``.
        """
        training_job_id = os.environ.get("TRAINING_JOB_ID")
        if not training_job_id:
            pytest.skip(
                "TRAINING_JOB_ID not set — run test_start_training first, "
                "then re-run with TRAINING_JOB_ID=<id>"
            )

        algorithm_name = algorithm_config_entry["name"]
        evaluations = _evaluations(algorithm_config_entry)

        nc.login()

        # Training should already be complete; poll briefly as a safety buffer.
        training_job_status = wait_for_training(
            job_id=training_job_id,
            timeout_minutes=TRAINING_WAIT_TIMEOUT_MINUTES,
            poll_seconds=60,
        )
        if training_job_status != "COMPLETED":
            failure_context = get_training_failure_context(training_job_id)
            raise ValueError(
                f"[{algorithm_name}] Training job did not complete, "
                f"status: {training_job_status}\n\n{failure_context}"
            )

        rates: dict[str, float] = {}
        failures: list[str] = []
        for evaluation in evaluations:
            label = _evaluation_label(evaluation)
            min_rate = _min_success_rate(evaluation, algorithm_config_entry)
            logger.info(
                f"[{algorithm_name}/{label}] Starting evaluation "
                f"(threshold={min_rate:.2%})"
            )
            success_rate = _run_evaluation(
                algorithm_name,
                training_job_id,
                algorithm_config_entry,
                evaluation,
            )
            rates[label] = success_rate
            logger.info(
                f"[{algorithm_name}/{label}] success_rate={success_rate:.2%} "
                f"(threshold={min_rate:.2%})"
            )
            if success_rate < min_rate:
                failures.append(f"{label}: {success_rate:.2%} < {min_rate:.2%}")

        # Report rates even when thresholds fail so the CI dashboard can show
        # what each variant achieved.
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            # Prefer the endpoint rate for the legacy single-value field; fall
            # back to the first variant. Full map goes in success_rates.
            primary = rates.get("endpoint", next(iter(rates.values())))
            with open(github_output, "a") as f:
                f.write(f"success_rate={primary}\n")
                f.write(f"success_rates={json.dumps(rates)}\n")

        if failures:
            raise ValueError(
                f"[{algorithm_name}] evaluation threshold(s) not met; "
                f"training job {training_job_id} left in place for inspection:\n"
                + "\n".join(f"  - {msg}" for msg in failures)
            )

        nc.delete_training_job(training_job_id)
        logger.info(
            f"[{algorithm_name}] All {len(rates)} evaluation(s) passed; "
            f"deleted training job {training_job_id}"
        )
