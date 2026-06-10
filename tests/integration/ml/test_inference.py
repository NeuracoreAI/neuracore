"""Inference integration tests (direct, local server, remote endpoint).

These tests are decoupled from the end-to-end flow test: instead of relying
on in-memory shared state, they discover the trained model by scanning for the
latest COMPLETED training run under ``INFERENCE_MODEL_TRAIN_RUN_PREFIX``. If no
such run exists (e.g. the e2e flow did not run first), they error out.

The resolved run is logged prominently and echoed in every step's PASSED line
so it is unmistakable which run inference is using. The selected run is
intentionally NOT deleted in teardown: it persists as a known-good model so
that a future, separate test session whose own training fails can still fall
back to this previously successful run. Superseded prefixed runs (older
COMPLETED runs and FAILED leftovers) are pruned instead.
"""

import logging
import os
import sys
import time

from neuracore_types import (
    DataType,
    JointData,
    LanguageData,
    ParallelGripperOpenAmountData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc
from neuracore.core.endpoint import Policy
from tests.integration.ml.shared.constants import (
    GRIPPER_NAMES,
    INPUT_EMBODIMENT_DESCRIPTION,
    JOINT_NAMES,
    LANGUAGE_LABEL,
    MJ_CAM_NAME,
    NC_CAM_NAME,
    OUTPUT_DATA_TYPES,
    OUTPUT_EMBODIMENT_DESCRIPTION,
)
from tests.integration.ml.shared.utils import (
    SelectedRun,
    prune_training_runs_except,
    resolve_latest_completed_run,
    unique_name,
)
from tests.integration.ml.test_training_flow import INFERENCE_MODEL_TRAIN_RUN_PREFIX

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.transfer_cube import make_sim_env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ENDPOINT_POLL_SECONDS = 20
ENDPOINT_TIMEOUT_MINUTES = 30
ENDPOINT_TTL_SECONDS = 60 * 30
LOCAL_SERVER_PORT = 8181
MUJOCO_ROBOT_NAME = "Mujoco VX300s"


def wait_for_endpoint(
    endpoint_id: str, timeout_minutes: int = ENDPOINT_TIMEOUT_MINUTES
) -> str:
    deadline = time.time() + timeout_minutes * 60
    while True:
        status = nc.get_endpoint_status(endpoint_id=endpoint_id)
        logger.info(f"Endpoint {endpoint_id}: {status}")
        if status != "creating":
            return status
        assert time.time() < deadline, (
            f"Endpoint {endpoint_id} did not become "
            f"active within {timeout_minutes} minutes"
        )
        time.sleep(ENDPOINT_POLL_SECONDS)


def make_sync_point(
    obs,
    *,
    joint_names: tuple[str, ...] | list[str],
    gripper_names: list[str],
    language_label: str,
    nc_cam_name: str,
    mj_cam_name: str,
) -> SynchronizedPoint:
    return SynchronizedPoint(
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=obs.qpos[name]) for name in joint_names
            },
            DataType.RGB_IMAGES: {
                nc_cam_name: RGBCameraData(frame=obs.cameras[mj_cam_name].rgb)
            },
            DataType.LANGUAGE: {language_label: LanguageData(text="pick and place")},
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                name: ParallelGripperOpenAmountData(
                    open_amount=float(obs.gripper_open_amounts[name])
                )
                for name in gripper_names
            },
        }
    )


def run_policy_inference(
    policy: Policy,
    *,
    joint_names: tuple[str, ...] | list[str],
    gripper_names: list[str],
    language_label: str,
    nc_cam_name: str,
    mj_cam_name: str,
    output_data_types: list[DataType],
) -> None:
    """Run inference via both sync-point and logging-function paths."""
    try:
        env = make_sim_env(seed=42)
        obs = env.reset()

        logger.info("Running sync-point inference (Path 1)")
        predictions = policy.predict(
            sync_point=make_sync_point(
                obs=obs,
                joint_names=joint_names,
                gripper_names=gripper_names,
                language_label=language_label,
                nc_cam_name=nc_cam_name,
                mj_cam_name=mj_cam_name,
            ),
            timeout=30,
        )
        for data_type in output_data_types:
            assert data_type in predictions, (
                f"Expected {data_type.value} in local "
                f"server output, got: {list(predictions.keys())}"
            )
        logger.info(f"Path 1 passed — output keys: {[k.value for k in predictions]}")

        logger.info("Running logging-function inference (Path 2)")
        nc.log_joint_positions(
            positions={name: float(obs.qpos[name]) for name in joint_names}
        )
        nc.log_language(name=language_label, language="pick and place")
        nc.log_parallel_gripper_open_amounts(
            values={
                name: float(obs.gripper_open_amounts[name]) for name in gripper_names
            }
        )
        nc.log_rgb(name=nc_cam_name, rgb=obs.cameras[mj_cam_name].rgb)
        predictions = policy.predict(timeout=30)
        for data_type in output_data_types:
            assert data_type in predictions, (
                f"Expected {data_type.value} in local "
                f"server output, got: {list(predictions.keys())}"
            )
        logger.info(f"Path 2 passed — output keys: {[k.value for k in predictions]}")
    finally:
        policy.disconnect()


class TestInference:
    """Serve the latest COMPLETED training run via direct, local, and remote paths."""

    track_step_teardown = True
    all_steps_passed: bool = True
    selected_run: SelectedRun | None = None
    endpoint_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.selected_run = None
        cls.endpoint_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestInference teardown cleanup: one or more steps failed"
            )
            return
        if cls.endpoint_id:
            try:
                nc.delete_endpoint(cls.endpoint_id)
            except Exception:
                logger.warning(
                    f"Failed to delete endpoint {cls.endpoint_id}", exc_info=True
                )
        # Keep the run we used as a known-good model for future sessions (so a
        # later run whose training fails can fall back to it), and prune the
        # superseded prefixed runs to avoid unbounded job accumulation.
        if cls.selected_run is not None:
            try:
                prune_training_runs_except(
                    INFERENCE_MODEL_TRAIN_RUN_PREFIX, keep_id=cls.selected_run.id
                )
            except Exception:
                logger.warning(
                    "Failed to prune superseded training runs", exc_info=True
                )

    def test_step1_resolve_trained_model(self) -> None:
        self.__class__.selected_run = resolve_latest_completed_run(
            INFERENCE_MODEL_TRAIN_RUN_PREFIX
        )
        logger.info(
            f"[STEP 1] [PASSED] Using training run name='{self.selected_run.name}'"
            f" id={self.selected_run.id}"
        )

    def test_step2_direct_inference(self) -> None:
        assert self.selected_run is not None, "[STEP 1] Did Not Complete"
        nc.connect_robot(robot_name=MUJOCO_ROBOT_NAME)
        policy = nc.policy(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            train_run_name=self.selected_run.name,
        )
        run_policy_inference(
            policy=policy,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            mj_cam_name=MJ_CAM_NAME,
            output_data_types=OUTPUT_DATA_TYPES,
        )
        logger.info(
            f"[STEP 2] [PASSED] Direct In-Process Inference Succeeded"
            f" (run name='{self.selected_run.name}' id={self.selected_run.id})"
        )

    def test_step3_local_server_inference(self) -> None:
        assert self.selected_run is not None, "[STEP 1] Did Not Complete"
        policy = nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            train_run_name=self.selected_run.name,
            port=LOCAL_SERVER_PORT,
        )
        run_policy_inference(
            policy=policy,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            mj_cam_name=MJ_CAM_NAME,
            output_data_types=OUTPUT_DATA_TYPES,
        )
        logger.info(
            f"[STEP 3] [PASSED] Local Server Inference Succeeded"
            f" (run name='{self.selected_run.name}' id={self.selected_run.id})"
        )

    def test_step4_deploy_remote_endpoint(self) -> None:
        assert self.selected_run is not None, "[STEP 1] Did Not Complete"
        endpoint_name = unique_name(prefix="flow_endpoint")
        endpoint_data = nc.deploy_model(
            job_id=self.selected_run.id,
            name=endpoint_name,
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            ttl=ENDPOINT_TTL_SECONDS,
        )
        self.__class__.endpoint_id = endpoint_data["id"]
        assert self.endpoint_id is not None
        final_status = wait_for_endpoint(endpoint_id=self.endpoint_id)
        assert (
            final_status == "active"
        ), f"Endpoint did not become active, status: {final_status!r}"
        logger.info(
            f"[STEP 4] [PASSED] Endpoint {self.endpoint_id} Is Active"
            f" (run name='{self.selected_run.name}' id={self.selected_run.id})"
        )
