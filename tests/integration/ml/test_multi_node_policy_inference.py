"""Peer-to-peer multi-node policy inference.

A robot instance can be driven by several processes at once. Here one node
logs joint positions, a second logs camera frames, and the main process — which
logs nothing at all — runs the policy. `get_latest_sync_point` merges what every
remote node provides into the sync point `Policy.predict()` consumes, so a
successful prediction is only possible if the peer-to-peer transport delivered
both modalities.
"""

import logging
import multiprocessing
import random
import time
import traceback
from collections.abc import Generator
from copy import deepcopy
from multiprocessing.synchronize import Event
from pathlib import Path

import numpy as np
import pytest
from neuracore_types import (
    BatchedJointData,
    CrossEmbodimentDescription,
    DataType,
    EmbodimentDescription,
    ModelInitDescription,
)
from ordered_set import OrderedSet

import neuracore as nc
from neuracore.core.endpoint import Policy
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP
from neuracore.ml.datasets.pytorch_dummy_dataset import (
    MAX_LEN_PER_DATA_TYPE,
    PytorchDummyDataset,
)
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.preprocessing.methods.resize_pad import ResizePad
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.nc_archive import create_nc_archive
from tests.integration.ml.shared.utils import unique_name
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    delete_cloud_robot,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = get_default_device()
INPUT_DATA_TYPES = OrderedSet([DataType.JOINT_POSITIONS, DataType.RGB_IMAGES])
OUTPUT_DATA_TYPES = OrderedSet([DataType.JOINT_TARGET_POSITIONS])


def _indexed_names(data_type: DataType) -> dict[int, str]:
    return {
        index: f"{data_type.value}_{index}" for index in range(MAX_LEN_PER_DATA_TYPE)
    }


INPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    data_type: _indexed_names(data_type) for data_type in INPUT_DATA_TYPES
}
OUTPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    data_type: _indexed_names(data_type) for data_type in OUTPUT_DATA_TYPES
}

JOINT_NAMES = tuple(INPUT_EMBODIMENT_DESCRIPTION[DataType.JOINT_POSITIONS].values())
CAMERA_NAMES = tuple(INPUT_EMBODIMENT_DESCRIPTION[DataType.RGB_IMAGES].values())

ROBOT_INSTANCE = 0
ROBOT_NAME_PREFIX = "multinode_policy_robot"

FRAME_SHAPE = (64, 64, 3)
NODE_READY_TIMEOUT_S = 180
NODE_CONNECT_TIMEOUT_S = 180
NODE_CONNECT_POLL_S = 1.0
NODE_JOIN_TIMEOUT_S = 15
PREDICT_TIMEOUT_S = 60

DUMMY_DATASET_SAMPLES = 5
TRAIN_BATCH_SIZE = 2


def _joint_positions() -> dict[str, float]:
    """Distinct per-joint values so the merge can be checked name by name."""
    return {name: 0.1 * (index + 1) for index, name in enumerate(JOINT_NAMES)}


def _camera_frames() -> dict[str, np.ndarray]:
    """Distinct per-camera frames, each with an identifying corner pixel."""
    frames = {}
    for index, name in enumerate(CAMERA_NAMES):
        frame = np.full(FRAME_SHAPE, 40 * (index + 1), dtype=np.uint8)
        frame[0, 0] = [255, index, 0]
        frames[name] = frame
    return frames


def _build_cnnmlp_archive(output_dir: Path) -> Path:
    """Train a CNNMLP for one step and save it as an .nc.zip archive.

    Mirrors the archive-export path in ``neuracore.ml.utils.validate``.

    Args:
        output_dir: Directory the archive is written into.

    Returns:
        Path to the written ``model.nc.zip``.
    """
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": dict(INPUT_EMBODIMENT_DESCRIPTION)
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": dict(OUTPUT_EMBODIMENT_DESCRIPTION)
    }
    dataset = PytorchDummyDataset(
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        num_samples=DUMMY_DATASET_SAMPLES,
    )
    model_init_description = ModelInitDescription(
        input_data_types=INPUT_DATA_TYPES,
        output_data_types=OUTPUT_DATA_TYPES,
        input_dataset_statistics={
            data_type: deepcopy(dataset.dataset_statistics["input"][data_type])
            for data_type in INPUT_DATA_TYPES
        },
        output_dataset_statistics={
            data_type: deepcopy(dataset.dataset_statistics["output"][data_type])
            for data_type in OUTPUT_DATA_TYPES
        },
        output_prediction_horizon=dataset.output_prediction_horizon,
    )

    model = CNNMLP(model_init_description=model_init_description).to(DEVICE)
    logger.info(
        f"Built CNNMLP with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # PolicyInference rejects an archive with no input preprocessing config.
    preprocessing_config = PreprocessingConfiguration({
        DataType.RGB_IMAGES: [ResizePad(size=(224, 224))],
    })
    archive_path = create_nc_archive(
        model,
        output_dir,
        {},
        input_cross_embodiment_description,
        output_cross_embodiment_description,
        preprocessing_config,
        preprocessing_config,
    )
    logger.info(f"Saved model archive to {archive_path}")
    return archive_path


def _joint_node_worker(
    robot_name: str,
    instance: int,
    joint_positions: dict[str, float],
    ready_event: Event,
    stop_event: Event,
    result_queue: multiprocessing.Queue,
) -> None:
    """Remote node A: logs joint positions and nothing else."""
    import neuracore as nc_remote

    try:
        nc_remote.login()
        nc_remote.connect_robot(robot_name, instance=instance, overwrite=False)
        nc_remote.log_joint_positions(
            joint_positions, robot_name=robot_name, instance=instance
        )
        result_queue.put({"ok": True, "node": "joints"})
        ready_event.set()
        stop_event.wait()
    except BaseException:
        result_queue.put({
            "ok": False,
            "node": "joints",
            "traceback": traceback.format_exc(),
        })
        ready_event.set()


def _camera_node_worker(
    robot_name: str,
    instance: int,
    frames: dict[str, np.ndarray],
    ready_event: Event,
    stop_event: Event,
    result_queue: multiprocessing.Queue,
) -> None:
    """Remote node B: logs camera frames and nothing else."""
    import neuracore as nc_remote

    try:
        nc_remote.login()
        nc_remote.connect_robot(robot_name, instance=instance, overwrite=False)
        for name, frame in frames.items():
            nc_remote.log_rgb(
                name=name, rgb=frame, robot_name=robot_name, instance=instance
            )
        result_queue.put({"ok": True, "node": "camera"})
        ready_event.set()
        stop_event.wait()
    except BaseException:
        result_queue.put({
            "ok": False,
            "node": "camera",
            "traceback": traceback.format_exc(),
        })
        ready_event.set()


def _wait_for_remote_nodes(
    robot_name: str, instance: int, num_remote_nodes: int
) -> None:
    """Block until the expected remote nodes have delivered data."""
    deadline = time.time() + NODE_CONNECT_TIMEOUT_S
    while not nc.check_remote_nodes_connected(
        num_remote_nodes=num_remote_nodes, robot_name=robot_name, instance=instance
    ):
        assert time.time() < deadline, (
            f"Timed out after {NODE_CONNECT_TIMEOUT_S}s waiting for "
            f"{num_remote_nodes} remote node(s) to connect"
        )
        time.sleep(NODE_CONNECT_POLL_S)


def _assert_no_worker_failures(result_queue: multiprocessing.Queue) -> None:
    """Fail with the child traceback if either logger node errored."""
    failures = []
    while not result_queue.empty():
        result = result_queue.get_nowait()
        if not result["ok"]:
            failures.append(f"[{result['node']}] {result['traceback']}")
    assert not failures, "Logger node failure(s):\n" + "\n".join(failures)


def _run_policy_inference_multi_node(
    policy: Policy,
    *,
    robot_name: str,
    instance: int,
    joint_positions: dict[str, float],
    frames: dict[str, np.ndarray],
    output_data_types: list[DataType],
) -> None:
    """Predict in this process off data logged by two other processes."""
    context = multiprocessing.get_context("spawn")
    joint_ready = context.Event()
    camera_ready = context.Event()
    stop_event = context.Event()
    result_queue = context.Queue()
    processes: list[multiprocessing.process.BaseProcess] = []

    try:
        # This process contributes nothing, so anything the merged sync point
        # holds later can only have arrived over the peer-to-peer transport.
        local_sync_point = nc.get_latest_sync_point(
            robot_name=robot_name, instance=instance, include_remote=False
        )
        assert not local_sync_point.data, (
            "Expected the predicting process to log nothing, but its local sync "
            f"point already holds: {list(local_sync_point.data)}"
        )

        processes = [
            context.Process(
                target=_joint_node_worker,
                name="joint-node",
                args=(
                    robot_name,
                    instance,
                    joint_positions,
                    joint_ready,
                    stop_event,
                    result_queue,
                ),
            ),
            context.Process(
                target=_camera_node_worker,
                name="camera-node",
                args=(
                    robot_name,
                    instance,
                    frames,
                    camera_ready,
                    stop_event,
                    result_queue,
                ),
            ),
        ]
        for process in processes:
            process.start()

        assert joint_ready.wait(
            timeout=NODE_READY_TIMEOUT_S
        ), "Joint node did not signal readiness in time"
        assert camera_ready.wait(
            timeout=NODE_READY_TIMEOUT_S
        ), "Camera node did not signal readiness in time"
        _assert_no_worker_failures(result_queue)
        logger.info("Both logger nodes are ready")

        _wait_for_remote_nodes(robot_name, instance, num_remote_nodes=2)
        logger.info("Both remote nodes are connected")

        sync_point = nc.get_latest_sync_point(
            robot_name=robot_name, instance=instance, include_remote=True
        )
        assert DataType.JOINT_POSITIONS in sync_point.data, (
            "Joint positions from the joint node not found, got: "
            f"{list(sync_point.data)}"
        )
        assert (
            DataType.RGB_IMAGES in sync_point.data
        ), f"RGB images from the camera node not found, got: {list(sync_point.data)}"
        for name, value in joint_positions.items():
            assert name in sync_point[DataType.JOINT_POSITIONS], (
                f"Joint {name!r} missing from merged sync point, got: "
                f"{sorted(sync_point[DataType.JOINT_POSITIONS])}"
            )
            assert sync_point[DataType.JOINT_POSITIONS][name].value == pytest.approx(
                value
            )
        for name, frame in frames.items():
            assert name in sync_point[DataType.RGB_IMAGES], (
                f"Camera {name!r} missing from merged sync point, got: "
                f"{sorted(sync_point[DataType.RGB_IMAGES])}"
            )
            # Frames cross the wire as lossless PNG data URIs.
            np.testing.assert_array_equal(
                sync_point[DataType.RGB_IMAGES][name].frame, frame
            )
        logger.info("Merged sync point holds both remote nodes' data")

        predictions = policy.predict(timeout=PREDICT_TIMEOUT_S)
        for data_type in output_data_types:
            assert data_type in predictions, (
                f"Expected {data_type.value} in policy output, got: "
                f"{[key.value for key in predictions]}"
            )
            for name, prediction in predictions[data_type].items():
                assert isinstance(
                    prediction, BatchedJointData
                ), f"Expected BatchedJointData for {name!r}, got {type(prediction)}"
                assert (
                    prediction.value.shape[0] == 1
                ), f"Expected a batch of 1 for {name!r}, got {prediction.value.shape}"
        logger.info(
            f"Multi-node inference passed — output keys: "
            f"{[key.value for key in predictions]}"
        )
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=NODE_JOIN_TIMEOUT_S)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        _assert_no_worker_failures(result_queue)
        policy.disconnect()


@pytest.fixture(scope="module")
def cnnmlp_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the CNNMLP archive once and share it across the tests."""
    return _build_cnnmlp_archive(tmp_path_factory.mktemp("cnnmlp_archive"))


@pytest.fixture
def robot_name() -> Generator[str, None, None]:
    """A freshly registered robot per test, deleted on teardown.

    A unique name guarantees the predicting process starts with empty local
    streams and keeps concurrent runs on a shared org from interfering.
    """
    nc.login()
    name = unique_name(prefix=ROBOT_NAME_PREFIX)
    nc.connect_robot(name, instance=ROBOT_INSTANCE, overwrite=False)
    try:
        yield name
    finally:
        delete_cloud_robot(name)


class TestMultiNodePolicyInference:
    """Predict off remote-node data through each locally loaded policy."""

    def test_direct_policy_multi_node_inference(
        self, cnnmlp_archive: Path, robot_name: str
    ) -> None:
        policy = nc.policy(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            model_file=str(cnnmlp_archive),
            device=str(DEVICE),
        )
        _run_policy_inference_multi_node(
            policy,
            robot_name=robot_name,
            instance=ROBOT_INSTANCE,
            joint_positions=_joint_positions(),
            frames=_camera_frames(),
            output_data_types=list(OUTPUT_DATA_TYPES),
        )
        logger.info("[PASSED] Direct policy multi-node inference")

    def test_local_server_policy_multi_node_inference(
        self, cnnmlp_archive: Path, robot_name: str
    ) -> None:
        policy = nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            model_file=str(cnnmlp_archive),
            device=str(DEVICE),
            port=random.randint(10000, 20000),
        )
        _run_policy_inference_multi_node(
            policy,
            robot_name=robot_name,
            instance=ROBOT_INSTANCE,
            joint_positions=_joint_positions(),
            frames=_camera_frames(),
            output_data_types=list(OUTPUT_DATA_TYPES),
        )
        logger.info("[PASSED] Local server policy multi-node inference")
