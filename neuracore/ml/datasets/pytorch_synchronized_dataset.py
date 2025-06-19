"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np
import torch
from PIL import Image

import neuracore as nc
from neuracore.core.data.cache_manager import CacheManager
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.nc_types import DataType, JointData, SyncPoint
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


# Single training sample is identical type, but with no batch dimension
TrainingSample = BatchedTrainingSamples

CHECK_MEMORY_INTERVAL = 100


class PytorchSynchronizedDataset(PytorchNeuracoreDataset):
    """Dataset for loading episodic robot data from GCS with filesystem caching."""

    def __init__(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int,
        cache_dir: Optional[str] = None,
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """Initialize the dataset.

        Args:
            synchronized_dataset: The synchronized dataset to load data from.
            input_data_types: List of input data types to include in the dataset.
            output_data_types: List of output data types to include in the dataset.
            output_prediction_horizon: Number of future timesteps to predict.
            cache_dir: Directory to use for caching data.
            tokenize_text: Optional function to tokenize text data.
        """
        if not isinstance(synchronized_dataset, SynchronizedDataset):
            raise TypeError(
                "synchronized_dataset must be an instance of SynchronizedDataset"
            )
        super().__init__(
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
            tokenize_text=tokenize_text,
            num_recordings=len(synchronized_dataset),
        )
        self.synchronized_dataset = synchronized_dataset
        self.dataset_description = self.synchronized_dataset.dataset_description

        # Setup cache
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "episodic_dataset_cache")
        self.cache_dir = Path(cache_dir)
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)  # clear cache directory if it exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = CacheManager(
            self.cache_dir,
        )

        self._max_error_count = 100
        self._error_count = 0
        self._memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=1.0, gpu_id=None
        )
        self._mem_check_counter = 0
        self._num_samples = self.synchronized_dataset.num_transitions
        self._logged_in = False

    @staticmethod
    def _get_timestep(episode_length: int) -> int:
        max_start = max(0, episode_length)
        return np.random.randint(0, max_start - 1)

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> BatchedTrainingSamples:
        """Load image from cache or GCS."""
        if not self._logged_in:
            # Ensure we only log in once per dataset instance
            nc.login()
            self._logged_in = True
            # Dataloaders already run in parallel, so set to 0
            os.environ["NEURACORE_NUM_PARALLEL_VIDEO_DOWNLOADS"] = "0"

        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1
        try:
            synced_recording = self.synchronized_dataset[episode_idx]
            synced_recording = cast(SynchronizedRecording, synced_recording)
            episode_length = len(synced_recording)
            timestep = timestep or self._get_timestep(episode_length)
            tensor_cache_path = self.cache_dir / f"ep_{episode_idx}_frame_{timestep}.pt"
            if tensor_cache_path.exists():
                return torch.load(tensor_cache_path, weights_only=False)
            else:
                # Check disk space periodically (based on check_interval)
                if not self.cache_manager.ensure_space_available():
                    logger.warning("Low disk space. Some cache files were removed.")

                sample = TrainingSample(
                    output_predicition_mask=torch.ones(
                        (self.output_prediction_horizon,), dtype=torch.float32
                    ),
                )
                sync_point = cast(SyncPoint, synced_recording[timestep])
                future_sync_points = cast(
                    list[SyncPoint],
                    synced_recording[
                        timestep + 1 : timestep + 1 + self.output_prediction_horizon
                    ],
                )
                # Padding for future sync points
                for _ in range(
                    self.output_prediction_horizon - len(future_sync_points)
                ):
                    future_sync_points.append(future_sync_points[-1])

                if sync_point.rgb_images:
                    if DataType.RGB_IMAGE in self.input_data_types:
                        rgbs_for_each_camera: list[Image.Image] = list(
                            [sp.frame for sp in sync_point.rgb_images.values()]
                        )
                        sample.inputs.rgb_images = (
                            self._create_camera_maskable_input_data(
                                rgbs_for_each_camera
                            )
                        )
                    if DataType.RGB_IMAGE in self.output_data_types:
                        future_frames = [
                            [cam_data.frame for cam_data in sp.rgb_images.values()]
                            for sp in future_sync_points
                            if sp.rgb_images is not None
                        ]
                        sample.outputs.rgb_images = (
                            self._create_camera_maskable_output_data(future_frames)
                        )

                if sync_point.depth_images:
                    if DataType.DEPTH_IMAGE in self.input_data_types:
                        depth_for_each_camera: list[Image.Image] = list(
                            [sp.frame for sp in sync_point.depth_images.values()]
                        )
                        sample.inputs.depth_images = (
                            self._create_camera_maskable_input_data(
                                depth_for_each_camera
                            )
                        )
                    if DataType.DEPTH_IMAGE in self.output_data_types:
                        future_frames = [
                            [cam_data.frame for cam_data in sp.depth_images.values()]
                            for sp in future_sync_points
                            if sp.depth_images is not None
                        ]
                        sample.outputs.depth_images = (
                            self._create_camera_maskable_output_data(future_frames)
                        )
                if sync_point.joint_positions:
                    if DataType.JOINT_POSITIONS in self.input_data_types:
                        sample.inputs.joint_positions = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_positions,
                                self.dataset_description.joint_positions.max_len,
                            )
                        )

                    if DataType.JOINT_POSITIONS in self.output_data_types:
                        sample.outputs.joint_positions = (
                            self._create_joint_maskable_output_data(
                                [
                                    sp.joint_positions
                                    for sp in future_sync_points
                                    if sp.joint_positions is not None
                                ],
                                self.dataset_description.joint_positions.max_len,
                            )
                        )

                if sync_point.joint_velocities:
                    if DataType.JOINT_VELOCITIES in self.input_data_types:
                        sample.inputs.joint_velocities = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_velocities,
                                self.dataset_description.joint_velocities.max_len,
                            )
                        )
                    if DataType.JOINT_VELOCITIES in self.output_data_types:
                        sample.outputs.joint_velocities = (
                            self._create_joint_maskable_output_data(
                                [
                                    sp.joint_velocities
                                    for sp in future_sync_points
                                    if sp.joint_velocities is not None
                                ],
                                self.dataset_description.joint_velocities.max_len,
                            )
                        )

                if sync_point.joint_torques:
                    if DataType.JOINT_TORQUES in self.input_data_types:
                        sample.inputs.joint_torques = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_torques,
                                self.dataset_description.joint_torques.max_len,
                            )
                        )
                    if DataType.JOINT_TORQUES in self.output_data_types:
                        sample.outputs.joint_torques = (
                            self._create_joint_maskable_output_data(
                                [
                                    sp.joint_torques
                                    for sp in future_sync_points
                                    if sp.joint_torques is not None
                                ],
                                self.dataset_description.joint_torques.max_len,
                            )
                        )

                if sync_point.joint_target_positions:
                    if DataType.JOINT_TARGET_POSITIONS in self.input_data_types:
                        sample.inputs.joint_target_positions = (
                            self._create_joint_maskable_input_data(
                                sync_point.joint_target_positions,
                                self.dataset_description.joint_target_positions.max_len,
                            )
                        )
                    if DataType.JOINT_TARGET_POSITIONS in self.output_data_types:
                        # We dont need to shift the sync_point by 1, since we are
                        # using the target joint positions as the action
                        jtp_points = cast(
                            list[SyncPoint],
                            synced_recording[
                                timestep : timestep + self.output_prediction_horizon
                            ],
                        )
                        for _ in range(
                            self.output_prediction_horizon - len(jtp_points)
                        ):
                            jtp_points.append(jtp_points[-1])
                        sample.outputs.joint_target_positions = (
                            self._create_joint_maskable_output_data(
                                [
                                    sp.joint_target_positions
                                    for sp in jtp_points
                                    if sp.joint_target_positions is not None
                                ],
                                self.dataset_description.joint_target_positions.max_len,
                            )
                        )

                if sync_point.language_data:
                    if self.tokenize_text is None:
                        raise ValueError(
                            "Failed to initialize tokenize_text for DataType.LANGUAGE"
                        )
                    input_ids, attention_mask = self.tokenize_text(
                        [sync_point.language_data.text]
                    )

                    language_tokens = MaskableData(input_ids, attention_mask)
                    if DataType.LANGUAGE in self.input_data_types:
                        sample.inputs.language_tokens = language_tokens
                    if DataType.LANGUAGE in self.output_data_types:
                        sample.outputs.language_tokens = language_tokens

                sample.output_predicition_mask = self._create_output_prediction_mask(
                    episode_length,
                    timestep,
                    self.output_prediction_horizon,
                )

                torch.save(sample, tensor_cache_path)

            return sample

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(
                f"Error loading frame {timestep} from episode {episode_idx}: {str(e)}"
            )
            raise e

    def _create_joint_maskable_input_data(
        self, joint_data: JointData, max_len: int
    ) -> MaskableData:
        jdata = torch.tensor(list(joint_data.values.values()), dtype=torch.float32)
        num_existing_states = jdata.shape[0]
        extra_states = max_len - num_existing_states
        if extra_states > 0:
            jdata = torch.cat(
                [jdata, torch.zeros(extra_states, dtype=torch.float32)], dim=0
            )
        jdata_mask = torch.tensor(
            [1.0] * num_existing_states + [0.0] * extra_states, dtype=torch.float32
        )
        return MaskableData(jdata, jdata_mask)

    def _create_joint_maskable_output_data(
        self, joint_data: list[JointData], max_len: int
    ) -> MaskableData:
        maskable_data_for_each_t = [
            self._create_joint_maskable_input_data(jd, max_len) for jd in joint_data
        ]
        stacked_maskable_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_maskable_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_maskable_data, stacked_maskable_mask)

    def _create_output_prediction_mask(
        self, episode_length: int, timestep: int, output_prediction_horizon: int
    ) -> torch.FloatTensor:
        output_prediction_mask = torch.zeros(
            output_prediction_horizon, dtype=torch.float32
        )
        for i in range(output_prediction_horizon):
            if timestep + i >= episode_length:
                break
            else:
                output_prediction_mask[i] = 1.0
        return output_prediction_mask

    def _create_camera_maskable_input_data(
        self, camera_data: list[Image.Image]
    ) -> MaskableData:
        # Want to create tensors of shape [CAMS, C, H, W]
        cam_image_tensors = torch.stack(
            [self.camera_transform(cam_data) for cam_data in camera_data]
        )
        num_cameras = cam_image_tensors.shape[0]
        extra_cameras = self.dataset_description.max_num_rgb_images - num_cameras
        if extra_cameras > 0:
            empty_image = torch.zeros_like(cam_image_tensors[0])
            cam_image_tensors = torch.cat(
                [cam_image_tensors, empty_image.repeat(extra_cameras, 1, 1, 1)],
                dim=0,
            )
        camera_images_mask = torch.tensor(
            [1.0] * num_cameras + [0.0] * extra_cameras,
            dtype=torch.float32,
        )
        return MaskableData(cam_image_tensors, camera_images_mask)

    def _create_camera_maskable_output_data(
        self, temporal_camera_data: list[list[Image.Image]]
    ) -> MaskableData:
        """Create maskable data for multiple cameras.

        Args:
            temporal_camera_data: A list of lists of shape [T, CAMS, ...].

        Returns:
            MaskableData: A MaskableData object containing the stacked camera images
                and their masks of shape [T, CAMS, C, H, W].
        """
        maskable_data_for_each_t = [
            self._create_camera_maskable_input_data(camera_data)
            for camera_data in temporal_camera_data
        ]
        stacked_maskable_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_maskable_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_maskable_data, stacked_maskable_mask)

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return self._num_samples
