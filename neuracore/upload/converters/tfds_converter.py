"""
TFDSConverter - Converter for TensorFlow Datasets to Neuracore format.

This module handles the conversion of robot datasets in TensorFlow Datasets
format to the Neuracore platform format.
"""

import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import tensorflow_datasets as tfds

from neuracore.core.robot import Robot
from neuracore.core.utils.depth_utils import MAX_DEPTH
from neuracore.upload.robot_utils import RobotInfo

from ..neura_uploader import DatasetConverter, DatasetInfo

# Configure logging
logger = logging.getLogger("TFDSConverter")

DEFAULT_STATE_KEYS = ["state", "robot_state"]
DEFAULT_LANGUAGE_KEYS = [
    "natural_language_instruction",
    "language_instruction",
    "structured_language_instruction",
]


class TFDSConverter(DatasetConverter):
    """Converter for TensorFlow Datasets (TFDS) robot datasets"""

    def __init__(
        self,
        tfds_name: str,
        robot: Robot,
        dataset_info: DatasetInfo,
        # Has to be defined, as theres no convention for how state is stored
        process_joint_positions_fn: Callable[
            [Robot, Any], tuple[Dict[str, float], Dict[str, float]]
        ],
        process_gripper_open_amounts_fn: Callable[[Robot, Any], Dict[str, float]],
        rgb_keys: Optional[List[str]] = None,
        depth_keys: Optional[List[str]] = None,
        depth_scale: float = 1.0,
        language_key: Optional[str] = None,
        state_key: Optional[str] = None,
        split: str = "all",
        data_dir: str = os.getenv(
            "TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets")
        ),
        # Custom processing hooks
        process_rgb_images_fn: Callable[[Any], Dict[str, np.ndarray]] = None,
        process_depth_images_fn: Callable[[Any], Dict[str, np.ndarray]] = None,
        process_actions_fn: Callable[[Any], Dict[str, float]] = None,
        process_language_fn: Callable[[Any], str] = None,
        # Full override option
        custom_process_fn: Optional[Callable] = None,
    ):
        """
        Initialize TFDS Converter

        Args:
            tfds_name: Name of the TFDS dataset
            robot: Robot information
            dataset_info: Dataset metadata
            process_joint_positions_fn: Function to process joint positions.
                Must take an observation dictionary and return a dictionary
                mapping joint names to joint positions.
            rgb_keys: List of keys for RGB images
            depth_keys: List of keys for depth images
            depth_scale: Scaling factor for depth images
            language_key: Key for language annotations
            state_key: Key for state data
            split: Dataset split to use (default: "train")
            data_dir: Directory to store TFDS data (default: ~/tensorflow_datasets)
            process_rgb_images_fn: Custom function to process RGB images
            process_depth_images_fn: Custom function to process depth images
            process_actions_fn: Custom function to process actions
            process_language_fn: Custom function to process language annotations
            custom_process_fn: Custom function to process entire episode
        """
        self.tfds_name = tfds_name
        self.data_dir = data_dir or os.path.expanduser("~/tensorflow_datasets")
        self.split = split
        self.rgb_keys = rgb_keys or []
        self.depth_keys = depth_keys or []
        self.depth_scale = depth_scale
        self.state_keys = [state_key] if state_key else DEFAULT_STATE_KEYS
        self.language_keys = [language_key] if language_key else DEFAULT_LANGUAGE_KEYS

        # Store custom processing hooks
        self.process_joint_positions_fn = process_joint_positions_fn
        self.process_gripper_open_amounts_fn = process_gripper_open_amounts_fn
        self.process_rgb_images_fn = process_rgb_images_fn
        self.process_depth_images_fn = process_depth_images_fn
        self.process_actions_fn = process_actions_fn
        self.process_language_fn = process_language_fn
        self.custom_process_fn = custom_process_fn

        # Set robot and dataset info
        self._robot = robot
        self._dataset_info = dataset_info

    @property
    def ds(self):
        """Return the loaded TFDS dataset"""
        try:
            tfds.builder(self.tfds_name, data_dir=self.data_dir)
            return tfds.load(self.tfds_name, split=self.split, data_dir=self.data_dir)
        except Exception as e:
            raise RuntimeError(f"Error initializing TFDS dataset {self.tfds_name}: {e}")

    def get_dataset_info(self) -> DatasetInfo:
        """Return dataset metadata"""
        return self._dataset_info

    def get_robot_info(self) -> RobotInfo:
        """Return robot information"""
        return self._robot.robot_info

    def get_episode_count(self) -> int:
        """Return the number of episodes in the dataset"""
        return len(self.ds)

    def get_episode_iterator(
        self, start_idx: int = 0, end_idx: Optional[int] = None
    ) -> Iterator:
        """Return an iterator over episodes in the specified range"""
        ds = self.ds
        end_idx = end_idx or len(ds)
        # Use skip and take to get the slice of episodes
        return iter(ds.skip(start_idx).take(end_idx - start_idx))

    def process_episode(self, episode: Any) -> List[Dict[str, Any]]:
        """Process an episode and return a list of timestep dictionaries"""
        # Use custom process function if provided
        if self.custom_process_fn:
            return self.custom_process_fn(episode)

        try:
            # Check if episode has "steps" field (common in TFDS robot datasets)
            if "steps" in episode:
                steps = episode["steps"]
                return [self._process_step(step) for step in steps]
            else:
                # Treat episode as a single step
                return [self._process_step(episode)]
        except Exception as e:
            logger.error(f"Error processing episode: {e}")
            return []

    def _process_step(self, step: Any) -> Dict[str, Any]:
        """Process a single step and extract relevant data"""
        result = {}

        # Get observation data (might be the step itself or in an "observation" field)
        observation = step.get("observation", step)

        # Process joint positions from state
        jp, additional_jp = self._extract_joint_positions(observation)
        result["joint_positions"] = jp
        result["additional_joint_positions"] = additional_jp

        # Process gripper open amounts
        result["gripper_open_amounts"] = self._extract_grpper_open_amounts(observation)

        # Process RGB images
        result["rgb_images"] = self._extract_rgb_images(observation)

        # Process depth images
        result["depth_images"] = self._extract_depth_images(observation)

        # Process language annotations
        language = self._extract_language_annotation(observation)
        if language:
            result["language_annotation"] = language

        return result

    def _extract_joint_positions(
        self, observation: Any
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Extract joint positions from state data"""
        # Use custom processor if provided
        return self.process_joint_positions_fn(self._robot, observation)

    def _extract_grpper_open_amounts(self, observation: Any) -> dict[str, float]:
        """Extract gripper open amounts from state data"""
        return self.process_gripper_open_amounts_fn(self._robot, observation)

    def _extract_rgb_images(self, observation: Any) -> Dict[str, np.ndarray]:
        """Extract RGB images from observation data"""
        # Use custom processor if provided
        if self.process_rgb_images_fn:
            rgb_images = self.process_rgb_images_fn(observation)
            if rgb_images:
                return rgb_images

        rgb_images = {}
        for tfds_key in self.rgb_keys:
            if tfds_key in observation:
                img_data = observation[tfds_key]
                if hasattr(img_data, "numpy"):
                    img_data = img_data.numpy()
                # Check if this looks like an RGB image (HxWx3)
                if len(img_data.shape) == 3 and img_data.shape[-1] == 3:
                    # Ensure image is uint8
                    if img_data.dtype != np.uint8:
                        if img_data.max() <= 1.0:
                            img_data = (img_data * 255).astype(np.uint8)
                        else:
                            img_data = img_data.astype(np.uint8)
                    rgb_images[tfds_key] = img_data
        return rgb_images

    def _extract_depth_images(self, observation: Any) -> Dict[str, np.ndarray]:
        """Extract depth images from observation data"""
        # Use custom processor if provided
        if self.process_depth_images_fn:
            depth_images = self.process_depth_images_fn(observation)
            if depth_images:
                return depth_images

        depth_images = {}
        for tfds_key in self.depth_keys:
            if tfds_key in observation:
                depth_data = observation[tfds_key]
                if hasattr(depth_data, "numpy"):
                    depth_data = depth_data.numpy()

                # Process depth data based on shape and type
                if len(depth_data.shape) == 2 or (
                    len(depth_data.shape) == 3 and depth_data.shape[-1] == 1
                ):
                    # Ensure 2D
                    if len(depth_data.shape) == 3:
                        depth_data = depth_data.squeeze(-1)

                    # Apply scaling factor
                    depth_data = depth_data.astype(np.float32) * self.depth_scale
                    if depth_data.max() > MAX_DEPTH:
                        depth_data = np.clip(depth_data, 0, MAX_DEPTH)
                    depth_images[tfds_key] = depth_data
        return depth_images

    def _extract_language_annotation(self, observation: Any) -> Optional[str]:
        """Extract language annotation from various possible keys"""
        # Use custom processor if provided
        if self.process_language_fn:
            language = self.process_language_fn(observation)
            if language:
                return language

        # Default processing logic
        for key in self.language_keys:
            value = observation.get(key, None)
            if value is not None:
                return value
        return None
