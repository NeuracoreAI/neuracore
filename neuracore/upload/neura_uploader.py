"""
NeuraUploader - Generic interface for uploading datasets to Neuracore.

This module provides a flexible, efficient, multiprocessing-enabled uploader
for converting various dataset formats to Neuracore's format.
"""

import logging
import multiprocessing
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

import neuracore as nc
from neuracore.upload.robot_utils import RobotInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NeuraUploader")


class DatasetFormat(Enum):
    """Supported dataset formats"""

    TFDS = "tensorflow_dataset"
    LEROBOT = "huggingface_lerobot"


@dataclass
class DatasetInfo:
    """Dataset metadata"""

    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    frequency: float = 10.0  # Hz
    visual_joint_names: List[str] = field(default_factory=list)


class DatasetConverter(ABC):
    """Abstract base class for dataset format converters"""

    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo:
        """Return dataset metadata"""
        pass

    @abstractmethod
    def get_robot_info(self) -> RobotInfo:
        """Return robot information"""
        pass

    @abstractmethod
    def get_episode_count(self) -> int:
        """Return the number of episodes in the dataset"""
        pass

    @abstractmethod
    def get_episode_iterator(
        self, start_idx: int = 0, end_idx: Optional[int] = None
    ) -> Iterator:
        """Return an iterator over episodes in the given range"""
        pass

    @abstractmethod
    def process_episode(self, episode: Any) -> List[Dict[str, Any]]:
        """
        Process an episode and return a list of timestep dictionaries

        Each timestep should contain:
        - rgb_images: Dict[str, np.ndarray] - camera_id -> RGB image
        - depth_images: Dict[str, np.ndarray] - camera_id -> depth
        - joint_positions: Dict[str, float] - joint_name -> position in radians
        - actions: Dict[str, float] or np.ndarray - actions
        - language_annotation: Optional[str] - language annotation
        - timestamp: Optional[float] - timestamp (will be generated if not provided)
        """
        pass


class NeuraUploader:
    """Main uploader interface for Neuracore"""

    def __init__(
        self,
        converter: DatasetConverter,
        max_workers: Optional[int] = None,
        chunk_size: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
        verbose: bool = True,
    ):
        """
        Initialize the uploader

        Args:
            converter: Dataset converter implementation
            max_workers: Maximum number of worker processes (defaults to CPU count - 1)
            chunk_size: Number of episodes to process in each worker
            retry_attempts: Number of upload retry attempts
            retry_delay: Delay between retry attempts in seconds
            verbose: Whether to show progress bars and detailed logging
        """
        self.converter = converter
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.verbose = verbose

        # Will be set during upload process
        self.dataset_id = None
        self.robot_name = None
        self._dataset_info = None

    def upload(
        self,
        episodes_range: Optional[Tuple[int, int]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """
        Upload the dataset to Neuracore

        Args:
            episodes_range: Optional tuple of (start_idx, end_idx) to upload a subset
            progress_callback: Optional callback function for progress updates

        Returns:
            Dataset ID
        """
        logger.info("Starting dataset upload process")

        # Initialize API connection
        self._initialize_api()

        # Get dataset and robot information
        self._dataset_info = self.converter.get_dataset_info()
        robot_info = self.converter.get_robot_info()

        # Register robot and create dataset
        self._register_robot(robot_info)
        self._get_or_create_dataset(self._dataset_info)

        # Determine episode range
        total_episodes = self.converter.get_episode_count()
        start_idx = 0
        end_idx = total_episodes

        if episodes_range:
            start_idx, end_idx = episodes_range
            end_idx = min(end_idx, total_episodes)

        # Create episode chunks for multiprocessing
        episode_chunks = []
        for chunk_start in range(start_idx, end_idx, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, end_idx)
            episode_chunks.append((chunk_start, chunk_end))

        # Create multiprocessing context
        ctx = multiprocessing.get_context("spawn")

        # Process episode chunks in parallel
        successful_chunks = 0
        total_chunks = len(episode_chunks)

        with tqdm(
            total=total_chunks, desc="Uploading episodes", disable=not self.verbose
        ) as pbar:
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=ctx
            ) as executor:
                # Submit all chunks for processing
                futures = [
                    executor.submit(
                        self._process_episode_chunk,
                        chunk_start,
                        chunk_end,
                        self._dataset_info.frequency,
                    )
                    for chunk_start, chunk_end in episode_chunks
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            successful_chunks += 1

                        if progress_callback:
                            progress_callback(successful_chunks / total_chunks)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        logger.error(traceback.format_exc())

                    pbar.update(1)

        success_rate = successful_chunks / total_chunks if total_chunks > 0 else 0
        logger.info(
            "Upload completed. "
            f"{successful_chunks}/{total_chunks} chunks successful ({success_rate:.1%})"
        )
        return self.dataset_id

    def _initialize_api(self):
        """Initialize Neuracore API with authentication"""
        logger.info("Initializing Neuracore API")
        nc.login()

    def _register_robot(self, robot_info: RobotInfo):
        """Register robot with Neuracore"""
        logger.info(f"Getting robot: {robot_info.name}")
        self.robot_name = robot_info.name

        # Connect robot using API
        nc.connect_robot(
            robot_name=robot_info.name,
            urdf_path=robot_info.urdf_path,
            mjcf_path=robot_info.mjcf_path,
            overwrite=False,
            shared=True,
        )

    def _get_or_create_dataset(self, dataset_info: DatasetInfo):
        """Create dataset in Neuracore"""
        logger.info(f"Getting or creating dataset: {dataset_info.name}")
        dataset = nc.create_dataset(
            name=dataset_info.name,
            description=dataset_info.description,
            tags=dataset_info.tags,
            shared=True,
        )
        self.dataset_id = dataset.id
        logger.info(f"Got dataset with ID: {self.dataset_id}")

    def _process_episode_chunk(
        self, chunk_start: int, chunk_end: int, frequency: float
    ) -> bool:
        """
        Process and upload a chunk of episodes

        Args:
            chunk_start: Starting episode index
            chunk_end: Ending episode index (exclusive)
            frequency: Dataset frequency in Hz

        Returns:
            True if successful, False otherwise
        """
        logger.debug(f"Processing episodes {chunk_start}-{chunk_end}")

        try:
            # Reinitialize API in the worker process
            nc.login()
            nc.connect_robot(self.robot_name, shared=True)
            nc.get_dataset(self._dataset_info.name)

            # Get episode iterator for this chunk
            episode_iterator = self.converter.get_episode_iterator(
                chunk_start, chunk_end
            )

            for episode_idx, episode in enumerate(episode_iterator, chunk_start):
                logger.info(f"Processing episode {episode_idx}")
                self._upload_episode(episode_idx, episode, frequency)

            return True

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _upload_episode(self, episode_idx: int, episode: Any, frequency: float) -> bool:
        """
        Process and upload a single episode

        Args:
            episode_idx: Episode index for logging
            episode: Episode data from the converter
            frequency: Dataset frequency in Hz

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.retry_attempts):
            try:
                # Process episode data
                logger.info(f"Processing episode {episode_idx}")
                timestep_data = self.converter.process_episode(episode)

                if not timestep_data:
                    logger.warning(f"Empty episode {episode_idx}, skipping")
                    return False

                # Start recording
                nc.start_recording()

                # Upload each timestep
                base_time = time.time()
                for idx, timestep in enumerate(timestep_data):
                    # Calculate timestamp if not provided
                    timestamp = timestep.get("timestamp", base_time + (idx / frequency))

                    # Log joint positions
                    if "joint_positions" in timestep and timestep["joint_positions"]:
                        joint_data = timestep["joint_positions"]
                        additional_urdf_positions = timestep[
                            "additional_joint_positions"
                        ]
                        nc.log_joint_positions(
                            joint_data,
                            additional_urdf_positions=additional_urdf_positions,
                            timestamp=timestamp,
                        )

                    if "gripper_open_amounts" in timestep:
                        nc.log_gripper_data(
                            timestep["gripper_open_amounts"], timestamp=timestamp
                        )

                    # Log actions
                    if "actions" in timestep:
                        nc.log_action(timestep["actions"], timestamp=timestamp)

                    # Log language annotations
                    if (
                        "language_annotation" in timestep
                        and timestep["language_annotation"]
                    ):
                        nc.log_language(
                            timestep["language_annotation"], timestamp=timestamp
                        )

                    # Log RGB images
                    for camera_id, image in timestep.get("rgb_images", {}).items():
                        nc.log_rgb(camera_id, image, timestamp=timestamp)

                    # Log depth images
                    for camera_id, depth in timestep.get("depth_images", {}).items():
                        nc.log_depth(camera_id, depth, timestamp=timestamp)

                # Stop recording
                nc.stop_recording()
                logger.debug(f"Successfully uploaded episode {episode_idx}")
                return True

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(
                        f"Upload attempt {attempt+1} failed for "
                        f"episode {episode_idx}: {e}, retrying in {self.retry_delay}s"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed to upload episode {episode_idx} after "
                        f"{self.retry_attempts} attempts: {e}"
                    )
                    return False
