"""Shared importer framework for dataset ingestion workflows."""

from __future__ import annotations

import inspect
import logging
import multiprocessing as mp
import os
import random
import re
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
from neuracore_types.importer.config import (
    ActionSpaceConfig,
    ActionTypeConfig,
    EndEffectorPoseInputTypeConfig,
    JointPositionInputTypeConfig,
)
from neuracore_types.importer.data_config import DataFormat
from neuracore_types.nc_data import DatasetImportConfig, DataType, NCDataImportConfig
from neuracore_types.nc_data.nc_data import MappingItem
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import neuracore as nc
from neuracore.core.robot import JointInfo
from neuracore.data_daemon.const import DEFAULT_RECORDING_ROOT_PATH
from neuracore.importer.core.robot_utils import RobotUtils
from neuracore.importer.core.validation import (
    JOINT_DATA_TYPES,
    validate_depth_images,
    validate_joint_positions,
    validate_joint_torques,
    validate_joint_velocities,
    validate_language,
    validate_point_clouds,
    validate_poses,
    validate_rgb_images,
)

from .exceptions import DataValidationError, DataValidationWarning, ImporterError

JOINT_TARGET_CHECK_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ImportItem:
    """Unit of import work (typically one episode)."""

    index: int
    split: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerError:
    """Captured failure from a worker process."""

    worker_id: int
    item_index: int | None
    message: str
    traceback: str | None = None


@dataclass(frozen=True)
class ProgressUpdate:
    """Progress event emitted from workers to update the TUI."""

    worker_id: int
    item_index: int
    step: int
    total_steps: int | None
    episode_label: str | None = None


_RICH_CONSOLE = Console(stderr=True, force_terminal=True)


def get_shared_console() -> Console:
    """Return the shared console used by logging and progress bars."""
    return _RICH_CONSOLE


class NeuracoreDatasetImporter(ABC):
    """Importer workflow that manages workers and Neuracore session setup."""

    DISK_CHECK_INTERVAL_SECS: float = 10.0

    def __init__(
        self,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        output_dataset_name: str,
        max_workers: int | None = 1,
        min_workers: int = 1,
        skip_on_error: str = "episode",
        progress_interval: int = 1,
        joint_info: dict[str, JointInfo] = {},
        urdf_path: str | None = None,
        ik_init_config: list[float] | None = None,
        dry_run: bool = False,
        suppress_warnings: bool = False,
        storage_limit: int = 5 * 1024**3,
        random_sample: int | None = None,
        shared: bool = False,
    ) -> None:
        """Initialize the base dataset importer.

        Args:
            dataset_dir: Root directory of the source dataset.
            dataset_config: Dataset configuration (robot, frequency, etc.).
            output_dataset_name: Name of the dataset to create.
            max_workers: Maximum number of worker processes; None for auto.
            min_workers: Minimum number of workers to use.
            skip_on_error: "episode" to skip a failed episode; "step" to skip only
                failing steps; "all" to abort on the first error.
            progress_interval: Emit progress updates every this many items (>= 1).
            joint_info: Joint name -> JointInfo for validation.
            urdf_path: Optional URDF path for robot utilities.
            ik_init_config: Optional initial joint configuration for IK.
            dry_run: If True, skip actual recording (validation only).
            suppress_warnings: If True, suppress warning messages.
            storage_limit: Pause workers when disk usage (used bytes)
                on the recording filesystem reaches this value (bytes).
            random_sample: If set, import only this many items chosen at random.
            shared: Whether the dataset should be shared/open-source.
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_config = dataset_config
        self.ordered_import_configs = self._get_ordered_import_configs()
        self.data_config = dataset_config  # Backwards-compat alias used by callers
        self.output_dataset_name = output_dataset_name
        self.robot_name = dataset_config.robot.name
        self.frequency = dataset_config.frequency
        self.joint_info = joint_info
        self.urdf_path = urdf_path
        self.ik_init_config = ik_init_config
        self.robot_utils: RobotUtils | None = None
        self.prev_ik_solution: list[float] | None = None
        self.curr_joint_positions: dict[str, float] = {}
        self.curr_end_effector_poses: dict[str, list[float]] = {}
        self.shared = shared
        if skip_on_error not in {"episode", "step", "all"}:
            raise ValueError("skip_on_error must be one of: 'episode', 'step', 'all'")

        self.max_workers = max_workers
        self.min_workers = min_workers
        self.skip_on_error = skip_on_error  # one of: "episode", "step", "all"
        self.progress_interval = max(1, progress_interval)
        self.dry_run = dry_run
        self.pre_check = False
        self.pre_check_joint_positions: dict[str, list[float]] = {}
        self.pre_check_joint_target_positions: dict[str, list[float]] = {}
        self.suppress_warnings = suppress_warnings
        self.storage_limit = storage_limit
        self.random_sample = random_sample
        self.worker_errors: list[WorkerError] = []
        self._logged_error_keys: set[tuple[int | None, int | None, str]] = set()
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._progress_queue: mp.Queue[ProgressUpdate] | None = None
        self._worker_id: int = -1
        self._error_queue: mp.Queue[WorkerError] | None = None
        self.completed_items_file = self._resolve_completed_items_file()

    @abstractmethod
    def build_work_items(self) -> Sequence[ImportItem]:
        """Enumerate importable units in deterministic order."""

    @abstractmethod
    def import_item(self, item: ImportItem) -> None:
        """Perform the dataset-specific import for a single item."""

    @abstractmethod
    def _record_step(self, step: dict, timestamp: float) -> None:
        """Record a single step of the dataset."""

    def _reset_episode_state(self) -> None:
        """Reset episode-specific state at the start of each episode."""
        self.prev_ik_solution = self.ik_init_config

    def _reset_step_state(self) -> None:
        """Reset step-specific state at the start of each step."""
        self.curr_joint_positions = {}
        self.curr_end_effector_poses = {}

    def _get_ordered_import_configs(self) -> list[tuple[DataType, NCDataImportConfig]]:
        """Get the ordered import configurations for the dataset.

        Process joint positions and end effector poses first in case ik or fk
        is requested or if relative action is requested.

        Returns:
            List of tuples of (DataType, NCDataImportConfig) in the order
            they should be processed.
        """
        data_import_config = self.dataset_config.data_import_config
        joint_position_config = data_import_config.get(DataType.JOINT_POSITIONS, None)
        end_effector_pose_config = data_import_config.get(
            DataType.END_EFFECTOR_POSES, None
        )

        if end_effector_pose_config is None:
            fk_requested = False
        else:
            fk_requested = (
                end_effector_pose_config.format.ee_pose_input_type
                == EndEffectorPoseInputTypeConfig.JOINT_POSITIONS
            )
        if joint_position_config is None:
            ik_requested = False
        else:
            ik_requested = (
                joint_position_config.format.joint_position_input_type
                == JointPositionInputTypeConfig.END_EFFECTOR
            )

        ordered_items = list(data_import_config.items())
        if fk_requested and ik_requested:
            raise ImportError("Cannot request both FK and IK at the same time.")
        elif fk_requested:
            ordered_items = sorted(
                list(data_import_config.items()),
                key=lambda x: (
                    x[0] != DataType.JOINT_POSITIONS,
                    x[0] != DataType.END_EFFECTOR_POSES,
                    x[0].value,
                ),
            )
        elif ik_requested:
            ordered_items = sorted(
                list(data_import_config.items()),
                key=lambda x: (
                    x[0] != DataType.END_EFFECTOR_POSES,
                    x[0] != DataType.JOINT_POSITIONS,
                    x[0].value,
                ),
            )
        return ordered_items

    def _save_pre_check_data(self, data_type: DataType, name: str, data: float) -> None:
        """Save pre-check data for a joint.

        Pre-check data is used to determine if joint target positions should be
        skipped. Joint target positions are skipped if they match joint positions
        in the next step.

        Args:
            data_type: The type of data to save.
            name: The name of the joint or end-effector.
            data: The data to save.
        """
        if data_type == DataType.JOINT_POSITIONS:
            if name not in self.pre_check_joint_positions:
                self.pre_check_joint_positions[name] = []
            self.pre_check_joint_positions[name].append(data)
        elif data_type == DataType.JOINT_TARGET_POSITIONS:
            if name not in self.pre_check_joint_target_positions:
                self.pre_check_joint_target_positions[name] = []
            self.pre_check_joint_target_positions[name].append(data)

    def _determine_skip_joint_target_positions(self) -> bool:
        """Determine if joint target positions should be skipped.

        If joint target positions being imported are equivalent to joint
        positions in the next step, skip importing joint target positions
        as they are not the actual target commands issued to the robot.

        Args:
            None
        Returns:
            True if joint target positions should be skipped, False otherwise.
        """
        skip_joint_target_positions = True
        if DataType.JOINT_TARGET_POSITIONS in [
            config[0] for config in self.ordered_import_configs
        ]:
            assert len(self.pre_check_joint_positions) == len(
                self.pre_check_joint_target_positions
            )
            for name, positions in self.pre_check_joint_positions.items():
                targets = self.pre_check_joint_target_positions[name]
                assert len(positions) == len(targets)
                # Align targets in the current step with positions in the next step.
                positions.pop(0)
                targets.pop(-1)
                if not (
                    np.allclose(positions, targets, atol=JOINT_TARGET_CHECK_TOLERANCE)
                ):
                    skip_joint_target_positions = False
                    break
        return skip_joint_target_positions

    def _validate_input_data(
        self, data_type: DataType, data: Any, format: DataFormat
    ) -> None:
        """Validate input data based on the data type and format.

        Args:
            data_type: The type of data to validate.
            data: The data to validate.
            format: The data format configuration.

        Raises:
            DataValidationError: If the data does not match the expected format.
        """
        if data_type == DataType.RGB_IMAGES:
            validate_rgb_images(data, format)
        elif data_type == DataType.DEPTH_IMAGES:
            validate_depth_images(data)
        elif data_type == DataType.POINT_CLOUDS:
            validate_point_clouds(data)
        elif data_type == DataType.LANGUAGE:
            validate_language(data, format)
        elif data_type == DataType.POSES or data_type == DataType.END_EFFECTOR_POSES:
            validate_poses(data, format)

    def _validate_joint_data(self, data_type: DataType, data: Any, name: str) -> None:
        """Validate joint data based on the data type and joint name.

        Args:
            data_type: The type of data to validate.
            data: The data to validate.
            name: The name of the joint.

        Raises:
            DataValidationError: If the data does not match the expected format.
        """
        if data_type == DataType.JOINT_POSITIONS:
            validate_joint_positions(data, name, self.joint_info)
        elif data_type == DataType.JOINT_VELOCITIES:
            validate_joint_velocities(data, name, self.joint_info)
        elif data_type == DataType.JOINT_TORQUES:
            validate_joint_torques(data, name, self.joint_info)
        elif data_type == DataType.JOINT_TARGET_POSITIONS:
            validate_joint_positions(data, name, self.joint_info)
        elif data_type == DataType.VISUAL_JOINT_POSITIONS:
            validate_joint_positions(data, name, self.joint_info)

    def _log_data(
        self,
        data_type: DataType,
        source_data: Any,
        item: MappingItem,
        format: DataFormat,
        timestamp: float,
        *,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> None:
        """Log a single data point to Neuracore.

        This method validates the source data, transforms it if necessary,
        and logs it to Neuracore. Transformed joint data is validated
        against the joint limits.

        Args:
            data_type: The type of data to import.
            source_data: The source data from the dataset.
            item: The mapping item to use for naming and transformation.
            format: The data format to use for validation.
            timestamp: Time when the data was logged.
            extrinsics: Optional 4x4 camera extrinsics matrix for camera streams.
            intrinsics: Optional 3x3 camera intrinsics matrix for camera streams.
        """
        ik_requested = (
            data_type == DataType.JOINT_POSITIONS
            and format.joint_position_input_type
            == JointPositionInputTypeConfig.END_EFFECTOR
        )
        fk_requested = (
            data_type == DataType.END_EFFECTOR_POSES
            and format.ee_pose_input_type
            == EndEffectorPoseInputTypeConfig.JOINT_POSITIONS
        )
        relative_action_requested = (
            data_type == DataType.JOINT_TARGET_POSITIONS
            and format.action_type == ActionTypeConfig.RELATIVE
        )
        absolute_action_requested = (
            data_type == DataType.JOINT_TARGET_POSITIONS
            and format.action_type == ActionTypeConfig.ABSOLUTE
        )
        try:
            if absolute_action_requested:
                if format.action_space == ActionSpaceConfig.END_EFFECTOR:
                    self._validate_input_data(
                        DataType.END_EFFECTOR_POSES, source_data, format
                    )
                elif format.action_space == ActionSpaceConfig.JOINT:
                    self._validate_input_data(data_type, source_data, format)
            elif not (ik_requested or fk_requested):
                self._validate_input_data(data_type, source_data, format)
        except DataValidationWarning as w:
            if not self.suppress_warnings:
                self.logger.warning("[WARNING] %s (%s): %s", data_type, item.name, w)
        except DataValidationError as e:
            self.logger.error(
                "[ERROR] %s (%s): %s -- skipping episode", data_type, item.name, e
            )
            raise

        try:
            if not (ik_requested or fk_requested):
                transformed_data = item.transforms(source_data)

            if ik_requested:
                if self.robot_utils is None:
                    raise ImporterError(
                        "Failed to convert end effector pose to joint positions: "
                        "Robot utilities are not initialized"
                    )
                transformed_data = self.robot_utils.end_effector_to_joint_positions(
                    self.curr_end_effector_poses[item.source_name],
                    item.name,
                    self.prev_ik_solution,
                )
                self.prev_ik_solution = list(transformed_data.values())
                for name, position in transformed_data.items():
                    self._validate_joint_data(data_type, position, name)
            elif fk_requested:
                if self.robot_utils is None:
                    raise ImporterError(
                        "Failed to convert joint positions to end effector pose: "
                        "Robot utilities are not initialized"
                    )
                transformed_data = (
                    self.robot_utils.joint_positions_to_end_effector_pose(
                        self.curr_joint_positions, item.name
                    )
                )
            elif absolute_action_requested:
                if format.action_space == ActionSpaceConfig.END_EFFECTOR:
                    if self.robot_utils is None:
                        raise ImporterError(
                            "Failed to convert action in end effector space "
                            "to joint space: Robot utilities are not initialized"
                        )
                    transformed_data = self.robot_utils.end_effector_to_joint_positions(
                        transformed_data,
                        item.name,
                        list(self.curr_joint_positions.values()),
                    )
                    for name, position in transformed_data.items():
                        self._validate_joint_data(data_type, position, name)
                elif format.action_space == ActionSpaceConfig.JOINT:
                    self._validate_joint_data(data_type, transformed_data, item.name)
            elif relative_action_requested:
                if format.action_space == ActionSpaceConfig.END_EFFECTOR:
                    if item.name not in self.curr_end_effector_poses:
                        raise DataValidationError(
                            f"End effector pose {item.name} not found in "
                            "current end effector poses"
                        )
                    transformed_data += self.curr_end_effector_poses[item.name]
                    # Get joint positions using IK from the end effector pose
                    if self.robot_utils is None:
                        raise ImporterError(
                            "Failed to convert action in end effector space "
                            "to joint space: Robot utilities are not initialized"
                        )
                    transformed_data = self.robot_utils.end_effector_to_joint_positions(
                        transformed_data, item.name, self.prev_ik_solution
                    )
                    self.prev_ik_solution = list(transformed_data.values())
                    for name, position in transformed_data.items():
                        self._validate_joint_data(
                            DataType.JOINT_POSITIONS, position, name
                        )
                elif format.action_space == ActionSpaceConfig.JOINT:
                    if item.name not in self.curr_joint_positions:
                        raise DataValidationError(
                            f"Joint position {item.name} not found in "
                            "current joint positions"
                        )
                    transformed_data += self.curr_joint_positions[item.name]
                    self._validate_joint_data(data_type, transformed_data, item.name)
            else:
                if data_type in JOINT_DATA_TYPES and self.joint_info:
                    self._validate_joint_data(data_type, transformed_data, item.name)
        except DataValidationWarning as w:
            if not self.suppress_warnings:
                self.logger.warning("[WARNING] %s (%s): %s", data_type, item.name, w)
        except DataValidationError as e:
            self.logger.error(
                "[ERROR] %s (%s): %s -- skipping episode", data_type, item.name, e
            )
            raise
        except Exception as e:
            self.logger.error(
                "[ERROR] %s (%s): %s -- skipping episode", data_type, item.name, e
            )
            raise

        try:
            if ik_requested or format.action_space == ActionSpaceConfig.END_EFFECTOR:
                for name, position in transformed_data.items():
                    self._log_transformed_data(
                        DataType.JOINT_POSITIONS,
                        position,
                        name,
                        timestamp,
                    )
            else:
                self._log_transformed_data(
                    data_type,
                    transformed_data,
                    item.name,
                    timestamp,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                )
        except Exception as e:
            self.logger.error(
                "[ERROR] %s (%s): %s -- skipping episode", data_type, item.name, e
            )
            raise

    def _log_transformed_data(
        self,
        data_type: DataType,
        transformed_data: Any,
        name: str,
        timestamp: float,
        *,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> None:
        """Log transformed data to Neuracore.

        Args:
            data_type: The type of data to log.
            transformed_data: The transformed data to log.
            name: The name of the data.
            timestamp: The timestamp of the data.
            extrinsics: Optional 4x4 camera extrinsics matrix for camera streams.
            intrinsics: Optional 3x3 camera intrinsics matrix for camera streams.
        """
        if data_type == DataType.RGB_IMAGES:
            nc.log_rgb(
                name=name,
                rgb=transformed_data,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.DEPTH_IMAGES:
            nc.log_depth(
                name=name,
                depth=transformed_data,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.POINT_CLOUDS:
            nc.log_point_cloud(
                name=name,
                points=transformed_data,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.LANGUAGE:
            nc.log_language(
                name=name,
                language=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_POSITIONS:
            self.curr_joint_positions[name] = transformed_data
            self._save_pre_check_data(DataType.JOINT_POSITIONS, name, transformed_data)
            nc.log_joint_position(
                name=name,
                position=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_VELOCITIES:
            nc.log_joint_velocity(
                name=name,
                velocity=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_TORQUES:
            nc.log_joint_torque(
                name=name,
                torque=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_TARGET_POSITIONS:
            self._save_pre_check_data(
                DataType.JOINT_TARGET_POSITIONS, name, transformed_data
            )
            nc.log_joint_target_position(
                name=name,
                target_position=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.VISUAL_JOINT_POSITIONS:
            nc.log_visual_joint_position(
                name=name,
                position=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
            nc.log_parallel_gripper_open_amount(
                name=name,
                value=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS:
            nc.log_parallel_gripper_target_open_amount(
                name=name,
                value=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.END_EFFECTOR_POSES:
            self.curr_end_effector_poses[name] = transformed_data
            nc.log_end_effector_pose(
                name=name,
                pose=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.POSES:
            nc.log_pose(
                name=name,
                pose=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.CUSTOM_1D:
            nc.log_custom_1d(
                name=name,
                data=transformed_data,
                robot_name=self.dataset_config.robot.name,
                instance=self._worker_id,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )

    def prepare_worker(
        self, worker_id: int, chunk: Sequence[ImportItem] | None = None
    ) -> None:
        """Log in and connect to Neuracore dataset for the worker."""
        nc.login()
        nc.connect_robot(self.robot_name, instance=worker_id, shared=self.shared)
        nc.get_dataset(self.output_dataset_name)
        if self.urdf_path is not None:
            urdf_packages_dir = os.path.dirname(self.urdf_path)
            self.robot_utils = RobotUtils(self.urdf_path, urdf_packages_dir)

    def import_all(self) -> None:
        """Run imports across workers while aggregating errors.

        High-level flow:
        1) Build the list of work items (episodes).
        2) Dry-run pre-check for errors
        3) Determine if joint target positions should be skipped
        4) Decide how many worker processes to spawn.
        5) Spin up workers and a progress queue.
        6) Listen for progress updates while workers run.
        7) Collect and summarize any errors.
        """
        items = list(self.build_work_items())
        if not items:
            self.logger.info("No import items found; nothing to do.")
            return

        original_count = len(items)
        completed_keys = self._load_completed_item_keys()
        if completed_keys:
            items = [
                item for item in items if self._item_key(item) not in completed_keys
            ]
            skipped_count = original_count - len(items)
            if skipped_count > 0:
                self.logger.info(
                    "Skipping %s previously completed episode(s) from %s.",
                    skipped_count,
                    self.completed_items_file,
                )
            if not items:
                self.logger.info(
                    "All episodes are already completed according to %s.",
                    self.completed_items_file,
                )
                return

        if self.random_sample is not None:
            n = min(self.random_sample, len(items))
            items = random.sample(items, n)
            self.logger.info(
                "Sampling %s random episode(s) from %s episodes.",
                n,
                original_count,
            )

        # Pre-check: dry run to check for errors
        precheck_items = items[:1]
        original_dry_run = self.dry_run
        self.dry_run = True
        self.pre_check = True
        skip_joint_target_positions = False
        try:
            self.logger.info(
                "Pre-check: importing %s episode(s) with 1 worker.",
                len(precheck_items),
            )
            precheck_errors, precheck_processes, precheck_result_queue = (
                self._run_import_workers(
                    precheck_items, 1, use_precheck_result_queue=True
                )
            )
            self._report_process_status(precheck_processes)
            if precheck_errors:
                first = precheck_errors[0]
                msg = f"Pre-check failed: {first.message}" + (
                    f" (worker {first.worker_id}, item {first.item_index})"
                    if first.item_index is not None
                    else f" (worker {first.worker_id})"
                )
                raise ImporterError(msg)
            if precheck_result_queue is not None:
                try:
                    skip_joint_target_positions = precheck_result_queue.get_nowait()
                except Empty:
                    skip_joint_target_positions = False
        finally:
            self.dry_run = original_dry_run
            self.pre_check = False
            if skip_joint_target_positions:
                self.ordered_import_configs = [
                    c
                    for c in self.ordered_import_configs
                    if c[0] != DataType.JOINT_TARGET_POSITIONS
                ]
                self.logger.warning(
                    "Joint target positions provided are equivalent to joint positions "
                    "in next step. Skip importing joint target positions."
                )

        worker_count = self._resolve_worker_count(len(items))
        live_status = "Live" if not self.dry_run else "Dry-run"
        self.logger.info(
            "%s: importing %s episodes with %s workers",
            live_status,
            len(items),
            worker_count,
        )

        self.worker_errors, processes, _ = self._run_import_workers(items, worker_count)
        self._report_process_status(processes)
        self._report_errors(self.worker_errors)

        if self.worker_errors and self.skip_on_error == "all":
            raise ImporterError("Import aborted due to worker errors.")

    def _run_import_workers(
        self,
        items: Sequence[ImportItem],
        worker_count: int,
        use_precheck_result_queue: bool = False,
    ) -> tuple[
        list[WorkerError],
        list[mp.context.SpawnProcess],
        mp.Queue[bool] | None,
    ]:
        """Run import with the given items and worker count."""
        ctx = mp.get_context("spawn")
        error_queue: mp.Queue[WorkerError] = ctx.Queue()
        progress_queue: mp.Queue[ProgressUpdate] = ctx.Queue()
        completed_queue: mp.Queue[str] = ctx.Queue()
        pause_event = ctx.Event()  # when set, workers pause at next item boundary
        precheck_result_queue: mp.Queue[bool] | None = (
            ctx.Queue() if use_precheck_result_queue else None
        )
        chunks = self._partition(items, worker_count)
        non_empty_chunks = [(wid, c) for wid, c in enumerate(chunks) if c]
        ready_barrier = ctx.Barrier(len(non_empty_chunks))
        processes: list[mp.context.SpawnProcess] = []

        for worker_id, chunk in non_empty_chunks:
            process = ctx.Process(
                target=self._worker_entry,
                args=(
                    chunk,
                    worker_id,
                    error_queue,
                    progress_queue,
                    completed_queue,
                    ready_barrier,
                    pause_event,
                    precheck_result_queue,
                ),
            )
            process.start()
            processes.append(process)

        self._monitor_progress(
            processes, progress_queue, len(items), pause_event, completed_queue
        )

        for process in processes:
            process.join()
        progress_queue.close()
        progress_queue.join_thread()
        completed_queue.close()
        completed_queue.join_thread()

        return (
            self._collect_errors(error_queue),
            processes,
            precheck_result_queue,
        )

    def _resolve_worker_count(self, total_items: int) -> int:
        """Pick a worker count similar to the archived scripts."""
        if self.max_workers is not None:
            return max(1, min(self.max_workers, total_items))
        cpu_count = os.cpu_count()
        default = max(
            self.min_workers,
            int(cpu_count * 0.8) if cpu_count is not None else self.min_workers,
        )
        return max(self.min_workers, min(default, total_items))

    def _partition(
        self, items: Sequence[ImportItem], worker_count: int
    ) -> list[list[ImportItem]]:
        """Partition work into contiguous chunks to preserve order."""
        total = len(items)
        if worker_count <= 1 or total <= 1:
            return [list(items)]

        chunk_size = max(1, total // worker_count)
        chunks: list[list[ImportItem]] = []
        for i in range(worker_count):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < worker_count - 1 else total
            if start >= total:
                break
            chunks.append(list(items[start:end]))
        return chunks

    def _worker_entry(
        self,
        chunk: Sequence[ImportItem],
        worker_id: int,
        error_queue: mp.Queue,
        progress_queue: mp.Queue | None,
        completed_queue: mp.Queue[str] | None,
        ready_barrier: mp.synchronize.Barrier | None = None,
        pause_event: mp.synchronize.Event | None = None,
        precheck_result_queue: mp.Queue[bool] | None = None,
    ) -> None:
        """Worker body that wraps import with error capture.

        All workers prepare first, then synchronise on *ready_barrier*
        before any of them starts importing items.  If any worker fails
        during preparation it aborts the barrier so the remaining workers
        can exit promptly. When *pause_event* is set (e.g. low disk),
        workers block at item boundaries until it is cleared.
        When *precheck_result_queue* is provided and pre_check is True,
        the result of _determine_skip_joint_target_positions() is put on the queue.
        """
        self._worker_id = worker_id
        self._progress_queue = progress_queue

        try:
            sig = inspect.signature(self.prepare_worker)
            if "chunk" in sig.parameters:
                self.prepare_worker(worker_id, chunk)
            else:
                self.prepare_worker(worker_id)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - propagate unexpected worker failures
            self._abort_barrier(ready_barrier)
            if error_queue:
                tb = traceback.format_exc()
                error_queue.put(
                    WorkerError(
                        worker_id=worker_id,
                        item_index=None,
                        message=str(exc),
                        traceback=tb,
                    )
                )
            self._log_worker_error(worker_id, None, str(exc))
            raise

        if ready_barrier is not None:
            try:
                ready_barrier.wait()
            except threading.BrokenBarrierError:
                return

        for idx, item in enumerate(chunk):
            while pause_event is not None and pause_event.is_set():
                time.sleep(1)
            try:
                self._step(
                    item,
                    worker_id,
                    idx,
                    len(chunk),
                    error_queue,
                    completed_queue,
                )
            except Exception as exc:  # noqa: BLE001 - keep traceback for summary
                if error_queue:
                    tb = traceback.format_exc()
                    error_queue.put(
                        WorkerError(
                            worker_id=worker_id,
                            item_index=item.index,
                            message=str(exc),
                            traceback=tb,
                        )
                    )
                self._log_worker_error(worker_id, item.index, str(exc))
                raise

        if self.pre_check and precheck_result_queue is not None:
            try:
                result = self._determine_skip_joint_target_positions()
                precheck_result_queue.put(result)
            except Exception:  # noqa: BLE001
                precheck_result_queue.put(False)

    @staticmethod
    def _abort_barrier(barrier: mp.synchronize.Barrier | None) -> None:
        """Break the barrier so waiting workers unblock immediately."""
        if barrier is not None:
            try:
                barrier.abort()
            except Exception:  # noqa: BLE001, S110
                pass

    def _step(
        self,
        item: ImportItem,
        worker_id: int,
        local_index: int,
        chunk_length: int,
        error_queue: mp.Queue,
        completed_queue: mp.Queue[str] | None,
    ) -> None:
        """Centralized step handler for progress and error capture."""
        self._error_queue = error_queue
        try:
            self.import_item(item)
        except Exception as exc:  # noqa: BLE001 - keep traceback for summary
            tb = traceback.format_exc()
            if self.skip_on_error == "episode":
                error_queue.put(
                    WorkerError(
                        worker_id=worker_id,
                        item_index=item.index,
                        message=str(exc),
                        traceback=tb,
                    )
                )
                # Defer logging to the post-run summary to avoid flickering
                # and duplicate error lines while the progress bar is live.
                return
            self._log_worker_error(worker_id, item.index, str(exc))
            raise

        if completed_queue is not None and not self.dry_run and not self.pre_check:
            completed_queue.put(self._item_key(item))

        # Progress bar already shows ongoing status; skip per-interval info logs.

    def _collect_errors(self, error_queue: mp.Queue) -> list[WorkerError]:
        """Drain the error queue after workers complete."""
        errors: list[WorkerError] = []
        try:
            while True:
                errors.append(error_queue.get_nowait())
        except Empty:
            pass
        return errors

    def _resolve_completed_items_file(self) -> Path:
        """Return the file path used to persist completed imports."""
        safe_dataset_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.output_dataset_name)
        return self.dataset_dir / f".neuracore_import_completed_{safe_dataset_name}.txt"

    def _item_key(self, item: ImportItem) -> str:
        """Build a stable key for an import item."""
        split = item.split if item.split is not None else ""
        return f"{split}:{item.index}"

    def _load_completed_item_keys(self) -> set[str]:
        """Load completed item keys from disk."""
        if not self.completed_items_file.exists():
            return set()
        try:
            with self.completed_items_file.open("r", encoding="utf-8") as file:
                return {line.strip() for line in file if line.strip()}
        except Exception as exc:  # noqa: BLE001 - best effort
            self.logger.warning(
                "Failed to read completed imports file '%s': %s",
                self.completed_items_file,
                exc,
            )
            return set()

    def _append_completed_item_keys(self, keys: Iterable[str]) -> None:
        """Append completed item keys to disk immediately.

        Used to periodically persist resume state while workers run.
        """
        if self.dry_run or self.pre_check:
            return
        keys = list(keys)
        if not keys:
            return
        self.completed_items_file.parent.mkdir(parents=True, exist_ok=True)
        with self.completed_items_file.open("a", encoding="utf-8") as file:
            for key in keys:
                file.write(f"{key}\n")

    def _report_process_status(
        self, processes: Iterable[mp.context.SpawnProcess]
    ) -> None:
        """Log any non-zero exit codes from worker processes."""
        for process in processes:
            if process.exitcode not in (0, None):
                self.logger.error(
                    "Worker pid=%s exited with status %s",
                    process.pid,
                    process.exitcode,
                )

    def _report_errors(self, errors: list[WorkerError]) -> None:
        """Summarize captured worker errors."""
        if not errors:
            self.logger.info("All workers completed without reported errors.")
            return

        deduped: dict[tuple[int | None, int | None, str], int] = {}
        for err in errors:
            key = (err.worker_id, err.item_index, err.message)
            deduped[key] = deduped.get(key, 0) + 1

        self.logger.error(
            "Completed with %s worker error event(s) (%s unique).",
            len(errors),
            len(deduped),
        )

        for (worker_id, item_index, message), count in deduped.items():
            prefix = f"[worker {worker_id}"
            if item_index is not None:
                prefix += f" item {item_index}"
            prefix += "]"
            suffix = f" (x{count})" if count > 1 else ""
            self.logger.error("%s %s%s", prefix, message, suffix)

        self.logger.error(
            "Import finished with errors. Re-run with DEBUG logging for tracebacks "
            "or fix the reported issues above."
        )

    def _log_worker_error(
        self, worker_id: int, item_index: int | None, message: str
    ) -> None:
        """Log a worker error immediately while the process is running."""
        key = (worker_id, item_index, message)
        if key in self._logged_error_keys:
            return
        self._logged_error_keys.add(key)

        prefix = f"[worker {worker_id}"
        if item_index is not None:
            prefix += f" item {item_index}"
        prefix += "]"
        self.logger.error("%s %s", prefix, message)

    def _emit_progress(
        self,
        item_index: int,
        step: int,
        total_steps: int | None,
        episode_label: str | None = None,
    ) -> None:
        """Send a progress update to the main process if available."""
        if self._progress_queue is None or self._worker_id == -1:
            return
        try:
            self._progress_queue.put_nowait(
                ProgressUpdate(
                    worker_id=self._worker_id,
                    item_index=item_index,
                    step=step,
                    total_steps=total_steps,
                    episode_label=episode_label,
                )
            )
        except Exception:  # noqa: BLE001 - best-effort progress updates
            self.logger.debug("Failed to emit progress update.", exc_info=True)

    @staticmethod
    def _format_bytes(num_bytes: int | float) -> str:
        """Format a byte count into a human-readable string."""
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(num_bytes) < 1024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.1f} PiB"

    def _get_disk_check_path(self) -> Path:
        """Return the filesystem path to monitor for disk/folder usage.

        Defaults to the daemon recording directory.
        """
        return DEFAULT_RECORDING_ROOT_PATH

    def _get_disk_usage(self) -> int:
        """Return total size in bytes of the monitored folder."""
        path = self._get_disk_check_path()
        if not path.exists():
            return 0
        total_size = 0
        if path.is_file():
            return path.stat().st_size
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except (OSError, FileNotFoundError):
                    pass  # Skip files that disappear or can't be accessed
        return total_size

    def _monitor_progress(
        self,
        processes: Sequence[mp.context.SpawnProcess],
        progress_queue: mp.Queue[ProgressUpdate],
        total_items: int,
        pause_event: mp.synchronize.Event | None = None,
        completed_queue: mp.Queue[str] | None = None,
    ) -> None:
        """Render a progress bar by aggregating all worker updates.

        Workers are paused when disk usage reaches storage_limit until
        usage drops below it. Completed item keys from completed_queue
        are appended to the resume file periodically.
        """
        if not processes:
            return

        completed_items: set[int] = set()
        worker_states: dict[int, ProgressUpdate] = {}
        last_disk_check = 0.0
        workers_paused = False

        def flush_completed_queue() -> None:
            if completed_queue is None:
                return
            batch: list[str] = []
            try:
                while True:
                    batch.append(completed_queue.get_nowait())
            except Empty:
                pass
            if batch:
                self._append_completed_item_keys(batch)

        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None, complete_style="green", pulse_style="cyan"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
            transient=True,
            console=get_shared_console(),
        ) as progress:
            overall_task = progress.add_task("Importing episodes", total=total_items)

            while True:
                flush_completed_queue()

                now = time.monotonic()
                if now - last_disk_check >= self.DISK_CHECK_INTERVAL_SECS:
                    last_disk_check = now
                    try:
                        used_bytes = self._get_disk_usage()
                        pct_used = used_bytes / self.storage_limit * 100
                        used_str = self._format_bytes(used_bytes)
                        limit_str = self._format_bytes(self.storage_limit)

                        if used_bytes >= self.storage_limit:
                            if pause_event is not None and not workers_paused:
                                workers_paused = True
                                pause_event.set()
                                self.logger.warning(
                                    "Local cache limit exceeded "
                                    f"({used_str} of {limit_str}). "
                                    "Pausing workers until usage drops below limit."
                                )
                            progress.update(
                                overall_task,
                                description=(
                                    f"Paused ({pct_used:.0f}% of local cache used: "
                                    f"{used_str} / {limit_str}). "
                                    "Waiting for recordings to upload to cloud "
                                    "and free up local cache."
                                ),
                            )
                            time.sleep(self.DISK_CHECK_INTERVAL_SECS)
                            continue
                        else:
                            if workers_paused and pause_event is not None:
                                workers_paused = False
                                pause_event.clear()
                                self.logger.info(
                                    "Local cache usage below limit "
                                    f"({used_str} of {limit_str}). "
                                    "Resuming import."
                                )
                            progress.update(
                                overall_task,
                                description=(
                                    f"Importing episodes "
                                    f"({pct_used:.0f}% of local cache used: "
                                    f"{used_str} / {limit_str})."
                                ),
                            )
                    except Exception:  # noqa: BLE001 - best-effort disk check
                        pass

                any_alive = any(proc.is_alive() for proc in processes)
                timeout = 0.1 if any_alive else 0
                try:
                    update = progress_queue.get(timeout=timeout)
                except Empty:
                    update = None

                if update is not None:
                    self._apply_progress_update(
                        progress,
                        overall_task,
                        completed_items,
                        worker_states,
                        update,
                    )

                if not any_alive:
                    while True:
                        try:
                            update = progress_queue.get_nowait()
                        except Empty:
                            break
                        self._apply_progress_update(
                            progress,
                            overall_task,
                            completed_items,
                            worker_states,
                            update,
                        )
                    flush_completed_queue()
                    progress.update(overall_task, completed=total_items)
                    return

    def _apply_progress_update(
        self,
        progress: Progress,
        overall_task: TaskID,
        completed_items: set[int],
        worker_states: dict[int, ProgressUpdate],
        update: ProgressUpdate,
    ) -> None:
        """Process a single progress update and refresh the unified bar."""
        prev = worker_states.get(update.worker_id)
        if prev is not None and prev.item_index != update.item_index:
            completed_items.add(prev.item_index)

        if update.total_steps is not None and update.step >= update.total_steps:
            completed_items.add(update.item_index)

        worker_states[update.worker_id] = update

        progress.update(
            overall_task,
            completed=len(completed_items),
        )
