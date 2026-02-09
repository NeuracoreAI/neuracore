"""Shared importer framework for dataset ingestion workflows."""

from __future__ import annotations

import inspect
import logging
import multiprocessing as mp
import os
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from neuracore_types.importer.data_config import DataFormat
from neuracore_types.nc_data import DatasetImportConfig, DataType
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
        dry_run: bool = False,
        suppress_warnings: bool = False,
    ) -> None:
        """Initialize the base dataset importer."""
        self.dataset_dir = Path(dataset_dir)
        self.dataset_config = dataset_config
        self.data_config = dataset_config  # Backwards-compat alias used by callers
        self.output_dataset_name = output_dataset_name
        self.robot_name = dataset_config.robot.name
        self.frequency = dataset_config.frequency
        self.joint_info = joint_info

        if skip_on_error not in {"episode", "step", "all"}:
            raise ValueError("skip_on_error must be one of: 'episode', 'step', 'all'")

        self.max_workers = 1
        self.min_workers = min_workers
        self.skip_on_error = skip_on_error  # one of: "episode", "step", "all"
        self.progress_interval = max(1, progress_interval)
        self.dry_run = dry_run
        self.suppress_warnings = suppress_warnings
        self.worker_errors: list[WorkerError] = []
        self._logged_error_keys: set[tuple[int | None, int | None, str]] = set()
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._progress_queue: mp.Queue[ProgressUpdate] | None = None
        self._worker_id: int | None = None
        self._error_queue: mp.Queue[WorkerError] | None = None

    @abstractmethod
    def build_work_items(self) -> Sequence[ImportItem]:
        """Enumerate importable units in deterministic order."""

    @abstractmethod
    def import_item(self, item: ImportItem) -> None:
        """Perform the dataset-specific import for a single item."""

    @abstractmethod
    def _record_step(self, step: dict, timestamp: float) -> None:
        """Record a single step of the dataset."""

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
        extrinsics: Any | None = None,
        intrinsics: Any | None = None,
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
        try:
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
            transformed_data = item.transforms(source_data)
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
        extrinsics: Any | None = None,
        intrinsics: Any | None = None,
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
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.LANGUAGE:
            nc.log_language(
                name=name,
                language=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_POSITIONS:
            nc.log_joint_position(
                name=name,
                position=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_VELOCITIES:
            nc.log_joint_velocity(
                name=name,
                velocity=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_TORQUES:
            nc.log_joint_torque(
                name=name,
                torque=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.JOINT_TARGET_POSITIONS:
            nc.log_joint_target_position(
                name=name,
                target_position=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.VISUAL_JOINT_POSITIONS:
            nc.log_visual_joint_position(
                name=name,
                position=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
            nc.log_parallel_gripper_open_amount(
                name=name,
                value=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS:
            nc.log_parallel_gripper_target_open_amount(
                name=name,
                value=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.END_EFFECTOR_POSES:
            nc.log_end_effector_pose(
                name=name,
                pose=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.POSES:
            nc.log_pose(
                name=name,
                pose=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )
        elif data_type == DataType.CUSTOM_1D:
            nc.log_custom_1d(
                name=name,
                data=transformed_data,
                robot_name=self.dataset_config.robot.name,
                timestamp=timestamp,
                dry_run=self.dry_run,
            )

    def prepare_worker(
        self, worker_id: int, chunk: Sequence[ImportItem] | None = None
    ) -> None:
        """Log in and connect to Neuracore dataset for the worker."""
        nc.login()
        nc.connect_robot(self.robot_name, instance=worker_id)
        nc.get_dataset(self.output_dataset_name)

    def import_all(self) -> None:
        """Run imports across workers while aggregating errors.

        High-level flow:
        1) Build the list of work items (episodes).
        2) Decide how many worker processes to spawn.
        3) Spin up workers and a progress queue.
        4) Listen for progress updates while workers run.
        5) Collect and summarize any errors.
        """
        items = list(self.build_work_items())
        if not items:
            self.logger.info("No import items found; nothing to do.")
            return

        worker_count = self._resolve_worker_count(len(items))
        os.cpu_count()

        ctx = mp.get_context("spawn")
        error_queue: mp.Queue[WorkerError] = ctx.Queue()
        progress_queue: mp.Queue[ProgressUpdate] = ctx.Queue()
        chunks = self._partition(items, worker_count)
        processes: list[mp.context.SpawnProcess] = []

        for worker_id, chunk in enumerate(chunks):
            if not chunk:
                continue
            process = ctx.Process(
                target=self._worker_entry,
                args=(chunk, worker_id, error_queue, progress_queue),
            )
            process.start()
            processes.append(process)

        self._monitor_progress(processes, progress_queue)

        for process in processes:
            process.join()
        progress_queue.close()
        progress_queue.join_thread()

        self.worker_errors = self._collect_errors(error_queue)
        self._report_process_status(processes)
        self._report_errors(self.worker_errors)

        if self.worker_errors and self.skip_on_error == "all":
            raise ImporterError("Import aborted due to worker errors.")

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
    ) -> None:
        """Worker body that wraps import with error capture."""
        self._worker_id = worker_id
        self._progress_queue = progress_queue
        try:
            sig = inspect.signature(self.prepare_worker)
            if "chunk" in sig.parameters:
                self.prepare_worker(worker_id, chunk)
            else:
                self.prepare_worker(worker_id)  # type: ignore[misc]
            # Progress bar will reflect work; keep startup quiet to reduce noise.
        except Exception as exc:  # noqa: BLE001 - propagate unexpected worker failures
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

        for idx, item in enumerate(chunk):
            try:
                self._step(item, worker_id, idx, len(chunk), error_queue)
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

    def _step(
        self,
        item: ImportItem,
        worker_id: int,
        local_index: int,
        chunk_length: int,
        error_queue: mp.Queue,
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
        if self._progress_queue is None or self._worker_id is None:
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

    def _monitor_progress(
        self,
        processes: Sequence[mp.context.SpawnProcess],
        progress_queue: mp.Queue[ProgressUpdate],
    ) -> None:
        """Render rich progress bars based on worker updates."""
        if not processes:
            return

        task_map: dict[int, TaskID] = {}
        current_items: dict[int, int] = {}

        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]Worker {task.fields[worker]}"),
            TextColumn("| Episode {task.fields[episode]}"),
            BarColumn(bar_width=None, complete_style="green", pulse_style="cyan"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
            transient=True,
            console=get_shared_console(),
        ) as progress:
            while True:
                any_alive = any(proc.is_alive() for proc in processes)
                timeout = 0.1 if any_alive else 0
                try:
                    update = progress_queue.get(timeout=timeout)
                except Empty:
                    update = None

                if update is not None:
                    self._apply_progress_update(
                        progress, task_map, current_items, update
                    )

                if not any_alive:
                    # Drain remaining updates before exiting.
                    while True:
                        try:
                            update = progress_queue.get_nowait()
                        except Empty:
                            return
                        self._apply_progress_update(
                            progress, task_map, current_items, update
                        )

    def _apply_progress_update(
        self,
        progress: Progress,
        task_map: dict[int, TaskID],
        current_items: dict[int, int],
        update: ProgressUpdate,
    ) -> None:
        """Update or create the progress task for a worker."""
        desc = update.episode_label or str(update.item_index)
        task_id = task_map.get(update.worker_id)

        if task_id is None:
            task_id = progress.add_task(
                f"Episode {desc}",
                total=update.total_steps,
                completed=update.step,
                worker=update.worker_id,
                episode=desc,
            )
            task_map[update.worker_id] = task_id
            current_items[update.worker_id] = update.item_index
            return

        if current_items.get(update.worker_id) != update.item_index:
            current_items[update.worker_id] = update.item_index
            progress.update(
                task_id,
                total=update.total_steps,
                completed=update.step,
                description=f"Episode {desc}",
                worker=update.worker_id,
                episode=desc,
                refresh=True,
            )
            return

        progress.update(
            task_id,
            total=update.total_steps,
            completed=update.step,
            description=f"Episode {desc}",
            episode=desc,
            refresh=True,
        )
