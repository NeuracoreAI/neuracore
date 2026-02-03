"""Shared importer for RLDS/TFDS-style datasets."""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tensorflow_datasets as tfds
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.nc_data import DatasetImportConfig

import neuracore as nc
from neuracore.core.robot import JointInfo
from neuracore.importer.core.base import (
    ImportItem,
    NeuracoreDatasetImporter,
    WorkerError,
)
from neuracore.importer.core.exceptions import ImportError

# Suppress TensorFlow informational messages (e.g., "End of sequence")
# 0 = all logs, 1 = no INFO, 2 = no WARNING, 3 = no ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


class RLDSAndTFDSDatasetImporterBase(NeuracoreDatasetImporter):
    """Base class for RLDS/TFDS dataset importers."""

    dataset_label = "DATASET"
    allow_mapping_name_fallback = False

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        joint_info: dict[str, JointInfo] = {},
        ik_urdf_path: str | None = None,
        ik_init_config: list[float] | None = None,
        dry_run: bool = False,
        suppress_warnings: bool = False,
        max_workers: int | None = 1,
        skip_on_error: str = "episode",
    ):
        """Initialize the RLDS/TFDS dataset importer.

        Args:
            input_dataset_name: Name of the dataset to import.
            output_dataset_name: Name of the dataset to create.
            dataset_dir: Directory containing the dataset.
            dataset_config: Dataset configuration.
            joint_info: Joint info to use for validation.
            ik_urdf_path: URDF path for IK (used to recreate IK in worker processes).
            ik_init_config: Initial joint configuration for IK.
            dry_run: If True, skip actual logging (validation only).
            suppress_warnings: If True, suppress warning messages.
            max_workers: Maximum number of worker processes.
            skip_on_error: "episode" to skip a failed episode; "step" to skip only
                failing steps; "all" to abort on the first error.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=max_workers,
            joint_info=joint_info,
            ik_urdf_path=ik_urdf_path,
            ik_init_config=ik_init_config,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
            skip_on_error=skip_on_error,
        )
        self.dataset_name = input_dataset_name
        self.builder_dir = self._resolve_builder_dir()
        if self.frequency is None:
            raise ImportError(
                f"Dataset frequency is required for {self.dataset_label} imports."
            )

        builder = self._load_builder()
        self.split = self._pick_split(builder)
        self.num_episodes = self._count_episodes(builder, self.split)

        self._builder: tfds.core.DatasetBuilder | None = None
        self._episode_iter = None

        self.logger.info(
            "Initialized %s importer for '%s' "
            "(split=%s, episodes=%s, freq=%s, dir=%s)",
            self.dataset_label,
            self.dataset_name,
            self.split,
            self.num_episodes,
            self.frequency,
            self.builder_dir,
        )

    def __getstate__(self) -> dict:
        """Drop worker-local handles when pickling for multiprocessing."""
        state = self.__dict__.copy()
        state.pop("_builder", None)
        state.pop("_episode_iter", None)
        return state

    def build_work_items(self) -> Sequence[ImportItem]:
        """Build work items for the dataset importer."""
        return [
            ImportItem(index=i, split=str(self.split)) for i in range(self.num_episodes)
        ]

    def prepare_worker(
        self, worker_id: int, chunk: Sequence[ImportItem] | None = None
    ) -> None:
        """Prepare the worker for the dataset importer."""
        super().prepare_worker(worker_id, chunk)
        self._builder = self._load_builder()
        chunk_start = chunk[0].index if chunk else 0
        chunk_length = len(chunk) if chunk else None

        self.logger.info(
            "[worker %s] Loading split=%s (start=%s count=%s) from %s",
            worker_id,
            self.split,
            chunk_start,
            chunk_length if chunk_length is not None else "remainder",
            self.builder_dir,
        )

        dataset = self._load_dataset(self._builder, self.split)
        if chunk_start:
            dataset = dataset.skip(chunk_start)
        if chunk_length is not None:
            dataset = dataset.take(chunk_length)
        self._episode_iter = iter(dataset)

    def import_item(self, item: ImportItem) -> None:
        """Import a single episode to the dataset importer."""
        if self._episode_iter is None:
            raise ImportError("Worker dataset iterator was not initialized.")

        try:
            episode = next(self._episode_iter)
        except Exception as exc:  # delegate to subclass
            self._handle_episode_load_error(exc, item)

        steps = episode["steps"]
        if self.frequency is None:
            raise ImportError("Frequency is required for importing episodes.")
        total_steps = self._infer_total_steps(steps)
        base_time = time.time()
        nc.start_recording()
        episode_label = (
            f"{item.split or 'episode'} #{item.index}"
            if item.split is not None
            else str(item.index)
        )
        worker_label = (
            f"worker {self._worker_id}" if self._worker_id is not None else "worker 0"
        )
        self.logger.info(
            "[%s] Importing %s (%s/%s, steps=%s)",
            worker_label,
            episode_label,
            item.index + 1,
            self.num_episodes,
            total_steps if total_steps is not None else "unknown",
        )
        self._emit_progress(
            item.index, step=0, total_steps=total_steps, episode_label=episode_label
        )
        for idx, step in enumerate(steps, start=1):
            timestamp = base_time + (idx / self.frequency)
            try:
                self._record_step(step, timestamp)
            except Exception as exc:  # importer-specific policy hook
                if self._handle_step_error(exc, item, idx):
                    continue
                raise
            self._emit_progress(
                item.index,
                step=idx,
                total_steps=total_steps,
                episode_label=episode_label,
            )
        nc.stop_recording(wait=True)
        self.logger.info("[%s] Completed %s", worker_label, episode_label)

    def _handle_step_error(
        self, exc: Exception, item: ImportItem, step_index: int
    ) -> bool:
        """Skip failing steps when configured with skip_on_error='step'.

        Returns:
            True if the error is handled and import should continue with next step.
            False to re-raise and fail the current item.
        """
        if getattr(self, "skip_on_error", "episode") != "step":
            return False

        worker_id_attr = getattr(self, "_worker_id", None)
        worker_id = worker_id_attr if worker_id_attr is not None else 0

        error_queue = getattr(self, "_error_queue", None)
        if error_queue is not None:
            error_queue.put(
                WorkerError(
                    worker_id=worker_id,
                    item_index=item.index,
                    message=f"Step {step_index}: {exc}",
                    traceback=traceback.format_exc(),
                )
            )
        self._log_worker_error(worker_id, item.index, f"Step {step_index}: {exc}")
        return True

    def _handle_episode_load_error(self, exc: Exception, item: ImportItem) -> None:
        """Map dataset iteration exceptions into importer-level errors."""
        if isinstance(exc, StopIteration):
            raise ImportError(
                f"No episode available for index {item.index} "
                f"(dataset has {self.num_episodes} episodes)."
            ) from exc
        raise exc

    def _resolve_builder_dir(self) -> Path:
        """Find the dataset version directory that contains dataset_info.json."""
        if (self.dataset_dir / "dataset_info.json").exists():
            return self.dataset_dir

        version_dirs = [
            path
            for path in self.dataset_dir.iterdir()
            if path.is_dir() and (path / "dataset_info.json").exists()
        ]
        if version_dirs:
            return sorted(version_dirs)[-1]

        name_dir = self.dataset_dir / self.dataset_name
        if (name_dir / "dataset_info.json").exists():
            return name_dir
        nested_versions = (
            [
                path
                for path in name_dir.iterdir()
                if path.is_dir() and (path / "dataset_info.json").exists()
            ]
            if name_dir.exists()
            else []
        )
        if nested_versions:
            return sorted(nested_versions)[-1]

        raise ImportError(
            f"Could not find dataset_info.json under {self.dataset_dir}. "
            "Pass either the dataset version directory or its parent."
        )

    def _load_builder(self) -> tfds.core.DatasetBuilder:
        """Load a TFDS builder directly from the local dataset directory."""
        self.logger.info(
            "Loading %s builder from %s", self.dataset_label, self.builder_dir
        )
        try:
            builder = tfds.builder_from_directory(str(self.builder_dir))
            self._on_builder_loaded(builder)
            return builder
        except Exception as exc:
            raise ImportError(
                f"Failed to load {self.dataset_label} builder from "
                f"'{self.builder_dir}': {exc}"
            ) from exc

    def _on_builder_loaded(self, builder: tfds.core.DatasetBuilder) -> None:
        """Hook for subclass-specific builder checks."""
        return None

    def _pick_split(self, builder: tfds.core.DatasetBuilder) -> tfds.typing.SplitArg:
        """Select split to inspect; default to all splits."""
        splits = list(builder.info.splits.keys())
        if not splits:
            raise ImportError(
                f"No splits found in {self.dataset_label} dataset at "
                f"'{self.builder_dir}'."
            )
        return tfds.Split.ALL

    def _count_episodes(
        self, builder: tfds.core.DatasetBuilder, split: tfds.typing.SplitArg
    ) -> int:
        """Count the number of episodes in the chosen split."""
        if split == tfds.Split.ALL or str(split).lower() == "all":
            return int(builder.info.splits.total_num_examples)
        try:
            split_info = builder.info.splits[split]
        except KeyError:
            split_info = builder.info.splits[str(split)]
        return int(split_info.num_examples)

    def _build_read_config(self) -> tfds.ReadConfig:
        """Build read config for normal dataset loading."""
        return tfds.ReadConfig(try_autocache=False)

    def _build_retry_read_config(self) -> tfds.ReadConfig | None:
        """Build read config for retry after missing-file errors."""
        return None

    @staticmethod
    def _is_missing_file_error(exc: Exception) -> bool:
        error_msg = str(exc).lower()
        return "no such file or directory" in error_msg or "not found" in error_msg

    def _load_dataset(
        self, builder: tfds.core.DatasetBuilder, split: tfds.typing.SplitArg
    ) -> tfds.core.dataset_builder.DatasetBuilder:
        """Load the TFDS dataset from the local builder."""
        self.logger.info("Opening dataset split '%s' for import.", split)
        try:
            return builder.as_dataset(
                split=split,
                shuffle_files=False,
                read_config=self._build_read_config(),
            )
        except Exception as exc:
            retry_config = self._build_retry_read_config()
            if self._is_missing_file_error(exc) and retry_config is not None:
                self.logger.warning(
                    "Some dataset shard files appear to be missing. "
                    "This may indicate an incomplete dataset. "
                    "Attempting to continue with available shards. Error: %s",
                    exc,
                )
                try:
                    return builder.as_dataset(
                        split=split,
                        shuffle_files=False,
                        read_config=retry_config,
                    )
                except Exception as retry_exc:
                    raise ImportError(
                        f"Failed to load {self.dataset_label} dataset split "
                        f"'{split}' even with lenient configuration. "
                        f"Original error: {exc}. Retry error: {retry_exc}. "
                        "Please ensure all dataset shard files are present."
                    ) from retry_exc
            raise ImportError(
                f"Failed to load {self.dataset_label} dataset split '{split}': {exc}"
            ) from exc

    def _infer_total_steps(self, steps: Any) -> int | None:
        """Best-effort step count extraction without materializing the dataset."""
        try:
            if not isinstance(steps, dict):
                length = len(steps)
                if isinstance(length, int):
                    return length
        except Exception:
            pass

        for attr in ("shape", "shapes"):
            try:
                shape = getattr(steps, attr)
                first_dim = shape[0] if shape else None
                if isinstance(first_dim, int):
                    return first_dim
            except Exception:
                continue

        if isinstance(steps, dict):
            for value in steps.values():
                try:
                    length = len(value)
                except Exception:
                    continue
                if isinstance(length, int):
                    return length
        return None

    def _record_step(self, step_data: dict, timestamp: float) -> None:
        """Record a single step to Neuracore."""
        import tensorflow as tf

        for data_type, import_config in self.dataset_config.data_import_config.items():
            source = step_data
            for path in import_config.source.split("."):
                source = source[path]

            for item in import_config.mapping:
                try:
                    source_data = self._extract_source_data(
                        source=source,
                        item=item,
                        import_source_path=import_config.source,
                        data_type=data_type,
                    )

                    source_data = self._convert_source_data(
                        source_data=source_data,
                        data_type=data_type,
                        language_type=import_config.format.language_type,
                        item_name=item.name,
                        import_source_path=import_config.source,
                        tf_module=tf,
                    )
                except (KeyError, IndexError, TypeError) as exc:
                    raise ImportError(
                        f"Failed to extract data for {data_type.value}."
                        f"{item.name if hasattr(item, 'name') else 'unknown'} "
                        f"from source '{import_config.source}': {exc}"
                    ) from exc

                self._log_data(
                    data_type, source_data, item, import_config.format, timestamp
                )

    def _extract_source_data(
        self,
        source: Any,
        item: Any,
        import_source_path: str,
        data_type: DataType,
    ) -> Any:
        if item.source_name is not None:
            source_data = source[item.source_name]
        else:
            source_data = source

        if item.index is not None:
            if isinstance(source_data, dict):
                raise ImportError(
                    "Cannot index dict with integer index for "
                    f"{data_type.value}."
                    f"{item.name if hasattr(item, 'name') else 'unknown'}. "
                    f"Source path '{import_source_path}' resolved to a dict, "
                    "not a tensor. Check your dataset config."
                )
            source_data = source_data[item.index]
        elif item.index_range is not None:
            if isinstance(source_data, dict):
                raise ImportError(
                    "Cannot slice dict with index_range for "
                    f"{data_type.value}."
                    f"{item.name if hasattr(item, 'name') else 'unknown'}. "
                    f"Source path '{import_source_path}' resolved to a dict, "
                    "not a tensor."
                )
            source_data = source_data[item.index_range.start : item.index_range.end]

        return source_data

    def _convert_source_data(
        self,
        source_data: Any,
        data_type: DataType,
        language_type: LanguageConfig,
        item_name: str | None,
        import_source_path: str,
        tf_module: Any,
    ) -> Any:
        if data_type == DataType.LANGUAGE and language_type == LanguageConfig.STRING:
            return source_data

        if isinstance(source_data, tf_module.Tensor) or (
            hasattr(source_data, "numpy")
            and not isinstance(source_data, (dict, list, str, bytes))
        ):
            try:
                return source_data.numpy()
            except Exception as exc:
                self.logger.warning(
                    "Failed to convert tensor to numpy for %s.%s: %s. Using raw value.",
                    data_type.value,
                    item_name if item_name else "unknown",
                    exc,
                )
                return source_data

        if isinstance(source_data, dict):
            raise ImportError(
                "Expected tensor but got dict for "
                f"{data_type.value}.{item_name if item_name else 'unknown'}. "
                f"Source path '{import_source_path}' may be incorrect."
            )

        if isinstance(source_data, list) and source_data:
            if isinstance(source_data[0], tf_module.Tensor) or hasattr(
                source_data[0], "numpy"
            ):
                try:
                    return [
                        value.numpy() if hasattr(value, "numpy") else value
                        for value in source_data
                    ]
                except Exception:
                    return source_data

        return source_data


class RLDSDatasetImporter(RLDSAndTFDSDatasetImporterBase):
    """Importer for RLDS datasets."""

    dataset_label = "RLDS"

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        joint_info: dict[str, JointInfo] = {},
        ik_urdf_path: str | None = None,
        ik_init_config: list[float] | None = None,
        dry_run: bool = False,
        suppress_warnings: bool = False,
        skip_on_error: str = "episode",
    ):
        """Initialize the RLDS/TFDS dataset importer.

        Args:
            input_dataset_name: Name of the dataset to import.
            output_dataset_name: Name of the dataset to create.
            dataset_dir: Directory containing the dataset.
            dataset_config: Dataset configuration.
            joint_info: Joint info to use for validation.
            ik_urdf_path: URDF path for IK (used to recreate IK in worker processes).
            ik_init_config: Initial joint configuration for IK.
            dry_run: If True, skip actual logging (validation only).
            suppress_warnings: If True, suppress warning messages.
            skip_on_error: "episode" to skip a failed episode; "step" to skip only
                failing steps; "all" to abort on the first error.
        """
        super().__init__(
            input_dataset_name=input_dataset_name,
            output_dataset_name=output_dataset_name,
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            joint_info=joint_info,
            ik_urdf_path=ik_urdf_path,
            ik_init_config=ik_init_config,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
            max_workers=1,
            skip_on_error=skip_on_error,
        )

    def _record_step(self, step_data: dict, timestamp: float) -> None:
        """Record a single step to Neuracore.

        RLDS configs may omit source paths, so we preserve that behavior here.
        """
        for data_type, import_config in self.dataset_config.data_import_config.items():
            if not import_config.source:
                source: Any = step_data
            else:
                source_path = import_config.source.split(".")
                source = step_data
                for path in source_path:
                    source = source[path]

            for item in import_config.mapping:
                source_data = (
                    source[item.source_name] if item.source_name is not None else source
                )

                if item.index is not None:
                    source_data = source_data[item.index]
                elif item.index_range is not None:
                    source_data = source_data[
                        item.index_range.start : item.index_range.end
                    ]

                if not (
                    data_type == DataType.LANGUAGE
                    and import_config.format.language_type == LanguageConfig.STRING
                ):
                    source_data = source_data.numpy()

                self._log_data(
                    data_type, source_data, item, import_config.format, timestamp
                )


class TFDSDatasetImporter(RLDSAndTFDSDatasetImporterBase):
    """Importer for TFDS (TensorFlow Datasets) datasets."""

    dataset_label = "TFDS"
    allow_mapping_name_fallback = True

    def _on_builder_loaded(self, builder: tfds.core.DatasetBuilder) -> None:
        """Check for missing shards after builder load."""
        self._check_missing_shards()

    def _build_read_config(self) -> tfds.ReadConfig:
        """Build read config for TFDS datasets."""
        return tfds.ReadConfig(
            try_autocache=False,
            skip_prefetch=True,
        )

    def _build_retry_read_config(self) -> tfds.ReadConfig | None:
        """Build lenient retry read config for TFDS shard issues."""
        return tfds.ReadConfig(
            try_autocache=False,
            skip_prefetch=True,
            interleave_cycle_length=1,
        )

    def _handle_episode_load_error(self, exc: Exception, item: ImportItem) -> None:
        """Treat missing-file episode failures as skippable import errors."""
        if isinstance(exc, StopIteration):
            super()._handle_episode_load_error(exc, item)
            return
        if self._is_missing_file_error(exc):
            self.logger.warning(
                "[worker %s item %s] Skipping episode due to missing shard file: %s",
                self._worker_id if self._worker_id is not None else 0,
                item.index,
                exc,
            )
            raise ImportError(
                f"Episode {item.index} cannot be loaded due to missing shard "
                f"file: {exc}"
            ) from exc
        raise exc

    def _check_missing_shards(self) -> None:
        """Check for TFRecord shards and warn when none are found."""
        try:
            tfrecord_files = list(self.builder_dir.glob("*.tfrecord*"))
            if not tfrecord_files:
                for subdir in self.builder_dir.iterdir():
                    if subdir.is_dir():
                        tfrecord_files.extend(subdir.glob("*.tfrecord*"))
            if tfrecord_files:
                self.logger.info(
                    "Found %d TFRecord shard files in dataset directory",
                    len(tfrecord_files),
                )
            else:
                self.logger.warning(
                    "No TFRecord files found in dataset directory. "
                    "Dataset may be incomplete or use a different format."
                )
        except Exception as exc:  # noqa: BLE001 - informational check only
            self.logger.debug("Could not check for missing shards: %s", exc)
