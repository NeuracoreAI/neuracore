"""Importer for RLDS/TFDS-style datasets."""

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tensorflow_datasets as tfds
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.nc_data import DatasetImportConfig

import neuracore as nc
from neuracore.core.robot import JointInfo
from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImportError

# Suppress TensorFlow informational messages (e.g., "End of sequence")
# 0 = all logs, 1 = no INFO, 2 = no WARNING, 3 = no ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


class RLDSDatasetImporter(NeuracoreDatasetImporter):
    """Importer for RLDS/TFDS-style datasets."""

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        joint_info: dict[str, JointInfo] = {},
        dry_run: bool = False,
        suppress_warnings: bool = False,
    ):
        """Initialize the RLDS/TFDS dataset importer.

        Args:
            input_dataset_name: Name of the dataset to import.
            output_dataset_name: Name of the dataset to create.
            dataset_dir: Directory containing the dataset.
            dataset_config: Dataset configuration.
            joint_info: Joint info to use for validation.
            dry_run: If True, skip actual logging (validation only).
            suppress_warnings: If True, suppress warning messages.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=1,
            joint_info=joint_info,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
        )
        self.dataset_name = input_dataset_name
        self.builder_dir = self._resolve_builder_dir()
        if self.frequency is None:
            raise ImportError("Dataset frequency is required for RLDS imports.")

        builder = self._load_builder()
        self.split = self._pick_split(builder)
        self.num_episodes = self._count_episodes(builder, self.split)

        self._builder: tfds.core.DatasetBuilder | None = None
        self._episode_iter = None

        self.logger.info(
            "Dataset ready: name=%s split=%s episodes=%s freq=%s dir=%s",
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

        dataset = self._load_dataset(self._builder, self.split)
        if chunk_start:
            dataset = dataset.skip(chunk_start)
        if chunk_length is not None:
            dataset = dataset.take(chunk_length)
        self._episode_iter = iter(dataset)

    def upload(self, item: ImportItem) -> None:
        """Import a single episode to the dataset importer."""
        if self._episode_iter is None:
            raise ImportError("Worker dataset iterator was not initialized.")

        try:
            episode = next(self._episode_iter)
        except StopIteration as exc:  # noqa: PERF203 - surface worker exhaustion
            raise ImportError(
                f"No episode available for index {item.index} "
                f"(dataset has {self.num_episodes} episodes)."
            ) from exc

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
            "[%s] Importing episode %s (%s/%s)",
            worker_label,
            episode_label,
            item.index + 1,
            self.num_episodes,
        )
        self._emit_progress(
            item.index, step=0, total_steps=total_steps, episode_label=episode_label
        )
        for idx, step in enumerate(steps, start=1):
            timestamp = base_time + (idx / self.frequency)
            self._record_step(step, timestamp)
            self._emit_progress(
                item.index,
                step=idx,
                total_steps=total_steps,
                episode_label=episode_label,
            )
        nc.stop_recording(wait=True)
        self.logger.info("[%s] Completed episode %s", worker_label, episode_label)

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
        self.logger.info("Loading RLDS builder from %s", self.builder_dir)
        try:
            return tfds.builder_from_directory(str(self.builder_dir))
        except Exception as exc:  # noqa: BLE001 - surface clear failure
            raise ImportError(
                f"Failed to load RLDS builder from '{self.builder_dir}': {exc}"
            ) from exc

    def _pick_split(self, builder: tfds.core.DatasetBuilder) -> tfds.typing.SplitArg:
        """Select split to inspect; default to all splits."""
        splits = list(builder.info.splits.keys())
        if not splits:
            raise ImportError(
                f"No splits found in RLDS dataset at '{self.builder_dir}'."
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

    def _load_dataset(
        self, builder: tfds.core.DatasetBuilder, split: tfds.typing.SplitArg
    ) -> tfds.core.dataset_builder.DatasetBuilder:
        """Load the TFDS dataset from the local builder."""
        self.logger.info("Opening dataset split '%s' for import.", split)
        try:
            return builder.as_dataset(
                split=split,
                shuffle_files=False,
                read_config=tfds.ReadConfig(try_autocache=False),
            )
        except Exception as exc:  # noqa: BLE001 - surface clear failure
            raise ImportError(
                f"Failed to load RLDS dataset split '{split}': {exc}"
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
        for data_type, import_config in self.dataset_config.data_import_config.items():
            # Get the data based on the source path
            source_path = import_config.source.split(".")
            source = step_data
            for path in source_path:
                source = source[path]

            for item in import_config.mapping:
                if item.source_name is not None:
                    source_data = source[item.source_name]
                elif item.index is not None:
                    source_data = source[item.index]
                elif item.index_range is not None:
                    source_data = source[item.index_range.start : item.index_range.end]
                else:
                    source_data = source

                if not (
                    data_type == DataType.LANGUAGE
                    and import_config.format.language_type == LanguageConfig.STRING
                ):
                    source_data = source_data.numpy()

                self._log_data(
                    data_type, source_data, item, import_config.format, timestamp
                )
