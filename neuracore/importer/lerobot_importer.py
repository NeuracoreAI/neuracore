"""Importer scaffold for LeRobot datasets."""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
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


class LeRobotDatasetImporter(NeuracoreDatasetImporter):
    """Importer for LeRobot datasets (Hugging Face based arrow format)."""

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
    ) -> None:
        """Initialize the LeRobot dataset importer.

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
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=1,
            joint_info=joint_info,
            ik_urdf_path=ik_urdf_path,
            ik_init_config=ik_init_config,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
            skip_on_error=skip_on_error,
        )
        self.dataset_name = input_dataset_name
        self.dataset_dir = Path(dataset_dir)
        self.dataset_root = self.dataset_dir

        meta = self._load_metadata()
        self.num_episodes = meta.total_episodes
        self.camera_keys = list(meta.camera_keys)
        self.frequency = self._resolve_frequency(meta.fps)

        self._dataset: LeRobotDataset | None = None
        self._episode_iter: Iterator[int] | None = None

    def __getstate__(self) -> dict:
        """Drop worker-local handles when pickling for multiprocessing."""
        state = self.__dict__.copy()
        state.pop("_dataset", None)
        state.pop("_episode_iter", None)
        return state

    def build_work_items(self) -> Sequence[ImportItem]:
        """Build work items for the dataset importer."""
        return [ImportItem(index=i) for i in range(self.num_episodes)]

    def prepare_worker(
        self, worker_id: int, chunk: Sequence[ImportItem] | None = None
    ) -> None:
        """Prepare the worker for the dataset importer."""
        super().prepare_worker(worker_id, chunk)
        self._dataset = self._load_dataset()
        episode_ids = self._collect_episode_ids(self._dataset)
        start = chunk[0].index if chunk else 0
        end = start + len(chunk) if chunk else len(episode_ids)
        self._episode_iter = iter(episode_ids[start:end])

    def import_item(self, item: ImportItem) -> None:
        """Import a single episode to the dataset importer."""
        self._reset_episode_state()
        if self._dataset is None or self._episode_iter is None:
            raise ImportError("Worker dataset was not initialized.")

        try:
            episode_id = next(self._episode_iter)
        except StopIteration as exc:  # noqa: PERF203
            raise ImportError(
                f"No episode available for index {item.index} "
                f"(dataset has {self.num_episodes} episodes)."
            ) from exc

        if self.frequency is None:
            raise ImportError("Frequency is required for importing episodes.")
        base_time = time.time()
        worker_label = (
            f"worker {self._worker_id}" if self._worker_id is not None else "worker 0"
        )
        self.logger.info(
            "[%s] Importing episode %s (%s/%s)",
            worker_label,
            episode_id,
            item.index + 1,
            self.num_episodes,
        )
        nc.start_recording()
        step_iter, total_steps = self._iter_episode_steps(self._dataset, episode_id)
        self._emit_progress(
            item.index, step=0, total_steps=total_steps, episode_label=str(episode_id)
        )
        for step_idx, step_data in enumerate(step_iter, start=1):
            timestamp = base_time + (step_idx / self.frequency)
            try:
                self._record_step(step_data, timestamp)
            except Exception as exc:  # noqa: BLE001
                if self.skip_on_error == "step":
                    if self._error_queue is not None:
                        self._error_queue.put(
                            WorkerError(
                                worker_id=self._worker_id or 0,
                                item_index=item.index,
                                message=f"Step {step_idx}: {exc}",
                                traceback=traceback.format_exc(),
                            )
                        )
                    self._log_worker_error(
                        self._worker_id or 0, item.index, f"Step {step_idx}: {exc}"
                    )
                    continue
                raise
            self._emit_progress(
                item.index,
                step=step_idx,
                total_steps=total_steps,
                episode_label=str(episode_id),
            )
        nc.stop_recording(wait=True)
        self.logger.info("[%s] Completed episode %s", worker_label, episode_id)

    def _load_metadata(self) -> LeRobotDatasetMetadata:
        """Fetch metadata without pulling down the entire dataset."""
        ds_meta = LeRobotDatasetMetadata(self.dataset_name, root=self.dataset_root)
        self.logger.info(
            "Dataset metadata loaded: name=%s root=%s episodes=%s "
            "camera_keys=%s fps=%s",
            self.dataset_name,
            self.dataset_root,
            ds_meta.total_episodes,
            ds_meta.camera_keys,
            ds_meta.fps,
        )
        return ds_meta

    def _resolve_frequency(self, meta_frequency: float | None) -> float:
        """Pick the frequency from config or dataset metadata."""
        if self.data_config.frequency is not None:
            if meta_frequency and meta_frequency != self.data_config.frequency:
                self.logger.warning(
                    "Dataset FPS %s does not match configured FPS %s",
                    meta_frequency,
                    self.data_config.frequency,
                )
            return self.data_config.frequency
        if meta_frequency is None:
            raise ImportError(
                "Frequency not provided in config and missing from metadata."
            )
        return float(meta_frequency)

    def _load_dataset(self) -> LeRobotDataset:
        """Load the actual dataset."""
        self.logger.info(
            "Loading LeRobot dataset '%s' from %s", self.dataset_name, self.dataset_root
        )
        try:
            return LeRobotDataset(self.dataset_name, root=self.dataset_root)
        except Exception as exc:  # noqa: BLE001 - provide clear context
            raise ImportError(
                f"Failed to load LeRobot dataset '{self.dataset_name}' "
                f"from '{self.dataset_root}': {exc}"
            ) from exc

    def _collect_episode_ids(self, ds: LeRobotDataset) -> list[int]:
        """Return sorted episode ids present in the dataset."""
        return sorted({int(ep) for ep in ds.hf_dataset["episode_index"]})

    def _iter_episode_steps(
        self, ds: LeRobotDataset, episode_id: int
    ) -> tuple[Iterable[dict], int]:
        """Yield step dictionaries for a single episode along with step count."""
        ep_rows = ds.hf_dataset.filter(
            lambda row, target=episode_id: row["episode_index"] == target
        ).sort("frame_index")
        total_steps = len(ep_rows)
        if "index" in ep_rows.column_names:
            indices = [int(i) for i in ep_rows["index"]]
            return (ds[i] for i in indices), total_steps
        return (ep_rows[i] for i in range(total_steps)), total_steps

    def _extract_source_data(
        self,
        source: Any,
        item: Any,
        import_source_path: str,
        data_type: DataType,
    ) -> Any:
        if not import_source_path:
            if item.source_name is not None:
                source_data = source[item.source_name]
            else:
                source_data = source
        else:
            if item.source_name is not None:
                source_data = source[".".join([import_source_path, item.source_name])]
            else:
                source_data = source[import_source_path]

        try:

            if item.index is not None:
                source_data = source_data[item.index]
            elif item.index_range is not None:
                source_data = source_data[item.index_range.start : item.index_range.end]
        except Exception as exc:
            shape_str = (
                f" with shape {source_data.shape}"
                if hasattr(source_data, "shape")
                else ""
            )
            raise ImportError(
                f"Cannot index or slice for '{data_type.value}'. "
                f"Source path '{import_source_path}' resolved to a "
                f"{type(source_data)}{shape_str}, not an indexable tensor. "
                f"Check your dataset config. {exc}"
            )

        return source_data

    def _convert_source_data(
        self,
        source_data: Any,
        data_type: DataType,
        language_type: LanguageConfig,
        item_name: str | None,
        import_source_path: str,
    ) -> Any:
        if data_type == DataType.LANGUAGE and language_type == LanguageConfig.STRING:
            return source_data

        try:
            return source_data.numpy()
        except Exception as exc:
            suffix = f".{item_name}" if item_name else ""
            raise ImportError(
                f"Failed to convert data to numpy array for "
                f"{data_type.value}{suffix}: {exc}."
            )

    def _record_step(self, step_data: dict, timestamp: float) -> None:
        """Record a single step to Neuracore."""
        for data_type, import_config in self.dataset_config.data_import_config.items():
            # Get the data based on the source path

            for item in import_config.mapping:

                source_data = self._extract_source_data(
                    source=step_data,
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
                )

                self._log_data(
                    data_type, source_data, item, import_config.format, timestamp
                )
