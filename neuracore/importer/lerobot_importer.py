"""Importer scaffold for LeRobot datasets."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.nc_data import DatasetImportConfig

import neuracore as nc
from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImportError


class LeRobotDatasetImporter(NeuracoreDatasetImporter):
    """Importer for LeRobot datasets (Hugging Face based arrow format)."""

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
    ) -> None:
        """Initialize the LeRobot dataset importer.

        Args:
            input_dataset_name: Name of the dataset to import.
            output_dataset_name: Name of the dataset to create.
            dataset_dir: Directory containing the dataset.
            dataset_config: Dataset configuration.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=1,
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

        self.logger.info(
            "Initialized LeRobot importer for '%s' "
            "(episodes=%s, cameras=%s, fps=%s, root=%s)",
            self.dataset_name,
            self.num_episodes,
            self.camera_keys,
            self.frequency,
            self.dataset_root,
        )

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

    def upload(self, item: ImportItem) -> None:
        """Upload a single episode to the dataset importer."""
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
            self._record_step(step_data, timestamp)
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

    def _record_step(self, step_data: dict, timestamp: float) -> None:
        """Record a single step to Neuracore."""
        for data_type, import_config in self.dataset_config.data_import_config.items():
            # Get the data based on the source path
            source_path = import_config.source

            for item in import_config.mapping:
                if item.source_name is not None:
                    source_data = step_data[".".join([source_path, item.source_name])]
                elif item.index is not None:
                    source_data = step_data[source_path][item.index]
                elif item.index_range is not None:
                    source_data = step_data[source_path][
                        item.index_range.start : item.index_range.end
                    ]
                else:
                    source_data = step_data[source_path]

                if (
                    data_type == DataType.LANGUAGE
                    and import_config.format.language_type == LanguageConfig.STRING
                ):
                    transformed_data = source_data
                else:
                    transformed_data = item.transforms(source_data.numpy())
                self._log_data(data_type, item.name, transformed_data, timestamp)
