"""MCAP dataset importer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from mcap.decoder import DecoderFactory
from mcap.reader import make_reader
from neuracore_types import DataType
from neuracore_types.nc_data import DatasetImportConfig

import neuracore as nc
from neuracore.core.robot import JointInfo
from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.mcap.utils import (
    build_topic_map,
    clip_depth,
    convert_decoded_mcap_data,
    estimate_total_messages,
    get_mcap_topics,
    iter_decoded_mcap_messages,
    iter_mcap_source_events,
    list_decoder_factories,
    log_mcap_header,
    log_mcap_summary_details,
    read_mcap_header,
    read_mcap_summary,
    validate_channel_decoder_support,
    validate_requested_topics,
)


class MCAPDatasetImporter(NeuracoreDatasetImporter):
    """Importer for MCAP datasets."""

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        joint_info: dict[str, JointInfo] = {},
        urdf_path: str | None = None,
        ik_init_config: list[float] | None = None,
        dry_run: bool = False,
        suppress_warnings: bool = False,
        max_workers: int | None = 1,
        skip_on_error: str = "episode",
        storage_limit: int = 5 * 1024**3,
        random_sample: int | None = None,
        shared: bool = False,
    ) -> None:
        """Initialize the MCAP dataset importer."""
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=max_workers,
            skip_on_error=skip_on_error,
            joint_info=joint_info,
            urdf_path=urdf_path,
            ik_init_config=ik_init_config,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
            storage_limit=storage_limit,
            random_sample=random_sample,
            shared=shared,
        )
        if max_workers is not None and max_workers > 1:
            self.logger.warning(
                f"MCAP import is configured with {max_workers} workers. Each MCAP "
                "file is streamed as one episode, so memory use remains bounded per "
                "worker.",
            )

        self.dataset_name = input_dataset_name
        self.dataset_dir = Path(dataset_dir)
        self.topic_map = build_topic_map(dataset_config=dataset_config)
        self.mcap_files = self._discover_mcap_files(dataset_dir=self.dataset_dir)
        self._decoder_factories: list[DecoderFactory] | None = None
        self._init_runtime_components()

        self.logger.info(
            f"Initialized MCAP importer for '{self.dataset_name}' "
            f"(files={len(self.mcap_files)}, "
            f"topics={len(get_mcap_topics(topic_map=self.topic_map))}, "
            f"root={self.dataset_dir})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state with decoder factories cleared."""
        # TODO: Replace this with lazy worker-local decoder factory initialization
        # so runtime decoder state is never stored on the importer before forking.
        state = self.__dict__.copy()
        state["_decoder_factories"] = None
        return state

    def build_work_items(self) -> Sequence[ImportItem]:
        """Return one work item per discovered MCAP file."""
        return [
            ImportItem(index=i, description=path.name, metadata={"path": str(path)})
            for i, path in enumerate(self.mcap_files)
        ]

    def prepare_worker(
        self,
        worker_id: int,
        chunk: Sequence[ImportItem] | None = None,
    ) -> None:
        """Initialize per-worker decoder factories after forking."""
        super().prepare_worker(worker_id=worker_id, chunk=chunk)
        self._init_runtime_components()

    def import_item(self, item: ImportItem) -> None:
        """Import one MCAP file."""
        self._ensure_runtime_components()
        self._reset_episode_state()

        file_path_raw = (item.metadata or {}).get("path")
        file_path = Path(file_path_raw) if file_path_raw else None
        if file_path is None or not file_path.exists():
            raise ImportError(f"MCAP file not found for item {item.index}.")

        label = item.description or file_path.name
        instance = max(0, self._worker_id)
        self.logger.info(
            f"Importing MCAP file {label} ({item.index + 1}/{len(self.mcap_files)})"
        )

        if not self.dry_run:
            nc.start_recording(robot_name=self.robot_name, instance=instance)
        try:
            message_count = self._stream_episode_file(
                episode_file_path=file_path,
                item=item,
                label=label,
            )
        finally:
            if not self.dry_run:
                nc.stop_recording(
                    robot_name=self.robot_name, instance=instance, wait=True
                )

        self.logger.info(f"Completed MCAP file {label} | messages={message_count}")

    def _record_step(self, step: dict, timestamp: float) -> None:
        """Log decoded data from each MCAP source topic in this step."""
        for topic, decoded_data in step.items():
            for event in iter_mcap_source_events(
                topic=topic,
                decoded_data=decoded_data,
                topic_map=self.topic_map,
                logger=self.logger,
                timestamp=timestamp,
            ):
                self._log_data(
                    data_type=event.data_type,
                    source_data=event.source_data,
                    item=event.item,
                    format=event.format,
                    timestamp=event.timestamp,
                )

    def _stream_episode_file(
        self, episode_file_path: Path, item: ImportItem, label: str
    ) -> int:
        """Stream messages from one MCAP episode file."""
        topics = get_mcap_topics(topic_map=self.topic_map)
        factories = list(self._decoder_factories or [])
        base_timestamp: float | None = None
        message_count = 0

        with episode_file_path.open("rb") as stream:
            reader = make_reader(stream=stream, decoder_factories=factories)
            header = read_mcap_header(reader=reader)
            log_mcap_header(header=header, logger=self.logger)
            summary = read_mcap_summary(reader=reader)
            log_mcap_summary_details(summary=summary, logger=self.logger)
            validate_requested_topics(summary=summary, topics=topics)
            validate_channel_decoder_support(
                summary=summary,
                topics=topics,
                decoder_factories=factories,
                logger=self.logger,
            )
            total = estimate_total_messages(summary=summary, topics=topics)

            for decoded_message in iter_decoded_mcap_messages(
                reader=reader,
                topics=topics,
            ):
                if base_timestamp is None:
                    base_timestamp = decoded_message.timestamp_seconds
                timestamp = max(0.0, decoded_message.timestamp_seconds - base_timestamp)
                decoded_data = convert_decoded_mcap_data(
                    decoded_data=decoded_message.data
                )
                self._record_step(
                    step={decoded_message.topic: decoded_data},
                    timestamp=timestamp,
                )
                message_count += 1
                if message_count % 100 == 0:
                    self._emit_progress(
                        item_index=item.index,
                        step=message_count,
                        total_steps=total,
                        episode_label=label,
                    )

        self._emit_progress(
            item_index=item.index,
            step=message_count,
            total_steps=total,
            episode_label=label,
        )
        return message_count

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
        """Clip depth arrays before delegating to the base logging path."""
        if data_type == DataType.DEPTH_IMAGES:
            transformed_data = clip_depth(data=transformed_data, logger=self.logger)
        super()._log_transformed_data(
            data_type=data_type,
            transformed_data=transformed_data,
            name=name,
            timestamp=timestamp,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

    def _init_runtime_components(self) -> None:
        self._decoder_factories = list_decoder_factories(logger=self.logger)

    def _ensure_runtime_components(self) -> None:
        if self._decoder_factories is None:
            self._init_runtime_components()

    @staticmethod
    def _discover_mcap_files(dataset_dir: Path) -> list[Path]:
        if dataset_dir.is_file():
            if dataset_dir.suffix.lower() != ".mcap":
                raise ImportError(
                    f"Expected an MCAP file, got '{dataset_dir.name}' instead."
                )
            return [dataset_dir]

        if not dataset_dir.exists():
            raise ImportError(f"Dataset path does not exist: {dataset_dir}")

        mcap_files = sorted(dataset_dir.rglob("*.mcap"))
        if not mcap_files:
            raise ImportError(
                f"No MCAP files found under '{dataset_dir}'. "
                "Provide a .mcap file or a directory containing MCAP files."
            )
        return mcap_files
