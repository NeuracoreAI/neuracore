"""MCAP dataset importer."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader
from neuracore_types import DataType
from neuracore_types.nc_data import DatasetImportConfig

import neuracore as nc
from neuracore.core.robot import JointInfo
from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.mcap.utils import (
    H264StreamDecoder,
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
    topic_schema_names,
    validate_channel_decoder_support,
    validate_requested_topics,
)


class MCAPDatasetImporter(NeuracoreDatasetImporter):
    """Importer for MCAP datasets."""

    def __init__(
        self,
        input_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        output_dataset_id: str,
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
        debug_target_ee_frame: str | None = None,
    ) -> None:
        """Initialize the MCAP dataset importer."""
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_id=output_dataset_id,
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
            debug_target_ee_frame=debug_target_ee_frame,
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
        # Per-episode H.264 decoders, keyed by topic. Video streams are
        # inter-frame, so each topic needs one decoder held for the whole file.
        self._video_decoders: dict[str, H264StreamDecoder] = {}

        self.logger.info(
            f"Initialized MCAP importer for '{self.dataset_name}' "
            f"(files={len(self.mcap_files)}, "
            f"topics={len(get_mcap_topics(topic_map=self.topic_map))}, "
            f"root={self.dataset_dir})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state with runtime decoder state cleared."""
        state = self.__dict__.copy()
        state["_video_decoders"] = {}
        return state

    def build_work_items(self) -> Sequence[ImportItem]:
        """Return one work item per discovered MCAP file."""
        return [
            ImportItem(index=i, description=path.name, metadata={"path": str(path)})
            for i, path in enumerate(self.mcap_files)
        ]

    def validate_work_items(self, items: Sequence[ImportItem]) -> None:
        """Reject a dataset whose episodes record different types on one topic.

        Compare every episode against the first on the message type of each
        mapped topic, reading only the summary section. Schema ids are file
        local and are deliberately not compared.

        Args:
            items: The work items selected for import.

        Raises:
            ImportError: If two episodes disagree on a mapped topic's type.
        """
        topics = get_mcap_topics(topic_map=self.topic_map)
        if not topics:
            return

        baseline: dict[str, str] = {}
        baseline_label = ""
        mismatches: list[str] = []

        for item in items:
            file_path_raw = (item.metadata or {}).get("path")
            if not file_path_raw:
                continue
            file_path = Path(file_path_raw)
            if not file_path.exists():
                continue
            with file_path.open("rb") as stream:
                summary = read_mcap_summary(reader=make_reader(stream=stream))
            schema_names = topic_schema_names(summary=summary, topics=topics)
            if not schema_names:
                continue
            if not baseline:
                baseline = schema_names
                baseline_label = file_path.name
                continue
            for topic in sorted(schema_names):
                expected = baseline.get(topic)
                if expected is not None and expected != schema_names[topic]:
                    mismatches.append(
                        f"  {topic}: '{expected}' in {baseline_label}, "
                        f"'{schema_names[topic]}' in {file_path.name}"
                    )

        if mismatches:
            raise ImportError(
                f"MCAP episodes disagree on the message type of "
                f"{len(mismatches)} topic(s). Every episode must record the same "
                "message type on a given topic:\n" + "\n".join(mismatches)
            )

        self.logger.debug(
            f"Schema check passed across {len(items)} MCAP episode(s) "
            f"for {len(topics)} topic(s)."
        )

    def import_item(self, item: ImportItem) -> None:
        """Import one MCAP file."""
        self._reset_episode_state()

        file_path_raw = (item.metadata or {}).get("path")
        file_path = Path(file_path_raw) if file_path_raw else None
        if file_path is None or not file_path.exists():
            raise ImportError(f"MCAP file not found for item {item.index}.")

        label = item.description or file_path.name
        instance = self.robot_instance(self._worker_id)
        self.logger.info(
            f"Importing MCAP file {label} ({item.index + 1}/{len(self.mcap_files)})"
        )

        recording_start_timestamp = time.time()
        recording_stop_timestamp = recording_start_timestamp

        if not self.dry_run:
            nc.start_recording(
                robot_name=self.robot_name,
                instance=instance,
                timestamp=recording_start_timestamp,
            )
        try:
            message_count, recording_stop_timestamp = self._stream_episode_file(
                episode_file_path=file_path,
                item=item,
                label=label,
                recording_start_timestamp=recording_start_timestamp,
            )
        finally:
            if not self.dry_run:
                nc.stop_recording(
                    robot_name=self.robot_name,
                    instance=instance,
                    wait=True,
                    timestamp=recording_stop_timestamp,
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
                video_decoders=self._video_decoders,
            ):
                self._log_data(
                    data_type=event.data_type,
                    source_data=event.source_data,
                    item=event.item,
                    format=event.format,
                    timestamp=event.timestamp,
                )

    def _stream_episode_file(
        self,
        episode_file_path: Path,
        item: ImportItem,
        label: str,
        recording_start_timestamp: float,
    ) -> tuple[int, float]:
        """Stream messages from one MCAP episode file."""
        topics = get_mcap_topics(topic_map=self.topic_map)
        # Fresh decoder factories per episode. mcap decoder factories cache
        # generated message classes by schema id, and schema ids are file local,
        # so a factory reused across files decodes with the wrong class.
        factories = list_decoder_factories(logger=self.logger)
        source_start_timestamp_ns: int | None = None
        recording_stop_timestamp = recording_start_timestamp
        message_count = 0
        # Fresh video decoders per episode: state must never carry across files.
        self._video_decoders = {}

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
                if source_start_timestamp_ns is None:
                    source_start_timestamp_ns = decoded_message.timestamp_ns

                relative_timestamp_ns = max(
                    0, decoded_message.timestamp_ns - source_start_timestamp_ns
                )
                timestamp = recording_start_timestamp + relative_timestamp_ns / 1e9
                recording_stop_timestamp = max(recording_stop_timestamp, timestamp)

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
        self._log_video_decoder_summary(label=label)
        self._video_decoders = {}
        return message_count, recording_stop_timestamp

    def _log_video_decoder_summary(self, label: str) -> None:
        """Report frames dropped before each video stream's first keyframe."""
        dropped = {
            topic: decoder.skipped_before_keyframe
            for topic, decoder in self._video_decoders.items()
            if decoder.skipped_before_keyframe
        }
        if dropped:
            self.logger.info(
                f"{label}: dropped leading frames before the first keyframe "
                f"per video topic: {dropped}"
            )

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
