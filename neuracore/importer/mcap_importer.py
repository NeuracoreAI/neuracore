"""MCAP dataset importer.

This importer uses a two-phase architecture:
1) preprocess MCAP messages to a disk cache
2) replay cached events into Neuracore with TTL-aware recording session rotation
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from mcap.decoder import DecoderFactory
from neuracore_types import DataType
from neuracore_types.nc_data import DatasetImportConfig

from neuracore.core.robot import JointInfo
from neuracore.importer.core.base import (
    ImportItem,
    NeuracoreDatasetImporter,
    WorkerError,
)
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.utils.mcap.config import MCAPImportConfig, SkipOnError
from neuracore.importer.utils.mcap.decoder import (
    ImageDecoder,
    MCAPMessageDecoder,
    list_decoder_factories,
)
from neuracore.importer.utils.mcap.logger import MessageLogger
from neuracore.importer.utils.mcap.preprocessor import MessagePreprocessor
from neuracore.importer.utils.mcap.progress import (
    EmitProgressReporter,
    LoggingProgressReporter,
    NullProgressReporter,
    ProgressReporter,
)
from neuracore.importer.utils.mcap.session import RecordingSession
from neuracore.importer.utils.mcap.topics import TopicMapper


class MCAPDatasetImporter(NeuracoreDatasetImporter):
    """Importer for MCAP datasets with TTL-safe replay logging."""

    CACHE_METADATA_VERSION = 1
    CACHE_METADATA_SUFFIX = ".meta.json"

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        joint_info: dict[str, JointInfo] = {},
        dry_run: bool = False,
        suppress_warnings: bool = False,
        *,
        max_workers: int | None = 1,
        skip_on_error: str = "episode",
    ) -> None:
        """Initialize importer state and runtime decoder components."""
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=max_workers,
            skip_on_error=skip_on_error,
            joint_info=joint_info,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
        )

        self.dataset_name = input_dataset_name
        self.dataset_dir = Path(dataset_dir)
        self.config = MCAPImportConfig.from_env(
            skip_on_error=cast(SkipOnError, skip_on_error)
        )
        self.topic_mapper = TopicMapper(dataset_config)
        self.mcap_files = self._discover_mcap_files(self.dataset_dir)
        self._dataset_config_hash = self._compute_dataset_config_hash(dataset_config)

        self._decoder_factories: list[DecoderFactory] | None = None
        self._image_decoder: ImageDecoder | None = None
        self._message_decoder: MCAPMessageDecoder | None = None
        self._init_runtime_components()

        self.logger.info(
            "Initialized MCAP importer for '%s' (files=%s, topics=%s, root=%s)",
            self.dataset_name,
            len(self.mcap_files),
            len(self.topic_mapper.get_all_topics()),
            self.dataset_dir,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Drop runtime-only decoder handles for worker pickling."""
        state = self.__dict__.copy()
        state["_decoder_factories"] = None
        state["_image_decoder"] = None
        state["_message_decoder"] = None
        return state

    def build_work_items(self) -> Sequence[ImportItem]:
        """Build one import item per MCAP file."""
        return [
            ImportItem(index=i, description=path.name, metadata={"path": str(path)})
            for i, path in enumerate(self.mcap_files)
        ]

    def prepare_worker(
        self,
        worker_id: int,
        chunk: Sequence[ImportItem] | None = None,
    ) -> None:
        """Initialize worker Neuracore context and runtime decoder components."""
        super().prepare_worker(worker_id, chunk)
        self._init_runtime_components()

    def import_item(self, item: ImportItem) -> None:
        """Import one MCAP file via preprocessing + replay logging phases."""
        self._ensure_runtime_components()

        file_path_raw = (item.metadata or {}).get("path")
        file_path = Path(file_path_raw) if file_path_raw else None
        if file_path is None or not file_path.exists():
            raise ImportError(f"MCAP file not found for item {item.index}.")

        worker_id = self._worker_id if self._worker_id is not None else 0
        worker_label = f"worker {worker_id}"
        label = item.description or file_path.name
        self.logger.info(
            "[%s] Importing MCAP file %s (%s/%s)",
            worker_label,
            label,
            item.index + 1,
            len(self.mcap_files),
        )

        cache_path = self._get_cache_path(file_path)
        cache_metadata_path = self._get_cache_metadata_path(cache_path)
        preserve_cache_for_retry = False

        try:
            expected_total_messages = self._get_reusable_cache_total(
                mcap_file=file_path,
                cache_path=cache_path,
                cache_metadata_path=cache_metadata_path,
            )
            if expected_total_messages is None:
                expected_total_messages = self._preprocess_item_to_cache(
                    item=item,
                    label=label,
                    mcap_file=file_path,
                    cache_path=cache_path,
                    cache_metadata_path=cache_metadata_path,
                )
            else:
                self.logger.info(
                    "Reusing preprocessed cache for %s: %s",
                    label,
                    cache_path,
                )

            if expected_total_messages == 0:
                self.logger.warning("No replay events generated for %s", label)
                return

            logging_stats = None
            for replay_attempt in range(2):
                logging_reporter = self._create_progress_reporter(item, label)
                session = RecordingSession(
                    self.output_dataset_name,
                    logger=self.logger,
                    rotation_interval_seconds=self.config.SESSION_ROTATION_SECONDS,
                )
                message_logger = MessageLogger(
                    session=session,
                    data_logger=self._log_transformed_data,
                    progress_reporter=logging_reporter,
                    logger=self.logger,
                    on_event_error=lambda event_index, topic, name, log_time_ns, exc: (
                        self._handle_phase_error(
                            item=item,
                            phase="logging",
                            unit_label="event",
                            unit_index=event_index,
                            topic=topic,
                            name=name,
                            log_time_ns=log_time_ns,
                            exc=exc,
                        )
                    ),
                    max_replay_bytes_per_second=self.config.REPLAY_MAX_BYTES_PER_SECOND,
                )
                preserve_cache_for_retry = True
                try:
                    logging_stats = message_logger.log_from_cache(
                        cache_path,
                        expected_total_messages=expected_total_messages,
                    )
                except FileNotFoundError as exc:
                    if replay_attempt == 0 and self._is_missing_cache_file(
                        exc, cache_path
                    ):
                        self.logger.warning(
                            "Cache disappeared before replay for %s; "
                            "rebuilding and retrying once: %s",
                            label,
                            cache_path,
                        )
                        expected_total_messages = self._preprocess_item_to_cache(
                            item=item,
                            label=label,
                            mcap_file=file_path,
                            cache_path=cache_path,
                            cache_metadata_path=cache_metadata_path,
                        )
                        if expected_total_messages == 0:
                            self.logger.warning(
                                "No replay events generated for %s", label
                            )
                            preserve_cache_for_retry = False
                            return
                        continue
                    raise
                else:
                    preserve_cache_for_retry = False
                    break

            if logging_stats is None:
                raise ImportError(f"Failed to replay cached events for {label}.")

            self.logger.info(
                "[%s] Completed MCAP file %s | events=%s sessions=%s duration=%.2fs",
                worker_label,
                label,
                logging_stats.message_count,
                logging_stats.session_count,
                logging_stats.duration_seconds,
            )
        finally:
            if preserve_cache_for_retry:
                self.logger.warning(
                    "Logging failed for %s. Keeping preprocessed cache for retry: %s",
                    label,
                    cache_path,
                )
            else:
                self._cleanup_cache_artifacts(cache_path, cache_metadata_path)

    def upload_all(self) -> None:
        """Compatibility shim for existing callers."""
        self.import_all()

    def _record_step(self, step: dict, timestamp: float) -> None:
        """Record a pre-decoded step dictionary (used by base importer contracts)."""
        self._ensure_runtime_components()
        message_decoder = self._message_decoder
        if message_decoder is None:
            raise RuntimeError("MCAP runtime components were not initialized.")

        if not isinstance(step, dict):
            raise ImportError("MCAP payload must be a dict of topic->message data.")

        for topic, decoded_payload in step.items():
            for event in message_decoder.iter_transformed_messages(
                topic,
                decoded_payload,
                timestamp=timestamp,
                log_time_ns=0,
            ):
                self._log_transformed_data(
                    DataType(event.data_type),
                    event.transformed_data,
                    event.name,
                    event.timestamp,
                )

    def _init_runtime_components(self) -> None:
        """Initialize decoder and transform components."""
        self._decoder_factories = list_decoder_factories(
            enable_discovery=self.config.enable_decoder_discovery,
            logger=self.logger,
        )
        self._image_decoder = ImageDecoder(self.logger)
        self._message_decoder = MCAPMessageDecoder(
            topic_mapper=self.topic_mapper,
            prepare_log_data=self._prepare_log_data,
            image_decoder=self._image_decoder,
        )

    def _ensure_runtime_components(self) -> None:
        """Ensure runtime components are initialized after process deserialization."""
        if (
            self._decoder_factories is None
            or self._image_decoder is None
            or self._message_decoder is None
        ):
            self._init_runtime_components()

    @staticmethod
    def _discover_mcap_files(dataset_dir: Path) -> list[Path]:
        """Find MCAP files from a single file path or recursively in a directory."""
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

    def _get_cache_path(self, mcap_file: Path) -> Path:
        """Choose a deterministic cache path so failed logging can be retried."""
        if self.config.stage_dir is not None:
            cache_root = self.config.stage_dir
        else:
            cache_root = Path(tempfile.gettempdir()) / "neuracore_mcap_cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        identity = str(mcap_file.resolve()).encode("utf-8")
        suffix = hashlib.sha1(identity).hexdigest()[:12]
        return cache_root / f"{mcap_file.stem}_{suffix}.msgpack"

    def _get_cache_metadata_path(self, cache_path: Path) -> Path:
        return Path(f"{cache_path}{self.CACHE_METADATA_SUFFIX}")

    def _compute_dataset_config_hash(self, dataset_config: DatasetImportConfig) -> str:
        try:
            payload: Any = dataset_config.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            payload = repr(dataset_config)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_cache_fingerprint(self, mcap_file: Path) -> dict[str, Any]:
        stat = mcap_file.stat()
        return {
            "mcap_path": str(mcap_file.resolve()),
            "mcap_size_bytes": int(stat.st_size),
            "mcap_mtime_ns": int(stat.st_mtime_ns),
            "dataset_config_hash": self._dataset_config_hash,
            "skip_on_error": self.config.skip_on_error,
            "enable_decoder_discovery": self.config.enable_decoder_discovery,
        }

    def _get_reusable_cache_total(
        self,
        *,
        mcap_file: Path,
        cache_path: Path,
        cache_metadata_path: Path,
    ) -> int | None:
        if not cache_path.exists() or not cache_metadata_path.exists():
            return None

        try:
            metadata = json.loads(cache_metadata_path.read_text())
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Ignoring unreadable MCAP cache metadata at %s: %s",
                cache_metadata_path,
                exc,
            )
            return None

        if metadata.get("version") != self.CACHE_METADATA_VERSION:
            return None
        if metadata.get("fingerprint") != self._build_cache_fingerprint(mcap_file):
            return None

        event_count = metadata.get("event_count")
        if not isinstance(event_count, int) or event_count < 0:
            return None
        return event_count

    @staticmethod
    def _is_missing_cache_file(exc: FileNotFoundError, cache_path: Path) -> bool:
        filename = getattr(exc, "filename", None)
        if filename is None:
            return False
        return Path(filename) == cache_path

    def _preprocess_item_to_cache(
        self,
        *,
        item: ImportItem,
        label: str,
        mcap_file: Path,
        cache_path: Path,
        cache_metadata_path: Path,
    ) -> int:
        self._ensure_runtime_components()
        message_decoder = self._message_decoder
        decoder_factories = self._decoder_factories
        if message_decoder is None or decoder_factories is None:
            raise RuntimeError("MCAP runtime components were not initialized.")

        preprocess_reporter = self._create_progress_reporter(item, label)
        preprocessor = MessagePreprocessor(
            mcap_file=mcap_file,
            topic_mapper=self.topic_mapper,
            message_decoder=message_decoder,
            decoder_factories=decoder_factories,
            progress_reporter=preprocess_reporter,
            logger=self.logger,
            on_message_error=lambda message_index, topic, log_time_ns, exc: (
                self._handle_phase_error(
                    item=item,
                    phase="preprocess",
                    unit_label="message",
                    unit_index=message_index,
                    topic=topic,
                    name=None,
                    log_time_ns=log_time_ns,
                    exc=exc,
                )
            ),
        )
        preprocess_stats = preprocessor.preprocess_to_cache(cache_path)
        self._write_cache_metadata(
            mcap_file=mcap_file,
            cache_path=cache_path,
            cache_metadata_path=cache_metadata_path,
            message_count=preprocess_stats.message_count,
            event_count=preprocess_stats.event_count,
            cache_size_bytes=preprocess_stats.cache_size_bytes,
        )
        self.logger.info(
            "Preprocessed %s: messages=%s events=%s size=%.2f MiB duration=%.2fs",
            label,
            preprocess_stats.message_count,
            preprocess_stats.event_count,
            preprocess_stats.cache_size_bytes / (1024 * 1024),
            preprocess_stats.duration_seconds,
        )
        return preprocess_stats.event_count

    def _write_cache_metadata(
        self,
        *,
        mcap_file: Path,
        cache_path: Path,
        cache_metadata_path: Path,
        message_count: int,
        event_count: int,
        cache_size_bytes: int,
    ) -> None:
        metadata = {
            "version": self.CACHE_METADATA_VERSION,
            "fingerprint": self._build_cache_fingerprint(mcap_file),
            "message_count": int(message_count),
            "event_count": int(event_count),
            "cache_size_bytes": int(cache_size_bytes),
            "cache_path": str(cache_path),
        }
        cache_metadata_path.write_text(json.dumps(metadata, sort_keys=True))

    def _cleanup_cache_artifacts(
        self,
        cache_path: Path,
        cache_metadata_path: Path,
    ) -> None:
        if cache_path.exists():
            cache_path.unlink()
        if cache_metadata_path.exists():
            cache_metadata_path.unlink()

    def _emit_progress_for_item(
        self,
        item: ImportItem,
        completed: int,
        total: int | None,
        episode_label: str | None,
    ) -> None:
        self._emit_progress(
            item_index=item.index,
            step=completed,
            total_steps=total,
            episode_label=episode_label,
        )

    def _handle_phase_error(
        self,
        *,
        item: ImportItem,
        phase: str,
        unit_label: str,
        unit_index: int,
        topic: str,
        name: str | None,
        log_time_ns: int,
        exc: Exception,
    ) -> bool:
        details = [f"{phase} {unit_label} {unit_index}"]
        if topic:
            details.append(f"topic={topic}")
        if name:
            details.append(f"name={name}")
        if log_time_ns > 0:
            details.append(f"log_time_ns={log_time_ns}")
        details_text = ", ".join(details)
        message = f"{details_text}: {exc}"

        if self.skip_on_error != "step":
            raise ImportError(message) from exc

        worker_id = self._worker_id if self._worker_id is not None else 0
        if self._error_queue is not None:
            self._error_queue.put(
                WorkerError(
                    worker_id=worker_id,
                    item_index=item.index,
                    message=message,
                    traceback=traceback.format_exc(),
                )
            )
        self._log_worker_error(worker_id, item.index, message)
        return True

    def _create_progress_reporter(
        self,
        item: ImportItem,
        label: str,
    ) -> ProgressReporter:
        """Create a reporter appropriate for worker/main execution context."""
        if self._progress_queue is not None and self._worker_id is not None:
            return EmitProgressReporter(
                emit_progress=lambda completed, total, episode_label: (
                    self._emit_progress_for_item(item, completed, total, episode_label)
                ),
                label=label,
                report_every=self.config.progress_emit_interval,
            )
        if self.logger.level <= logging.INFO:
            return LoggingProgressReporter(
                self.logger,
                label,
                report_every=self.config.progress_emit_interval,
            )
        return NullProgressReporter()
