"""Background streaming of training and Hydra logs to cloud storage."""

from pathlib import Path

from neuracore.ml.logging.base_log_streamer import BaseLogStreamer
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler


class CloudLogStreamer(BaseLogStreamer):
    """Tail train.log and upload chunked log objects to cloud storage."""

    def __init__(
        self,
        storage_handler: TrainingStorageHandler,
        output_dir: Path,
        chunk_max_lines: int = 100,
        flush_interval_s: int = 2,
        max_buffer_bytes: int = 1024 * 1024,
    ) -> None:
        """Initialize cloud log streamer settings and state."""
        super().__init__(
            storage_handler=storage_handler,
            output_dir=output_dir,
            chunk_max_lines=chunk_max_lines,
            flush_interval_s=flush_interval_s,
            max_buffer_bytes=max_buffer_bytes,
        )
        self._train_log_path = self.output_dir / "train.log"
        self._hydra_files = {
            self.output_dir / ".hydra" / "config.yaml": "logs/hydra/config.yaml",
            self.output_dir / ".hydra" / "hydra.yaml": "logs/hydra/hydra.yaml",
            self.output_dir / ".hydra" / "overrides.yaml": "logs/hydra/overrides.yaml",
        }
        self._uploaded_hydra_files: set[Path] = set()

    @property
    def _log_path(self) -> Path:
        return self._train_log_path

    def _chunk_remote_path(self, chunk_index: int) -> str:
        return f"logs/train/chunk-{chunk_index:06d}.log"

    def _index_remote_path(self) -> str:
        return "logs/train/index.json"

    def _retry_warning_message(self) -> str:
        return "Retrying cloud log chunk upload after failure."

    def _before_sync_locked(self, final_sync: bool) -> None:
        self._upload_hydra_files(final_sync=final_sync)

    def _upload_hydra_files(self, final_sync: bool) -> None:
        for local_path, remote_path in self._hydra_files.items():
            if not local_path.exists() or not local_path.is_file():
                continue
            if not final_sync and local_path in self._uploaded_hydra_files:
                continue
            uploaded = self.storage_handler.upload_file(
                local_path=local_path,
                remote_filepath=remote_path,
                content_type="text/plain",
            )
            if uploaded:
                self._uploaded_hydra_files.add(local_path)
