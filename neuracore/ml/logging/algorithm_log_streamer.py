"""Background streaming of algorithm validation logs to cloud storage."""

from pathlib import Path

from neuracore.ml.logging.base_log_streamer import BaseLogStreamer
from neuracore.ml.utils.algorithm_storage_handler import AlgorithmStorageHandler


class AlgorithmLogStreamer(BaseLogStreamer):
    """Tail validate.log and upload chunked log objects to cloud storage."""

    def __init__(
        self,
        storage_handler: AlgorithmStorageHandler,
        output_dir: Path,
        chunk_max_lines: int = 100,
        flush_interval_s: int = 2,
        max_buffer_bytes: int = 1024 * 1024,
    ) -> None:
        """Initialize algorithm validation log streamer state."""
        super().__init__(
            storage_handler=storage_handler,
            output_dir=output_dir,
            chunk_max_lines=chunk_max_lines,
            flush_interval_s=flush_interval_s,
            max_buffer_bytes=max_buffer_bytes,
        )
        self._validate_log_path = self.output_dir / "validate.log"

    @property
    def _log_path(self) -> Path:
        return self._validate_log_path

    def _chunk_remote_path(self, chunk_index: int) -> str:
        return f"logs/validate/chunk-{chunk_index:06d}.log"

    def _index_remote_path(self) -> str:
        return "logs/validate/index.json"

    def _retry_warning_message(self) -> str:
        return "Retrying algorithm log chunk upload after failure."
