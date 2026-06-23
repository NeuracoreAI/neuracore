"""Cache manager to handle disk space usage for cache files."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages the cache directory to prevent disk space exhaustion."""

    def __init__(
        self,
        cache_dir: Path,
        max_usage_percent: float = 80.0,
        check_interval: int = 100,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: The directory where cache files are stored
            max_usage_percent: Maximum percentage of disk space to use
            check_interval: Number of operations between disk space checks
        """
        self.cache_dir = cache_dir
        self.max_usage_percent = max_usage_percent
        self.check_interval = check_interval
        self.op_counter = 0
        self.last_check_time = 0
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_disk_stats(self) -> dict[str, float]:
        """Get total and available disk space for the partition containing the cache.

        Returns:
            A dictionary with total, used, free space and percent used.
        """
        stats = shutil.disk_usage(self.cache_dir)
        return {
            "total": stats.total,
            "used": stats.used,
            "free": stats.free,
            "percent_used": (stats.used / stats.total) * 100,
        }

    def get_cache_size(self) -> int:
        """Calculate the total size of the cache directory.

        Returns:
            Total size of cache files in bytes.
        """
        total_size = 0
        for entry in self.cache_dir.glob("**/*"):
            if entry.is_file():
                total_size += entry.stat().st_size
        return total_size

    def cleanup_cache(self, percent_to_remove: float = 20.0) -> None:
        """Evict whole cached recordings to free space.

        Recordings are evicted as complete units, never partially: a reader treats
        a present recording directory as fully decoded, so deleting individual
        frame files would leave undetectable holes that surface later as
        ``FileNotFoundError``. Recordings with an active decoding lock are skipped
        so an in-progress decode is not destroyed; an evicted recording is simply
        re-decoded the next time it is needed.

        Args:
            percent_to_remove: Percentage of eligible recordings to remove.
        """
        candidates = [
            d
            for d in self.cache_dir.iterdir()
            if d.is_dir() and not self._is_recording_locked(d)
        ]
        if not candidates:
            return

        num_to_remove = max(1, int(len(candidates) * percent_to_remove / 100))
        for recording_dir in candidates[:num_to_remove]:
            shutil.rmtree(recording_dir, ignore_errors=True)

        logger.info(f"Cache cleanup: evicted {num_to_remove} recording(s)")

    @staticmethod
    def _is_recording_locked(recording_dir: Path) -> bool:
        """Return True if a decoding lock exists anywhere under the recording."""
        return any(recording_dir.rglob("*recording.lock"))

    def ensure_space_available(self, force_check: bool = False) -> bool:
        """Check if cache is using too much space and clean up if necessary.

        Only performs the check periodically based on the check_interval.

        Args:
            force_check: If True, forces a disk space check regardless of the counter

        Returns:
            True if space is available, False if cleanup failed.
        """
        self.op_counter += 1

        # Only check disk space periodically or when forced
        if not force_check and self.op_counter % self.check_interval != 0:
            return True

        disk_stats = self.get_disk_stats()

        # Check if we're using more than our allowed percentage
        if disk_stats["percent_used"] > self.max_usage_percent:
            logger.warning(
                f"Disk usage ({disk_stats['percent_used']:.1f}%) exceeds threshold "
                f"({self.max_usage_percent:.1f}%). Cleaning up cache..."
            )
            self.cleanup_cache()

            # Check if cleanup was successful
            new_stats = self.get_disk_stats()
            if new_stats["percent_used"] > self.max_usage_percent:
                # If still too high, try more aggressive cleanup
                logger.warning(
                    "First cleanup insufficient, removing more cache files..."
                )
                self.cleanup_cache(percent_to_remove=40.0)

                # Final check
                final_stats = self.get_disk_stats()
                if final_stats["percent_used"] > self.max_usage_percent:
                    logger.error(
                        f"Failed to reduce disk usage below threshold. "
                        f"Current usage: {final_stats['percent_used']:.1f}%",
                        exc_info=True,
                    )
                    return False

        return True
