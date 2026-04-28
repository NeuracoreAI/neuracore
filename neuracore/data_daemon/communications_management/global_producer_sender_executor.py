"""Global executor for producer shared sender threads."""

import atexit
import logging
import os
import threading
from collections.abc import Callable
from queue import Queue
from typing import Any

logger = logging.getLogger(__name__)


class GlobalProducerSenderExecutor:
    """Global executor for producer shared sender threads.

    producers have affinity for the same workers
    """

    def __init__(self, num_workers: int):
        """Initialize the executor.

        Args:
            num_workers: The number of worker threads to use.
        """
        self._num_workers = num_workers
        self._worker_queues: list[Queue[tuple[Callable, tuple, dict] | None]] = [
            Queue(maxsize=4) for _ in range(num_workers)
        ]
        self._worker_affinity: dict[str, int] = {}
        self._next_worker = 0
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start the sender worker threads."""
        for worker_id in range(self._num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"producer-channel-sender-{worker_id}",
                daemon=True,
                args=(worker_id,),
            )
            thread.start()
            self._threads.append(thread)

    def _get_worker_affinity(self, producer_id: str) -> int:
        """Get the worker affinity for a producer ID.

        Args:
            producer_id: The producer ID.

        Returns:
            The worker affinity for the producer ID.
        """
        affinity = self._worker_affinity.get(producer_id, None)
        if affinity is None:
            affinity = self._next_worker
            self._worker_affinity[producer_id] = affinity
            logger.info("Assigned affinity %s to producer %s", affinity, producer_id)
            # round-robin
            self._next_worker = (self._next_worker + 1) % self._num_workers
        return affinity

    def submit(
        self, producer_id: str, callable: Callable, *args: Any, **kwargs: Any
    ) -> None:
        """Submit a callable to be executed on a worker thread.

        Args:
            producer_id: The producer ID.
            callable: The callable to execute.
            args: The arguments to pass to the callable.
            kwargs: The keyword arguments to pass to the callable.
        """
        worker_id = self._get_worker_affinity(producer_id)
        self._worker_queues[worker_id].put((callable, args, kwargs))

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        """Stop the sender worker threads and wait briefly for shutdown."""
        self._stop_event.set()
        for queue in self._worker_queues:
            queue.put(None)
        for thread in self._threads:
            thread.join(timeout=join_timeout_s)
        self._threads.clear()

    def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for sending messages for a producer."""
        queue = self._worker_queues[worker_id]
        while not self._stop_event.is_set():
            item = queue.get()
            if item is None:
                return

            try:
                callable, args, kwargs = item
                callable(*args, **kwargs)
            except Exception:
                logger.exception("Failed to send message")


_global_producer_sender_executor: GlobalProducerSenderExecutor | None = None


def get_global_producer_sender_executor() -> GlobalProducerSenderExecutor:
    """Return the global sender executor.

    Returns:
        The the global sender executor.
    """
    global _global_producer_sender_executor
    if _global_producer_sender_executor is None:
        # 10 seconds maximum
        _global_producer_sender_executor = GlobalProducerSenderExecutor(
            num_workers=os.cpu_count() or 1
        )
        _global_producer_sender_executor.start()
        atexit.register(_global_producer_sender_executor.stop)
    return _global_producer_sender_executor
