"""This module provides a class for managing background coroutines."""

import asyncio
import logging
from typing import Coroutine, List, Optional

from neuracore.core.streaming.event_loop_utils import get_running_loop

logger = logging.getLogger(__name__)


class BackgroundCoroutineTracker:
    """This class schedules and keeps track of background coroutines.

    This is helpful to avoid fire and forget tasks from becoming garbage
    collected before they complete. and is more lightweight than scheduling
    with `run_coroutine_threadsafe`
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        """Initialise the background coroutine manager.

        Args:
            loop: the event loop to run on. Defaults to the running loop if not
                    provided.
        """
        self.background_tasks: List[asyncio.Task] = []
        self.loop = loop

    def _task_done(self, task: asyncio.Task) -> None:
        """Cleanup after task completion.

        Args:
            task: The task that has been completed.
        """
        if task in self.background_tasks:
            self.background_tasks.remove(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Background task cancelled")
        except Exception as e:
            logger.exception("Background task raised exception: %s", e)

    async def _submit_background_coroutine(self, coroutine: Coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self.background_tasks.append(task)
        task.add_done_callback(self._task_done)

    def submit_background_coroutine(self, coroutine: Coroutine) -> None:
        """Submit coroutine to run later.

        This method keeps tracks of the running tasks to ensure they aren't
        garbage collected until complete.

        Args:
            coroutine: the coroutine to be run at another time.
        """
        try:
            task = asyncio.create_task(coroutine)
            self.background_tasks.append(task)
            task.add_done_callback(self._task_done)
        except RuntimeError:
            if not self.loop:
                self.loop = get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self._submit_background_coroutine(coroutine), self.loop
            )

    def stop_background_coroutines(self) -> None:
        """Stop all background coroutines.

        cancels all running background coroutines
        """
        for task in self.background_tasks:
            task.cancel()
        self.background_tasks.clear()
