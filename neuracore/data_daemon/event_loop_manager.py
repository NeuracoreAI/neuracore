"""Event loop manager for dual-loop async architecture.

This module provides an EventLoopManager that manages two separate asyncio event loops:
1. Encoder Loop: For CPU/codec-bound work
2. General Loop: For I/O-bound work

"""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any

from neuracore.data_daemon.event_emitter import init_emitter

logger = logging.getLogger(__name__)


class EventLoopManager:
    """Manages two separate asyncio event loops in dedicated threads.

    This class provides lifecycle management for the dual-loop architecture,
    ensuring proper startup, cross-loop communication, and graceful shutdown.
    """

    def __init__(self) -> None:
        """Initialize the event loop manager."""
        # Event loops
        self.encoder_loop: asyncio.AbstractEventLoop | None = None
        self.general_loop: asyncio.AbstractEventLoop | None = None

        # Threads
        self._encoder_thread: threading.Thread | None = None
        self._general_thread: threading.Thread | None = None

        # Shutdown events
        self._encoder_shutdown: threading.Event = threading.Event()
        self._general_shutdown: threading.Event = threading.Event()

        # Ready events
        self._encoder_ready: threading.Event = threading.Event()
        self._general_ready: threading.Event = threading.Event()

        self._started = False

    def start(self) -> None:
        """Start both event loops in separate threads.

        Raises:
            RuntimeError: If already started or if loops fail to start.
        """
        if self._started:
            raise RuntimeError("EventLoopManager already started")

        logger.info("Starting EventLoopManager...")

        self._general_thread = threading.Thread(
            target=self._run_general_loop,
            name="general-event-loop",
            daemon=False,
        )
        self._general_thread.start()

        if not self._general_ready.wait(timeout=5.0):
            raise RuntimeError("General event loop failed to start within timeout")

        # Initialize the global emitter with the general loop
        if self.general_loop:
            init_emitter(loop=self.general_loop)

        self._encoder_thread = threading.Thread(
            target=self._run_encoder_loop,
            name="encoder-event-loop",
            daemon=False,
        )
        self._encoder_thread.start()

        if not self._encoder_ready.wait(timeout=5.0):
            raise RuntimeError("Encoder event loop failed to start within timeout")

        self._started = True
        logger.info("EventLoopManager started successfully")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop both event loops gracefully.

        Signals shutdown and waits for loop threads to finish.

        Args:
            timeout: Maximum time to wait for each loop to stop.

        Raises:
            RuntimeError: If not started.
        """
        if not self._started:
            raise RuntimeError("EventLoopManager not started")

        logger.info("Stopping EventLoopManager...")

        self._encoder_shutdown.set()
        self._general_shutdown.set()

        # Wait for threads to finish
        if self._general_thread and self._general_thread.is_alive():
            self._general_thread.join(timeout=timeout)
            if self._general_thread.is_alive():
                logger.warning("General loop thread did not stop within timeout")

        if self._encoder_thread and self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=timeout)
            if self._encoder_thread.is_alive():
                logger.warning("Encoder loop thread did not stop within timeout")

        self._started = False
        logger.info("EventLoopManager stopped")

    def schedule_on_general_loop(
        self, coroutine: Coroutine[Any, Any, Any]
    ) -> Future[Any]:
        """Schedule a coroutine to run on the general loop from any thread.

        Args:
            coroutine: Coroutine to execute on the general loop.

        Returns:
            Future that will be completed when the coroutine finishes.

        Raises:
            RuntimeError: If the general loop is not running.
        """
        if not self.general_loop:
            raise RuntimeError("General loop not running")

        return asyncio.run_coroutine_threadsafe(coroutine, self.general_loop)

    def schedule_on_encoder_loop(
        self, coroutine: Coroutine[Any, Any, Any]
    ) -> Future[Any]:
        """Schedule a coroutine to run on the encoder loop from any thread.

        Args:
            coroutine: Coroutine to execute on the encoder loop.

        Returns:
            Future that will be completed when the coroutine finishes.

        Raises:
            RuntimeError: If the encoder loop is not running.
        """
        if not self.encoder_loop:
            raise RuntimeError("Encoder loop not running")

        return asyncio.run_coroutine_threadsafe(coroutine, self.encoder_loop)

    def _run_general_loop(self) -> None:
        """Run the general event loop in its dedicated thread.

        This is the main loop for I/O-bound operations
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.general_loop = loop

            logger.info("General event loop started")
            self._general_ready.set()

            async def monitor_shutdown() -> None:
                while not self._general_shutdown.is_set():
                    await asyncio.sleep(0.1)
                loop.stop()

            loop.create_task(monitor_shutdown())

            loop.run_forever()

            logger.info("General event loop shutting down")

        except Exception as e:
            logger.error(f"Error in general event loop: {e}", exc_info=True)
            raise

        finally:
            # Cancel remaining tasks
            try:
                tasks = [
                    task
                    for task in asyncio.all_tasks(self.general_loop)
                    if not task.done()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    loop.run_until_complete(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )
            except Exception as e:
                logger.warning(f"Error cancelling tasks: {e}")

            if not loop.is_closed():
                loop.close()
            logger.info("General event loop stopped")

    def _run_encoder_loop(self) -> None:
        """Run the encoder event loop in its dedicated thread.

        This loop is dedicated to CPU/codec-bound work
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.encoder_loop = loop

            logger.info("Encoder event loop started")
            self._encoder_ready.set()

            async def monitor_shutdown() -> None:
                while not self._encoder_shutdown.is_set():
                    await asyncio.sleep(0.1)
                loop.stop()

            loop.create_task(monitor_shutdown())

            loop.run_forever()

            logger.info("Encoder event loop shutting down")

        except Exception as e:
            logger.error(f"Error in encoder event loop: {e}", exc_info=True)
            raise

        finally:
            # Cancel remaining tasks
            try:
                tasks = [
                    task
                    for task in asyncio.all_tasks(self.encoder_loop)
                    if not task.done()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    loop.run_until_complete(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )
            except Exception as e:
                logger.warning(f"Error cancelling tasks: {e}")

            if not loop.is_closed():
                loop.close()
            logger.info("Encoder event loop stopped")

    def is_running(self) -> bool:
        """Check if both event loops are running.

        Returns:
            True if both loops are running, False otherwise.
        """
        return (
            self._started
            and self.general_loop is not None
            and self.general_loop.is_running()
            and self.encoder_loop is not None
            and self.encoder_loop.is_running()
        )
