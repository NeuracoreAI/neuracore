import asyncio
import logging

import pytest

from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker


async def _failing_coroutine():
    """A simple coroutine that is designed to fail."""
    await asyncio.sleep(0.01)
    raise ValueError("This is a test exception.")


@pytest.mark.asyncio
async def test_logs_exception_from_coroutine(caplog):
    """
    Verify that exceptions raised in a tracked coroutine are logged.
    """
    caplog.set_level(logging.ERROR)

    tracker = BackgroundCoroutineTracker(loop=asyncio.get_running_loop())

    # Submit the coroutine that is expected to fail
    tracker.submit_background_coroutine(_failing_coroutine())

    # Give the coroutine time to execute and fail
    await asyncio.sleep(0.1)

    # Verify that the exception was logged
    assert "Background task raised exception" in caplog.text
    assert "This is a test exception." in caplog.text

    # Ensure the task list is empty after completion
    assert len(tracker.background_tasks) == 0

    await asyncio.sleep(0.01)


async def _dummy_coroutine():
    """A simple coroutine that does nothing."""
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_handles_closed_event_loop(caplog):
    """
    Verify that submitting a coroutine to a closed loop is handled gracefully.
    """
    caplog.set_level(logging.WARNING)

    loop = asyncio.new_event_loop()
    tracker = BackgroundCoroutineTracker(loop=loop)
    tracker.submit_background_coroutine(_dummy_coroutine())

    await asyncio.sleep(0.1)

    assert "Cannot submit coroutine; event loop is closed." not in caplog.text
    # Close the loop
    loop.stop()
    loop.close()

    # Submit a coroutine to the closed loop
    tracker.submit_background_coroutine(_dummy_coroutine())

    # Verify that a warning was logged
    assert "Cannot submit coroutine; event loop is closed." in caplog.text

    # Ensure no task was added
    assert len(tracker.background_tasks) == 0
