"""Tests for EventLoopManager."""

import asyncio
import threading
import time

import pytest

import neuracore.data_daemon.event_emitter as em_module
from neuracore.data_daemon.event_loop_manager import EventLoopManager


@pytest.fixture
def loop_manager() -> EventLoopManager:
    """EventLoopManager instance."""
    em_module._emitter = None

    manager = EventLoopManager()
    yield manager
    if manager.is_running():
        try:
            manager.stop(timeout=2.0)
        except RuntimeError:
            pass

    em_module._emitter = None


@pytest.fixture
def running_loop_manager(loop_manager: EventLoopManager) -> EventLoopManager:
    em_module._emitter = None

    loop_manager.start()
    yield loop_manager
    # Only stop if still running
    if loop_manager.is_running():
        loop_manager.stop(timeout=2.0)

    # Clean up emitter after test
    em_module._emitter = None


def test_initial_state(loop_manager: EventLoopManager):
    assert loop_manager.encoder_loop is None
    assert loop_manager.general_loop is None
    assert not loop_manager.is_running()
    assert not loop_manager._started


def test_start_creates_loops(loop_manager: EventLoopManager):
    loop_manager.start()

    assert loop_manager.encoder_loop is not None
    assert loop_manager.general_loop is not None
    assert loop_manager.encoder_loop.is_running()
    assert loop_manager.general_loop.is_running()
    assert loop_manager.is_running()
    assert loop_manager._started

    loop_manager.stop(timeout=2.0)


def test_start_creates_threads(loop_manager: EventLoopManager):
    loop_manager.start()

    assert loop_manager._encoder_thread is not None
    assert loop_manager._general_thread is not None
    assert loop_manager._encoder_thread.is_alive()
    assert loop_manager._general_thread.is_alive()

    loop_manager.stop(timeout=2.0)


def test_double_start_raises_error(loop_manager: EventLoopManager):
    loop_manager.start()

    with pytest.raises(RuntimeError, match="already started"):
        loop_manager.start()

    loop_manager.stop(timeout=2.0)


def test_stop_terminates_loops(running_loop_manager: EventLoopManager):
    assert running_loop_manager.is_running()

    running_loop_manager.stop(timeout=2.0)

    assert not running_loop_manager.is_running()
    assert not running_loop_manager._started

    time.sleep(0.1)
    if running_loop_manager._encoder_thread:
        assert not running_loop_manager._encoder_thread.is_alive()
    if running_loop_manager._general_thread:
        assert not running_loop_manager._general_thread.is_alive()


def test_stop_not_started_raises_error(loop_manager: EventLoopManager):
    with pytest.raises(RuntimeError, match="not started"):
        loop_manager.stop()


def test_start_stop_start_cycle(loop_manager: EventLoopManager):
    loop_manager.start()
    assert loop_manager.is_running()
    loop_manager.stop(timeout=2.0)
    assert not loop_manager.is_running()

    loop_manager._encoder_ready.clear()
    loop_manager._general_ready.clear()
    loop_manager._encoder_shutdown.clear()
    loop_manager._general_shutdown.clear()

    # Reset emitter for the second start
    em_module._emitter = None

    loop_manager.start()
    assert loop_manager.is_running()
    loop_manager.stop(timeout=2.0)
    assert not loop_manager.is_running()


def test_schedule_on_general_loop(running_loop_manager: EventLoopManager):
    result_container = []

    async def test_coroutine():
        await asyncio.sleep(0.05)
        result_container.append("executed")
        return "success"

    future = running_loop_manager.schedule_on_general_loop(test_coroutine())

    result = future.result(timeout=2.0)

    assert result == "success"
    assert "executed" in result_container


def test_schedule_multiple_coroutines(running_loop_manager: EventLoopManager):
    results = []

    async def test_coroutine(value: str):
        await asyncio.sleep(0.05)
        results.append(value)
        return value

    future1 = running_loop_manager.schedule_on_general_loop(test_coroutine("first"))
    future2 = running_loop_manager.schedule_on_general_loop(test_coroutine("second"))
    future3 = running_loop_manager.schedule_on_general_loop(test_coroutine("third"))

    r1 = future1.result(timeout=2.0)
    r2 = future2.result(timeout=2.0)
    r3 = future3.result(timeout=2.0)

    assert r1 == "first"
    assert r2 == "second"
    assert r3 == "third"
    assert set(results) == {"first", "second", "third"}


def test_schedule_with_exception(running_loop_manager: EventLoopManager):
    async def failing_coroutine():
        await asyncio.sleep(0.05)
        raise ValueError("Intentional error")

    future = running_loop_manager.schedule_on_general_loop(failing_coroutine())

    with pytest.raises(ValueError, match="Intentional error"):
        future.result(timeout=2.0)


def test_schedule_on_not_started_raises_error(loop_manager: EventLoopManager):

    async def test_coroutine():
        pass

    with pytest.raises(RuntimeError, match="not initialized|not running"):
        loop_manager.schedule_on_general_loop(test_coroutine())


def test_multiple_threads_can_schedule(running_loop_manager: EventLoopManager):
    results = []

    async def test_coroutine(thread_id: int):
        await asyncio.sleep(0.05)
        results.append(thread_id)

    def worker(thread_id: int):
        future = running_loop_manager.schedule_on_general_loop(
            test_coroutine(thread_id)
        )
        future.result(timeout=2.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    assert len(results) == 5
    assert set(results) == {0, 1, 2, 3, 4}


def test_shutdown_waits_for_tasks(running_loop_manager: EventLoopManager):
    """Test that shutdown properly handles running tasks."""
    completion_flag = []

    async def long_running_task():
        try:
            await asyncio.sleep(10.0)
            completion_flag.append("completed")
        except asyncio.CancelledError:
            completion_flag.append("cancelled")
            raise

    running_loop_manager.schedule_on_general_loop(long_running_task())

    time.sleep(0.1)

    start = time.time()
    running_loop_manager.stop(timeout=2.0)
    elapsed = time.time() - start

    assert elapsed < 2.0
    assert "cancelled" in completion_flag


def test_shutdown_cancels_tasks(running_loop_manager: EventLoopManager):
    cancellation_flag = []

    async def task_that_handles_cancellation():
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancellation_flag.append("cancelled")
            raise

    running_loop_manager.schedule_on_general_loop(task_that_handles_cancellation())

    start = time.time()
    running_loop_manager.stop(timeout=0.5)
    elapsed = time.time() - start

    assert elapsed < 2.0
    assert "cancelled" in cancellation_flag


def test_shutdown_handles_multiple_tasks(running_loop_manager: EventLoopManager):
    completed = []

    async def task(task_id: int, duration: float):
        try:
            await asyncio.sleep(duration)
            completed.append(task_id)
        except asyncio.CancelledError:
            completed.append(f"cancelled-{task_id}")
            raise

    running_loop_manager.schedule_on_general_loop(task(1, 0.1))
    running_loop_manager.schedule_on_general_loop(task(2, 0.2))
    running_loop_manager.schedule_on_general_loop(task(3, 5.0))  # Will be cancelled

    time.sleep(0.15)
    running_loop_manager.stop(timeout=0.5)

    assert 1 in completed
    assert "cancelled-3" in completed
