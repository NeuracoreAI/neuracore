"""Composite daemon lifecycle context managers for integration tests.

Sits at the top of the shared-module import graph: combines process control
(:mod:`process_control`), profile management (:mod:`profiles`), and
process/socket assertions (:mod:`assertions`) into convenience wrappers
used by every test suite.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from neuracore.data_daemon.const import DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS
from neuracore.data_daemon.daemon_control import ensure_daemon_running
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_daemon_cleanup,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    stop_daemon,
)
from tests.integration.platform.data_daemon.shared.profiles import (
    scoped_offline_profile,
    scoped_online_mode,
)
from tests.integration.platform.data_daemon.shared.reporting import report_step
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    EmittedFrame,
    _split_video_producer,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
    PACING_BURST_ALL,
)


@contextmanager
def scoped_daemon_storage_env() -> Generator[None]:
    """Point the daemon at the shared ``.data_daemon_test_state`` directory for the
    duration of the block.

    Yields:
        ``None`` — the storage env vars are configured while the body runs.
    """
    OFFLINE_RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    previous_recordings_root = os.environ.get("NEURACORE_DAEMON_RECORDINGS_ROOT")
    previous_db_path = os.environ.get("NEURACORE_DAEMON_DB_PATH")
    if previous_recordings_root is None:
        os.environ["NEURACORE_DAEMON_RECORDINGS_ROOT"] = str(OFFLINE_RECORDINGS_ROOT)
    if previous_db_path is None:
        os.environ["NEURACORE_DAEMON_DB_PATH"] = str(OFFLINE_DB_PATH)
    try:
        yield
    finally:
        if previous_recordings_root is None:
            os.environ.pop("NEURACORE_DAEMON_RECORDINGS_ROOT", None)
        if previous_db_path is None:
            os.environ.pop("NEURACORE_DAEMON_DB_PATH", None)


@contextmanager
def offline_daemon_running() -> Generator[None]:
    """Run the daemon in offline mode for the duration of the block.

    Asserts clean process/socket state before starting the daemon and again
    after it stops, so tests do not need to call :func:`assert_daemon_cleanup`
    themselves.

    Composes :func:`~profiles.scoped_offline_profile` (profile env)
    with :func:`~process_control.stop_daemon` /
    ``ensure_daemon_running`` (process lifecycle).

    Yields:
        ``None`` — the daemon is running in offline mode while the body
        executes.
    """
    with scoped_daemon_storage_env(), scoped_offline_profile():
        try:
            stop_daemon()
            assert_daemon_cleanup()
            ensure_daemon_running(timeout_s=DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS)
            yield
        finally:
            stop_daemon()
            assert_daemon_cleanup()


@contextmanager
def online_daemon_running() -> Generator[None]:
    """Run the daemon in online mode for the duration of the block.

    Stops any suite-owned leftover daemon, asserts clean process/socket state,
    then starts a fresh daemon and asserts cleanup again after it stops.

    Forces ``NCD_OFFLINE=0`` and clears ``NEURACORE_DAEMON_PROFILE`` so
    callers cannot inherit a temporary offline profile from prior tests.

    Yields:
        ``None`` — the daemon is running in online mode while the body
        executes.
    """
    with scoped_daemon_storage_env(), scoped_online_mode():
        try:
            with report_step("Start clean online daemon"):
                with Timer(
                    DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS + 15,
                    label="daemon.online_startup",
                    always_log=True,
                    assert_deadline=False,
                ):
                    stop_daemon()
                    assert_daemon_cleanup()
                    ensure_daemon_running(
                        timeout_s=DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS
                    )
            yield
        finally:
            with report_step("Gracefully stop online daemon"):
                with Timer(
                    30.0,
                    label="daemon.online_shutdown",
                    always_log=True,
                    assert_deadline=False,
                ):
                    stop_daemon()
                    assert_daemon_cleanup()


_SPLIT_VIDEO_PROCESS_TIMEOUT_S = 30.0


@contextmanager
def split_video_process_running(
    *,
    robot_name: str,
    dataset_name: str,
    camera_name: str,
    case: DataDaemonTestCase,
) -> Generator[dict[str, list[EmittedFrame]]]:
    """Run a video-only producer for *robot_name* in a separate OS process.

    Reproduces a real deployment where the recording owner and the camera are
    different processes sharing one source: starts :func:`_split_video_producer`
    in a ``"spawn"`` child and blocks until it signals ready (its own
    ``connect_robot`` plus frame-bank prewarm complete).

    Yields:
        An initially-empty mapping of trace key -> every frame the child logged,
        filled in from the child's report once it has exited. Only readable
        *after* the block: what the producer logged is not known until it stops.
    """
    spawn_ctx = multiprocessing.get_context("spawn")
    ready_event = spawn_ctx.Event()
    stop_event = spawn_ctx.Event()
    result_queue = spawn_ctx.Queue()
    process = spawn_ctx.Process(
        target=_split_video_producer,
        args=(
            robot_name,
            dataset_name,
            [camera_name],
            case.image_width,
            case.image_height,
            case.video_fps,
            case.video_detail,
            0.0,  # timestamp_start_s: capture clock, decoupled from wall clock
            False,  # random_phase
            PACING_BURST_ALL,
            0,  # context_index
            ready_event,
            stop_event,
            result_queue,
        ),
    )
    process.start()
    logged_frames: dict[str, list[EmittedFrame]] = {}
    try:
        assert ready_event.wait(timeout=MAX_TIME_TO_START_S), (
            "split-process video producer did not become ready within "
            f"{MAX_TIME_TO_START_S}s"
        )
        yield logged_frames
    finally:
        stop_event.set()
        process.join(timeout=_SPLIT_VIDEO_PROCESS_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    outcome: dict[str, Any] = result_queue.get(timeout=5.0)
    assert outcome[
        "ok"
    ], f"split-process video producer failed:\n{outcome.get('traceback')}"
    assert (
        process.exitcode == 0
    ), f"split-process video producer exited with code {process.exitcode}"
    logged_frames.update(outcome["report"])
