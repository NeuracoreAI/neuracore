from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from neuracore_types import (
    IMPORT_SESSION_RESERVED_FLOOR,
    ImportSession,
    ImportSessionStatus,
)

from neuracore.importer.core import base as base_module
from neuracore.importer.core.base import (
    ImportItem,
    NeuracoreDatasetImporter,
    WorkerError,
)
from neuracore.importer.core.exceptions import ImporterError, ImportError
from neuracore.importer.core.import_session import ImportSessionError

INSTANCE_START = IMPORT_SESSION_RESERVED_FLOOR + 3 * 1024


class _SessionTestImporter(NeuracoreDatasetImporter):
    def build_work_items(self):
        return [ImportItem(index=0), ImportItem(index=1)]

    def import_item(self, item):
        return None

    def _record_step(self, step, timestamp):
        return None

    def _resolve_source_path(self, source, source_name):
        return source


def _session(worker_count: int = 2) -> ImportSession:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ImportSession(
        id="session-1",
        org_id="org-1",
        robot_id="robot-id",
        dataset_id="dataset-id",
        created_by="user-1",
        range_index=3,
        instance_start=INSTANCE_START,
        worker_count=worker_count,
        status=ImportSessionStatus.ACTIVE,
        created_at=now,
        last_heartbeat_at=now,
        closed_at=None,
        heartbeat_timeout_s=300.0,
    )


@pytest.fixture
def importer(tmp_path):
    config = MagicMock()
    config.robot.name = "test_robot"
    config.frequency = 30.0
    return _SessionTestImporter(
        dataset_dir=tmp_path,
        dataset_config=config,
        output_dataset_id="dataset-id",
        robot_id="robot-id",
        max_workers=2,
    )


class _SessionApi:
    """Records the session calls import_all makes."""

    def __init__(self, monkeypatch, close_error: Exception | None = None):
        self.events: list[tuple] = []
        self.close_error = close_error
        api = self

        def create(robot_id, dataset_id, worker_count):
            api.events.append(("create", robot_id, dataset_id, worker_count))
            return _session(worker_count)

        def close(session_id, status):
            api.events.append(("close", session_id, status))
            if api.close_error is not None:
                raise api.close_error

        class FakeHeartbeat:
            def __init__(self, session):
                api.events.append(("heartbeat_created", session.id))

            def start(self):
                api.events.append(("heartbeat_start",))

            def stop(self):
                api.events.append(("heartbeat_stop",))

        monkeypatch.setattr(base_module, "create_import_session", create)
        monkeypatch.setattr(base_module, "close_import_session", close)
        monkeypatch.setattr(base_module, "ImportSessionHeartbeat", FakeHeartbeat)

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


def _run_workers_with(importer, monkeypatch, live_run):
    """Pre-check passes; the live run calls ``live_run(items, worker_count)``."""

    def run_import_workers(items, worker_count, use_precheck_result_queue=False):
        if use_precheck_result_queue:
            return [], [], None
        return live_run(items, worker_count), [], None

    monkeypatch.setattr(importer, "_run_import_workers", run_import_workers)


def test_robot_instance_is_offset_from_the_session_instance_start(importer):
    importer._instance_start = INSTANCE_START

    assert importer.robot_instance(0) == INSTANCE_START
    assert importer.robot_instance(5) == INSTANCE_START + 5
    assert importer.robot_instance(-1) == INSTANCE_START


def test_starting_a_session_requires_a_robot_id(importer):
    importer.robot_id = None

    with pytest.raises(ImporterError, match="robot_id is required"):
        importer._start_import_session(2)


def test_live_import_records_through_a_session_and_closes_it_completed(
    importer, monkeypatch
):
    api = _SessionApi(monkeypatch)
    live_instance_starts: list[int] = []

    def live_run(items, worker_count):
        live_instance_starts.append(importer._instance_start)
        return []

    _run_workers_with(importer, monkeypatch, live_run)

    importer.import_all()

    assert api.events == [
        ("create", "robot-id", "dataset-id", 2),
        ("heartbeat_created", "session-1"),
        ("heartbeat_start",),
        ("heartbeat_stop",),
        ("close", "session-1", ImportSessionStatus.COMPLETED),
    ]
    assert live_instance_starts == [INSTANCE_START]


def test_the_pre_check_runs_before_any_session_exists(importer, monkeypatch):
    api = _SessionApi(monkeypatch)
    sessions_at_precheck: list[list[str]] = []

    def run_import_workers(items, worker_count, use_precheck_result_queue=False):
        if use_precheck_result_queue:
            sessions_at_precheck.append(api.names())
        return [], [], None

    monkeypatch.setattr(importer, "_run_import_workers", run_import_workers)

    importer.import_all()

    assert sessions_at_precheck == [[]]
    assert api.names()[0] == "create"


def test_a_dry_run_creates_no_session(importer, monkeypatch):
    api = _SessionApi(monkeypatch)
    importer.dry_run = True
    live_runs: list[int] = []
    _run_workers_with(
        importer, monkeypatch, lambda items, count: live_runs.append(count) or []
    )

    importer.import_all()

    assert live_runs == [2]
    assert api.events == []


@pytest.mark.parametrize(
    "close_error",
    [None, ImportSessionError("backend down")],
    ids=["close_succeeds", "close_fails"],
)
def test_a_failed_import_cancels_the_session_and_keeps_its_error(
    importer, monkeypatch, close_error
):
    api = _SessionApi(monkeypatch, close_error=close_error)

    def live_run(items, worker_count):
        raise RuntimeError("worker crashed")

    _run_workers_with(importer, monkeypatch, live_run)

    with pytest.raises(RuntimeError, match="worker crashed"):
        importer.import_all()

    assert api.events[-2:] == [
        ("heartbeat_stop",),
        ("close", "session-1", ImportSessionStatus.CANCELLED),
    ]


def test_an_import_aborted_on_worker_errors_cancels_the_session(importer, monkeypatch):
    api = _SessionApi(monkeypatch)
    importer.skip_on_error = "all"
    _run_workers_with(
        importer,
        monkeypatch,
        lambda items, count: [WorkerError(worker_id=0, item_index=0, message="bad")],
    )

    with pytest.raises(ImporterError, match="aborted"):
        importer.import_all()

    assert api.events[-1] == ("close", "session-1", ImportSessionStatus.CANCELLED)


def test_a_close_failure_after_a_successful_import_is_only_logged(
    importer, monkeypatch
):
    _SessionApi(monkeypatch, close_error=ImportSessionError("backend down"))
    _run_workers_with(importer, monkeypatch, lambda items, count: [])
    importer.logger = MagicMock()

    importer.import_all()

    importer.logger.warning.assert_called_once()


def test_a_session_create_failure_stops_the_import_before_workers_run(
    importer, monkeypatch
):
    def create(**_kwargs):
        raise ImportSessionError("Failed to create import session: HTTP 403")

    monkeypatch.setattr(base_module, "create_import_session", create)
    live_runs: list[int] = []
    _run_workers_with(
        importer, monkeypatch, lambda items, count: live_runs.append(count) or []
    )

    with pytest.raises(ImportSessionError, match="HTTP 403"):
        importer.import_all()

    assert live_runs == []


@pytest.mark.parametrize("worker_id", [0, 1])
def test_workers_attach_to_their_own_session_source_without_registering(
    importer, monkeypatch, worker_id
):
    attached: list[tuple] = []
    monkeypatch.setattr(base_module.nc, "login", lambda: None)
    monkeypatch.setattr(base_module.nc, "get_dataset", lambda id: None)
    monkeypatch.setattr(
        base_module,
        "_attach_robot",
        lambda *args: attached.append(args),
    )

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("workers must not register a robot instance")

    monkeypatch.setattr(base_module.nc, "connect_robot", unexpected_connect)
    importer._instance_start = INSTANCE_START

    importer.prepare_worker(worker_id=worker_id)

    assert attached == [("test_robot", "robot-id", INSTANCE_START + worker_id, False)]


@pytest.fixture
def worker_nc(importer, monkeypatch):
    """Worker 0 of the session with a mocked ``nc`` module."""
    nc = MagicMock()
    monkeypatch.setattr(base_module, "nc", nc)
    importer._instance_start = INSTANCE_START
    importer._worker_id = 0
    return nc


def test_stopping_a_cloud_recording_waits_for_it(importer, worker_nc):
    worker_nc.get_cloud_recording_id.return_value = "recording-1"
    importer._worker_id = 1

    importer._stop_recording(12.5)

    worker_nc.stop_recording.assert_called_once_with(
        robot_name="test_robot",
        instance=INSTANCE_START + 1,
        wait=True,
        timestamp=12.5,
    )


def test_a_recording_that_never_reached_cloud_fails_the_episode(importer, worker_nc):
    worker_nc.get_cloud_recording_id.return_value = None

    with pytest.raises(ImportError, match="not registered with Neuracore"):
        importer._stop_recording(12.5)

    worker_nc.stop_recording.assert_not_called()
    worker_nc.get_cloud_recording_id.assert_called_once_with(
        robot_name="test_robot", instance=INSTANCE_START
    )


def test_an_episode_failing_at_stop_is_cancelled_and_not_completed(
    importer, monkeypatch, worker_nc
):
    worker_nc.get_cloud_recording_id.return_value = None
    importer.skip_on_error = "episode"
    monkeypatch.setattr(
        importer, "import_item", lambda item: importer._stop_recording(1.0)
    )
    error_queue = MagicMock()
    completed_queue = MagicMock()

    importer._step(ImportItem(index=4), 0, 0, 1, error_queue, completed_queue)

    worker_nc.cancel_recording.assert_called_once_with(
        robot_name="test_robot", instance=INSTANCE_START
    )
    completed_queue.put.assert_not_called()
    assert error_queue.put.call_args.args[0].item_index == 4
