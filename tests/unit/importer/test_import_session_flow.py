from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from neuracore_types import (
    IMPORT_SESSION_ITEM_KEY_MAX_LENGTH,
    ImportSession,
    ImportSessionItemState,
)

from neuracore.importer.core import base as base_module
from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImporterError, ImportError
from neuracore.importer.core.import_session import (
    ImportSessionError,
    ImportSessionManifest,
    load_import_session_manifest,
    save_import_session_manifest,
)


class _SessionTestImporter(NeuracoreDatasetImporter):
    item_count = 4

    def build_work_items(self):
        return [ImportItem(index=i) for i in range(self.item_count)]

    def item_key(self, item):
        return f"episode-{item.index}"

    def import_item(self, item):
        return None

    def _record_step(self, step, timestamp):
        return None

    def _resolve_source_path(self, source, source_name):
        return source


def _session(
    expected_recording_count: int = 4, recording_count: int = 0
) -> ImportSession:
    return ImportSession(
        id="session-1",
        org_id="org-1",
        robot_id="robot-id",
        dataset_id="dataset-id",
        created_by="user-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        expected_recording_count=expected_recording_count,
        recording_count=recording_count,
        is_complete=recording_count == expected_recording_count,
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
    """Plays the backend for the session calls import_all makes."""

    def __init__(self, monkeypatch):
        self.created: list[tuple] = []
        self.session = _session()
        self.item_keys = {
            ImportSessionItemState.SAVED: set(),
            ImportSessionItemState.IN_FLIGHT: set(),
        }
        self.listed: list[ImportSessionItemState] = []
        api = self

        def create(robot_id, dataset_id, expected_recording_count):
            api.created.append((robot_id, dataset_id, expected_recording_count))
            return _session(expected_recording_count)

        def get(session_id):
            assert session_id == api.session.id
            return api.session

        def list_item_keys(session_id, state):
            assert session_id == api.session.id
            api.listed.append(state)
            return set(api.item_keys[state])

        monkeypatch.setattr(base_module, "create_import_session", create)
        monkeypatch.setattr(base_module, "get_import_session", get)
        monkeypatch.setattr(
            base_module, "list_import_session_item_keys", list_item_keys
        )


@pytest.fixture
def api(monkeypatch):
    return _SessionApi(monkeypatch)


@pytest.fixture
def live_runs(importer, monkeypatch):
    """Pre-check passes; records the session and items of each live run."""
    runs: list[tuple[str | None, list[int]]] = []

    def run_import_workers(items, worker_count, use_precheck_result_queue=False):
        if not use_precheck_result_queue:
            runs.append((importer._import_session_id, [item.index for item in items]))
        return [], [], None

    monkeypatch.setattr(importer, "_run_import_workers", run_import_workers)
    return runs


def _write_manifest(importer, item_keys, robot_id="robot-id"):
    save_import_session_manifest(
        importer.manifest_file,
        ImportSessionManifest(
            import_session_id="session-1",
            robot_id=robot_id,
            dataset_id="dataset-id",
            item_keys=item_keys,
        ),
    )


def test_a_new_import_creates_a_session_for_the_selection_and_saves_a_manifest(
    importer, api, live_runs
):
    importer.import_all()

    assert api.created == [("robot-id", "dataset-id", 4)]
    assert live_runs == [("session-1", [0, 1, 2, 3])]
    assert load_import_session_manifest(importer.manifest_file) == (
        ImportSessionManifest(
            import_session_id="session-1",
            robot_id="robot-id",
            dataset_id="dataset-id",
            item_keys=["episode-0", "episode-1", "episode-2", "episode-3"],
        )
    )


def test_the_expected_count_is_the_random_sample(importer, api, live_runs):
    importer.random_sample = 2

    importer.import_all()

    assert api.created == [("robot-id", "dataset-id", 2)]
    assert len(live_runs[0][1]) == 2


def test_the_pre_check_runs_before_the_session_is_created(importer, api, monkeypatch):
    created_at_precheck: list[int] = []

    def run_import_workers(items, worker_count, use_precheck_result_queue=False):
        if use_precheck_result_queue:
            created_at_precheck.append(len(api.created))
        return [], [], None

    monkeypatch.setattr(importer, "_run_import_workers", run_import_workers)

    importer.import_all()

    assert created_at_precheck == [0]
    assert len(api.created) == 1


def test_a_dry_run_creates_no_session_and_ignores_the_manifest(
    importer, api, live_runs
):
    _write_manifest(importer, ["episode-0"], robot_id="other-robot")
    importer.dry_run = True

    importer.import_all()

    assert api.created == []
    assert live_runs == [(None, [0, 1, 2, 3])]


def test_a_session_create_failure_stops_the_import_without_a_manifest(
    importer, monkeypatch, live_runs
):
    def create(**_kwargs):
        raise ImportSessionError("Failed to create import session: HTTP 403")

    monkeypatch.setattr(base_module, "create_import_session", create)

    with pytest.raises(ImportSessionError, match="HTTP 403"):
        importer.import_all()

    assert live_runs == []
    assert not importer.manifest_file.exists()


def test_creating_a_session_requires_a_robot_id(importer, api, live_runs):
    importer.robot_id = None

    with pytest.raises(ImporterError, match="robot_id is required"):
        importer.import_all()

    assert api.created == []


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        (["a", "a"], "More than one item"),
        (["a", ""], "1 to"),
        (["a", "k" * (IMPORT_SESSION_ITEM_KEY_MAX_LENGTH + 1)], "1 to"),
    ],
    ids=["duplicate", "empty", "too_long"],
)
def test_invalid_item_keys_stop_the_import_before_a_session_exists(
    importer, api, live_runs, monkeypatch, keys, message
):
    importer.item_count = len(keys)
    monkeypatch.setattr(importer, "item_key", lambda item: keys[item.index])

    with pytest.raises(ImporterError, match=message):
        importer.import_all()

    assert api.created == []


class TestResume:
    def test_imports_only_items_neither_saved_nor_uploading(
        self, importer, api, live_runs
    ):
        _write_manifest(importer, ["episode-3", "episode-1", "episode-0"])
        api.session = _session(expected_recording_count=3, recording_count=1)
        api.item_keys[ImportSessionItemState.SAVED] = {"episode-1"}
        api.item_keys[ImportSessionItemState.IN_FLIGHT] = {"episode-0"}

        importer.import_all()

        assert api.created == []
        assert live_runs == [("session-1", [3])]
        assert api.listed == [
            ImportSessionItemState.IN_FLIGHT,
            ImportSessionItemState.SAVED,
        ]

    def test_keeps_the_original_selection_when_sampling(self, importer, api, live_runs):
        _write_manifest(importer, ["episode-2"])
        api.session = _session(expected_recording_count=1)
        importer.random_sample = 3

        importer.import_all()

        assert live_runs == [("session-1", [2])]

    def test_a_complete_session_imports_nothing(self, importer, api, live_runs):
        _write_manifest(importer, ["episode-0", "episode-1"])
        api.session = _session(expected_recording_count=2, recording_count=2)

        importer.import_all()

        assert live_runs == []
        assert api.listed == []

    def test_a_session_with_every_item_uploading_imports_nothing(
        self, importer, api, live_runs
    ):
        _write_manifest(importer, ["episode-0", "episode-1"])
        api.session = _session(expected_recording_count=2, recording_count=1)
        api.item_keys[ImportSessionItemState.SAVED] = {"episode-0"}
        api.item_keys[ImportSessionItemState.IN_FLIGHT] = {"episode-1"}

        importer.import_all()

        assert live_runs == []

    def test_a_manifest_of_another_robot_is_refused(self, importer, api, live_runs):
        _write_manifest(importer, ["episode-0"], robot_id="other-robot")

        with pytest.raises(ImporterError, match="different robot or dataset"):
            importer.import_all()

        assert live_runs == []

    def test_a_manifest_item_missing_from_the_dataset_is_refused(
        self, importer, api, live_runs
    ):
        _write_manifest(importer, ["episode-0", "episode-9"])

        with pytest.raises(ImporterError, match="no longer in the dataset"):
            importer.import_all()

        assert live_runs == []

    def test_a_corrupt_manifest_is_refused(self, importer, api, live_runs):
        importer.manifest_file.write_text("not json", encoding="utf-8")

        with pytest.raises(ImporterError, match="corrupt"):
            importer.import_all()

        assert api.created == []


@pytest.mark.parametrize("worker_id", [0, 1])
def test_workers_disable_live_data_and_attach_without_registering(
    importer, monkeypatch, worker_id
):
    calls: list[tuple] = []
    nc = MagicMock()
    nc.login.side_effect = lambda: calls.append(("login",))
    nc.stop_live_data.side_effect = lambda: calls.append(("stop_live_data",))
    nc.connect_robot.side_effect = AssertionError("workers must not register")
    monkeypatch.setattr(base_module, "nc", nc)
    monkeypatch.setattr(
        base_module, "_attach_robot", lambda *args: calls.append(("attach", *args))
    )

    importer.prepare_worker(worker_id=worker_id)

    assert calls == [
        ("login",),
        ("stop_live_data",),
        ("attach", "test_robot", "robot-id", False),
    ]


@pytest.fixture
def worker_nc(importer, monkeypatch):
    """Worker 0 of the session with a mocked ``nc`` module."""
    nc = MagicMock()
    monkeypatch.setattr(base_module, "nc", nc)
    importer._import_session_id = "session-1"
    importer._worker_id = 0
    return nc


def test_recordings_start_in_the_import_session_under_the_item_key(importer, worker_nc):
    importer._start_recording(ImportItem(index=2), 5.0)

    worker_nc.start_recording.assert_called_once_with(
        timestamp=5.0,
        import_session_id="session-1",
        item_key="episode-2",
    )


def test_stopping_a_cloud_recording_waits_for_it(importer, worker_nc):
    worker_nc.get_cloud_recording_id.return_value = "recording-1"

    importer._stop_recording(12.5)

    worker_nc.stop_recording.assert_called_once_with(wait=True, timestamp=12.5)


def test_a_recording_that_never_reached_cloud_fails_the_episode(importer, worker_nc):
    worker_nc.get_cloud_recording_id.return_value = None

    with pytest.raises(ImportError, match="may already be saved"):
        importer._stop_recording(12.5)

    worker_nc.stop_recording.assert_not_called()
    worker_nc.get_cloud_recording_id.assert_called_once_with()


def test_an_episode_failing_at_stop_is_cancelled(importer, monkeypatch, worker_nc):
    worker_nc.get_cloud_recording_id.return_value = None
    importer.skip_on_error = "episode"

    def import_item(item):
        importer._start_recording(item, 0.0)
        importer._stop_recording(1.0)

    monkeypatch.setattr(importer, "import_item", import_item)
    error_queue = MagicMock()

    importer._step(ImportItem(index=4), 0, 0, 1, error_queue)

    worker_nc.cancel_recording.assert_called_once_with()
    worker_nc.stop_recording.assert_not_called()
    assert error_queue.put.call_args.args[0].item_index == 4


def test_an_episode_failing_before_it_starts_cancels_nothing(
    importer, monkeypatch, worker_nc
):
    importer.skip_on_error = "episode"

    def import_item(item):
        raise ImportError("Episode has no frames")

    monkeypatch.setattr(importer, "import_item", import_item)
    error_queue = MagicMock()

    importer._step(ImportItem(index=4), 0, 0, 1, error_queue)

    worker_nc.cancel_recording.assert_not_called()
    assert error_queue.put.call_args.args[0].item_index == 4


def test_a_finished_episode_is_not_cancelled_by_the_next_failure(
    importer, monkeypatch, worker_nc
):
    worker_nc.get_cloud_recording_id.return_value = "recording-1"
    importer.skip_on_error = "episode"
    importer._start_recording(ImportItem(index=1), 0.0)
    importer._stop_recording(1.0)

    def import_item(item):
        raise ImportError("Episode has no frames")

    monkeypatch.setattr(importer, "import_item", import_item)

    importer._step(ImportItem(index=2), 0, 0, 1, MagicMock())

    worker_nc.cancel_recording.assert_not_called()
