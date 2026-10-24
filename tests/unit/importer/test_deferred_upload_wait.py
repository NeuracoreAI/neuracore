from unittest.mock import MagicMock

from neuracore.importer.core import base
from neuracore.importer.core.base import NeuracoreDatasetImporter


class _Importer(NeuracoreDatasetImporter):
    def build_work_items(self):
        return []

    def import_item(self, item) -> None:
        return None

    def _record_step(self, step: dict, timestamp: float) -> None:
        return None

    def _resolve_source_path(self, source, source_name):
        return source


def _make_importer() -> _Importer:
    importer = object.__new__(_Importer)
    importer.logger = MagicMock()
    importer.robot_name = "robot"
    importer._worker_id = 0
    importer._instance_base = 100_000
    importer._recording_queue = MagicMock()
    return importer


def test_stop_recording_does_not_wait_and_queues_recording(monkeypatch):
    importer = _make_importer()
    stop = MagicMock()
    monkeypatch.setattr(base.nc, "stop_recording", stop)
    monkeypatch.setattr(
        base.nc, "get_cloud_recording_id", lambda robot_name, instance: "rec-1"
    )

    importer._stop_recording(12.5)

    assert stop.call_args.kwargs["wait"] is False
    assert stop.call_args.kwargs["timestamp"] == 12.5
    importer._recording_queue.put.assert_called_once_with("rec-1")
