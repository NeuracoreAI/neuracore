"""Unit tests for importer robot instance allocation and cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImporterError


class _InstanceTestImporter(NeuracoreDatasetImporter):
    """Minimal concrete importer for instance allocation tests."""

    def build_work_items(self):
        return [ImportItem(index=0)]

    def import_item(self, item):
        return None

    def _record_step(self, step, timestamp):
        return None

    def _resolve_source_path(self, source, source_name):
        return source


@pytest.fixture
def mock_dataset_config():
    config = MagicMock()
    config.robot.name = "test_robot"
    config.frequency = 30.0
    return config


@pytest.fixture
def importer(mock_dataset_config, tmp_path):
    return _InstanceTestImporter(
        dataset_dir=tmp_path,
        dataset_config=mock_dataset_config,
        output_dataset_id="dataset-id",
        robot_id="robot-id",
        max_workers=3,
    )


def test_allocate_instance_base_uses_existing_max_floor(importer, monkeypatch):
    """Existing high instance IDs raise the allocation floor."""
    monkeypatch.setattr("neuracore.importer.core.base.nc.login", lambda: None)
    monkeypatch.setattr(
        importer,
        "_api_robot",
        lambda: MagicMock(max_instance_id=lambda: 150_000),
    )
    calls: list[tuple[int, int]] = []

    def fake_randrange(start, stop):
        calls.append((start, stop))
        return start

    monkeypatch.setattr(
        "neuracore.importer.core.base.random.randrange",
        fake_randrange,
    )

    importer._allocate_instance_base()

    assert importer._instance_base == 150_001
    assert calls == [(150_001, 250_000)]


def test_allocate_instance_base_uses_minimum_floor(importer, monkeypatch):
    """Only low instance IDs still allocate above 100_000."""
    monkeypatch.setattr("neuracore.importer.core.base.nc.login", lambda: None)
    monkeypatch.setattr(
        importer,
        "_api_robot",
        lambda: MagicMock(max_instance_id=lambda: 0),
    )
    calls: list[tuple[int, int]] = []

    def fake_randrange(start, stop):
        calls.append((start, stop))
        return start

    monkeypatch.setattr(
        "neuracore.importer.core.base.random.randrange",
        fake_randrange,
    )

    importer._allocate_instance_base()

    assert importer._instance_base == 100_001
    assert calls == [(100_001, 200_000)]


def test_allocate_instance_base_requires_robot_id(
    mock_dataset_config, tmp_path, monkeypatch
):
    """Allocation fails clearly when robot_id was not provided."""
    monkeypatch.setattr("neuracore.importer.core.base.nc.login", lambda: None)
    importer = _InstanceTestImporter(
        dataset_dir=tmp_path,
        dataset_config=mock_dataset_config,
        output_dataset_id="dataset-id",
        robot_id=None,
    )
    with pytest.raises(ImporterError, match="robot_id is required"):
        importer._allocate_instance_base()


def test_cleanup_worker_instances_removes_worker_range_only(importer, monkeypatch):
    """Cleanup deletes base..base+N-1 and never touches parent instance 0."""
    removed: list[int] = []
    robot = MagicMock()
    robot.remove_instance.side_effect = lambda instance: removed.append(instance)
    monkeypatch.setattr(importer, "_api_robot", lambda: robot)

    importer._instance_base = 250_000
    importer._cleanup_worker_instances(3)

    assert removed == [250_000, 250_001, 250_002]
    assert 0 not in removed


def test_cleanup_worker_instances_skips_when_base_unresolved(importer, monkeypatch):
    """Cleanup is a no-op before allocation resolves an instance base."""
    robot = MagicMock()
    monkeypatch.setattr(importer, "_api_robot", lambda: robot)

    importer._instance_base = 0
    importer._cleanup_worker_instances(3)

    robot.remove_instance.assert_not_called()


def test_cleanup_worker_instances_swallows_errors(importer, monkeypatch):
    """Cleanup failures are logged and do not raise."""
    robot = MagicMock()
    robot.remove_instance.side_effect = RuntimeError("boom")
    monkeypatch.setattr(importer, "_api_robot", lambda: robot)

    importer._instance_base = 100_001
    importer._cleanup_worker_instances(2)
