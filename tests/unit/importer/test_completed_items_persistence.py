"""Tests resume file path, keys, loading, and append on NeuracoreDatasetImporter."""

import logging
from pathlib import Path

import pytest

from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter


class _CompletedItemsTestImporter(NeuracoreDatasetImporter):
    """Stub importer wiring dataset_dir, output name, and completed_items_file."""

    def __init__(self, dataset_dir: Path, output_dataset_name: str):
        self.dataset_dir = dataset_dir
        self.output_dataset_name = output_dataset_name
        self.logger = logging.getLogger(__name__)
        self.dry_run = False
        self.pre_check = False
        self.completed_items_file = self._resolve_completed_items_file()

    def build_work_items(self):
        return [ImportItem(index=0)]

    def import_item(self, item: ImportItem) -> None:
        return None

    def _record_step(self, step: dict, timestamp: float) -> None:
        return None


def test_resolve_completed_items_file_sanitizes_dataset_name(tmp_path):
    """Resume path uses a safe filename suffix from the output dataset name."""
    importer = _CompletedItemsTestImporter(
        dataset_dir=tmp_path,
        output_dataset_name="my dataset/name:with?chars",
    )

    expected = tmp_path / ".neuracore_import_completed_my_dataset_name_with_chars.txt"
    assert importer._resolve_completed_items_file() == expected


def test_item_key_uses_split_and_index(tmp_path):
    """Keys use split:index; None split is an empty prefix before the colon."""
    importer = _CompletedItemsTestImporter(tmp_path, "dataset")

    assert importer._item_key(ImportItem(index=7, split="train")) == "train:7"
    assert importer._item_key(ImportItem(index=3, split=None)) == ":3"


def test_load_completed_item_keys_returns_empty_when_file_missing(tmp_path):
    """Missing resume file yields an empty completed-keys set."""
    importer = _CompletedItemsTestImporter(tmp_path, "dataset")
    assert importer._load_completed_item_keys() == set()


def test_load_completed_item_keys_reads_and_strips_lines(tmp_path):
    """Non-empty lines become set members; blank lines are ignored."""
    importer = _CompletedItemsTestImporter(tmp_path, "dataset")
    importer.completed_items_file.write_text("train:1\n\nval:2\n", encoding="utf-8")

    assert importer._load_completed_item_keys() == {"train:1", "val:2"}


def test_load_completed_item_keys_returns_empty_on_read_error(tmp_path, monkeypatch):
    """Read failures are logged best-effort and return an empty set."""
    importer = _CompletedItemsTestImporter(tmp_path, "dataset")
    importer.completed_items_file.write_text("train:1\n", encoding="utf-8")

    def raise_open(*args, **kwargs):
        raise OSError("Error")

    monkeypatch.setattr(Path, "open", raise_open)
    assert importer._load_completed_item_keys() == set()


def test_append_completed_item_keys_writes_lines_and_creates_parent(tmp_path):
    """Append creates parent dirs and writes one key per line."""
    nested_root = tmp_path / "nested" / "dataset"
    importer = _CompletedItemsTestImporter(nested_root, "dataset")

    importer._append_completed_item_keys(["train:1", "val:2"])

    assert importer.completed_items_file.exists()
    assert (
        importer.completed_items_file.read_text(encoding="utf-8") == "train:1\nval:2\n"
    )


@pytest.mark.parametrize("dry_run,pre_check", [(True, False), (False, True)])
def test_append_completed_item_keys_skips_when_disabled_modes(
    tmp_path, dry_run, pre_check
):
    """Dry run and pre-check do not write the resume file."""
    importer = _CompletedItemsTestImporter(tmp_path, "dataset")
    importer.dry_run = dry_run
    importer.pre_check = pre_check

    importer._append_completed_item_keys(["train:1"])

    assert not importer.completed_items_file.exists()


def test_append_completed_item_keys_skips_empty_input(tmp_path):
    """No-op when the key list is empty."""
    importer = _CompletedItemsTestImporter(tmp_path, "dataset")

    importer._append_completed_item_keys([])

    assert not importer.completed_items_file.exists()
