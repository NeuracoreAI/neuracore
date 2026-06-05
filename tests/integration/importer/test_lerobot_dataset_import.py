"""Integration tests for LeRobot importer datasets with daemon lifecycle."""

from __future__ import annotations

from pathlib import Path

from tests.integration.importer import cli_option_tests
from tests.integration.importer.utils import LEROBOT_KIND


def test_import_lerobot_dataset(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_import_dataset(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND, shared=False
    )


def test_import_lerobot_dataset_shared(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_import_dataset(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND, shared=True
    )


def test_import_lerobot_dataset_dry_run(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_dry_run_import(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND
    )


def test_import_lerobot_dataset_overwrite_private(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_overwrite_import(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND, shared=False
    )


def test_import_lerobot_dataset_overwrite_shared(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_overwrite_import(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND, shared=True
    )


def test_import_lerobot_dataset_max_workers(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_max_workers_import(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND
    )


def test_import_lerobot_dataset_random_sample(
    tmp_path: Path, importer_lerobot_dataset_case: dict
) -> None:
    cli_option_tests.test_random_sample_import(
        tmp_path, importer_lerobot_dataset_case, LEROBOT_KIND
    )
