"""Integration tests for RLDS importer datasets with daemon lifecycle."""

from __future__ import annotations

from pathlib import Path

from tests.integration.importer import cli_option_tests
from tests.integration.importer.utils import RLDS_KIND


def test_import_rlds_dataset(tmp_path: Path, importer_rlds_dataset_case: dict) -> None:
    cli_option_tests.test_import_dataset(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND, shared=False
    )


def test_import_rlds_dataset_shared(
    tmp_path: Path, importer_rlds_dataset_case: dict
) -> None:
    cli_option_tests.test_import_dataset(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND, shared=True
    )


def test_import_rlds_dataset_dry_run(
    tmp_path: Path, importer_rlds_dataset_case: dict
) -> None:
    cli_option_tests.test_dry_run_import(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND
    )


def test_import_rlds_dataset_overwrite_private(
    tmp_path: Path, importer_rlds_dataset_case: dict
) -> None:
    cli_option_tests.test_overwrite_import(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND, shared=False
    )


def test_import_rlds_dataset_overwrite_shared(
    tmp_path: Path, importer_rlds_dataset_case: dict
) -> None:
    cli_option_tests.test_overwrite_import(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND, shared=True
    )


def test_import_rlds_dataset_max_workers(
    tmp_path: Path, importer_rlds_dataset_case: dict
) -> None:
    cli_option_tests.test_max_workers_import(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND
    )


def test_import_rlds_dataset_random_sample(
    tmp_path: Path, importer_rlds_dataset_case: dict
) -> None:
    cli_option_tests.test_random_sample_import(
        tmp_path, importer_rlds_dataset_case, RLDS_KIND
    )
