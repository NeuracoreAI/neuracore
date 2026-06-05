"""Integration tests for MCAP importer datasets with daemon lifecycle."""

from __future__ import annotations

from pathlib import Path

from tests.integration.importer import cli_option_tests
from tests.integration.importer.utils import MCAP_KIND


def test_import_mcap_dataset(tmp_path: Path, importer_mcap_dataset_case: dict) -> None:
    cli_option_tests.test_import_dataset(
        tmp_path, importer_mcap_dataset_case, MCAP_KIND, shared=False
    )


def test_import_mcap_dataset_shared(
    tmp_path: Path, importer_mcap_dataset_case: dict
) -> None:
    cli_option_tests.test_import_dataset(
        tmp_path, importer_mcap_dataset_case, MCAP_KIND, shared=True
    )


def test_import_mcap_dataset_dry_run(
    tmp_path: Path, importer_mcap_dataset_case: dict
) -> None:
    cli_option_tests.test_dry_run_import(
        tmp_path, importer_mcap_dataset_case, MCAP_KIND
    )


def test_import_mcap_dataset_overwrite_private(
    tmp_path: Path, importer_mcap_dataset_case: dict
) -> None:
    cli_option_tests.test_overwrite_import(
        tmp_path, importer_mcap_dataset_case, MCAP_KIND, shared=False
    )


def test_import_mcap_dataset_overwrite_shared(
    tmp_path: Path, importer_mcap_dataset_case: dict
) -> None:
    cli_option_tests.test_overwrite_import(
        tmp_path, importer_mcap_dataset_case, MCAP_KIND, shared=True
    )
