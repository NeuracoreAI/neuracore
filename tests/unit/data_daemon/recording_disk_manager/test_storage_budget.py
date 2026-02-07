from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

try:
    from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (  # noqa: E501
        StorageBudget,
        StoragePolicy,
        get_free_bytes,
        scan_dir_bytes,
        scan_used_bytes,
    )
except ImportError:  # pragma: no cover
    from neuracore.data_daemon.recording_disk_manager.storage_budget import (  # type: ignore[assignment]  # noqa: E501
        StorageBudget,
        StoragePolicy,
        get_free_bytes,
        scan_dir_bytes,
        scan_used_bytes,
    )


def test_scan_used_bytes_returns_zero_for_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert scan_used_bytes(missing) == 0


def test_scan_used_bytes_counts_nested_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "b" / "c").mkdir(parents=True)

    (root / "a" / "f1.bin").write_bytes(b"x" * 10)
    (root / "b" / "f2.bin").write_bytes(b"y" * 25)
    (root / "b" / "c" / "f3.bin").write_bytes(b"z" * 7)

    assert scan_used_bytes(root) == 42
    assert scan_dir_bytes(root / "b") == 32


def test_get_free_bytes_returns_positive_for_existing_path(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    free = get_free_bytes(tmp_path)
    assert isinstance(free, int)
    assert free > 0


def test_storage_budget_unlimited_always_reserves_and_never_over_limit(
    tmp_path: Path,
) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=None,
        min_free_disk_bytes=0,
        refresh_seconds=0.0,
    )
    budget = StorageBudget(recordings_root, policy)

    assert budget.reserve(10**9) is True
    budget.release(10**9)
    assert budget.is_over_limit() is False


def test_storage_budget_reserve_release_tracks_used_bytes(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=1000,
        min_free_disk_bytes=0,
        refresh_seconds=0.0,
    )
    budget = StorageBudget(recordings_root, policy)

    assert budget.reserve(250) is True
    assert budget.reserve(750) is True
    assert budget.reserve(1) is False

    budget.release(500)
    assert budget.reserve(1) is True


def test_storage_budget_release_clamps_to_zero(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=1000,
        min_free_disk_bytes=0,
        refresh_seconds=0.0,
    )
    budget = StorageBudget(recordings_root, policy)

    assert budget.reserve(10) is True
    budget.release(10_000)
    assert budget.reserve(1000) is True


def test_storage_budget_is_over_limit_detects_overage(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=10,
        min_free_disk_bytes=0,
        refresh_seconds=0.0,
    )
    budget = StorageBudget(recordings_root, policy)

    assert budget.is_over_limit() is False
    assert budget.reserve(11) is False


def test_storage_budget_refresh_if_stale_reconciles_from_disk(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=10_000,
        min_free_disk_bytes=0,
        refresh_seconds=0.01,
    )
    budget = StorageBudget(recordings_root, policy)

    (recordings_root / "a.bin").write_bytes(b"a" * 123)
    time.sleep(0.02)
    budget.refresh_if_stale()

    assert budget.reserve(1) is True


@pytest.mark.skipif(
    os.name == "nt", reason="Monotonic timing differences can be flaky on CI"
)
def test_storage_budget_refresh_if_stale_noop_when_recent(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=10_000,
        min_free_disk_bytes=0,
        refresh_seconds=60.0,
    )
    budget = StorageBudget(recordings_root, policy)

    (recordings_root / "a.bin").write_bytes(b"a" * 123)
    budget.refresh_if_stale()

    assert budget.reserve(1) is True


def test_storage_budget_has_free_disk_for_write_respects_margin(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir(parents=True, exist_ok=True)

    policy = StoragePolicy(
        storage_limit_bytes=10_000,
        min_free_disk_bytes=10**15,
        refresh_seconds=0.0,
    )
    budget = StorageBudget(recordings_root, policy)

    assert budget.has_free_disk_for_write(1) is False
