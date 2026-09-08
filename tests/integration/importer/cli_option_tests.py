"""Shared importer CLI flag integration test implementations."""

from __future__ import annotations

from pathlib import Path

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from tests.integration.importer.utils import (
    ImporterIntegrationKind,
    assert_no_high_robot_instances,
    build_importer_command,
    count_source_episodes,
    delete_importer_test_robot,
    find_org_robot_by_name,
    prepare_importer_run,
    run_checked,
    skip_without_configured_org,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

IMPORTER_DATASET_READY_TIMEOUT_S = 300.0


def _cleanup_importer_resources(
    *,
    dataset_name: str,
    robot_name: str | None = None,
    is_shared: bool = False,
) -> None:
    dataset = Dataset.get_by_name(dataset_name, non_exist_ok=True)
    if dataset is not None:
        dataset.delete()
    if robot_name is not None and not is_shared:
        delete_importer_test_robot(robot_name, is_shared=is_shared)


def test_import_dataset(
    tmp_path: Path,
    case: dict,
    kind: ImporterIntegrationKind,
    *,
    shared: bool = False,
) -> None:
    """Import a cached dataset and verify all source episodes were uploaded."""
    skip_without_configured_org()
    expected_episodes = count_source_episodes(case, kind)
    name_suffix = "shared_it" if shared else "import_it"
    context = prepare_importer_run(
        tmp_path,
        case,
        kind,
        name_suffix=name_suffix,
        unique_robot_name=not shared,
    )
    extra_args = ["--overwrite"]
    if shared:
        extra_args.insert(0, "--shared")
    try:
        with online_daemon_running():
            run_checked(build_importer_command(context, case, extra_args))
            wait_for_dataset_ready(
                context.dataset_name,
                expected_recording_count=expected_episodes,
                timeout_s=IMPORTER_DATASET_READY_TIMEOUT_S,
            )
            imported_dataset = nc.get_dataset(context.dataset_name)
            if shared:
                assert imported_dataset.is_shared is True
            assert (
                len(imported_dataset) == expected_episodes
            ), f"Expected {expected_episodes} recordings, got {len(imported_dataset)}."
            if not shared:
                archived_robot = find_org_robot_by_name(
                    context.robot_name, mode="archived"
                )
                assert (
                    archived_robot is not None
                ), f"Expected imported robot '{context.robot_name}' to be archived."
                assert archived_robot.get("archived") is True
                assert (
                    find_org_robot_by_name(context.robot_name, mode="current") is None
                )
                assert_no_high_robot_instances(context.robot_name)
    finally:
        _cleanup_importer_resources(
            dataset_name=context.dataset_name,
            robot_name=context.robot_name,
            is_shared=shared,
        )


def test_dry_run_import(
    tmp_path: Path,
    case: dict,
    kind: ImporterIntegrationKind,
) -> None:
    """Import with --dry-run and verify temporary dataset/robot are cleaned up."""
    skip_without_configured_org()
    context = prepare_importer_run(tmp_path, case, kind, name_suffix="dry_run_it")
    try:
        run_checked(build_importer_command(context, case, ["--dry-run"]))
        assert Dataset.get_by_name(context.dataset_name, non_exist_ok=True) is None
        assert (
            find_org_robot_by_name(context.robot_name, mode="mixed") is None
        ), f"Expected dry-run robot '{context.robot_name}' to be deleted."
    finally:
        _cleanup_importer_resources(
            dataset_name=context.dataset_name,
            robot_name=context.robot_name,
        )


def test_overwrite_import(
    tmp_path: Path,
    case: dict,
    kind: ImporterIntegrationKind,
    *,
    shared: bool,
) -> None:
    """Import twice with --overwrite and verify the dataset id changes."""
    skip_without_configured_org()
    expected_episodes = count_source_episodes(case, kind)
    suffix = "overwrite_shared_it" if shared else "overwrite_it"
    context = prepare_importer_run(
        tmp_path,
        case,
        kind,
        name_suffix=suffix,
        unique_robot_name=not shared,
    )
    extra_args = ["--overwrite"]
    if shared:
        extra_args.append("--shared")
    try:
        with online_daemon_running():
            run_checked(build_importer_command(context, case, extra_args))
            wait_for_dataset_ready(
                context.dataset_name,
                expected_recording_count=expected_episodes,
                timeout_s=IMPORTER_DATASET_READY_TIMEOUT_S,
            )
            first_dataset = nc.get_dataset(context.dataset_name)
            first_dataset_id = first_dataset.id
            if shared:
                assert first_dataset.is_shared is True
            assert (
                len(first_dataset) == expected_episodes
            ), f"Expected {expected_episodes} recordings, got {len(first_dataset)}."

            run_checked(build_importer_command(context, case, extra_args))
            wait_for_dataset_ready(
                context.dataset_name,
                expected_recording_count=expected_episodes,
                timeout_s=IMPORTER_DATASET_READY_TIMEOUT_S,
            )
            second_dataset = nc.get_dataset(context.dataset_name)
            assert second_dataset.id != first_dataset_id
            if shared:
                assert second_dataset.is_shared is True
            assert (
                len(second_dataset) == expected_episodes
            ), f"Expected {expected_episodes} recordings, got {len(second_dataset)}."
            if not shared:
                archived_robot = find_org_robot_by_name(
                    context.robot_name, mode="archived"
                )
                assert (
                    archived_robot is not None
                ), f"Expected imported robot '{context.robot_name}' to be archived."
                assert archived_robot.get("archived") is True
                assert_no_high_robot_instances(context.robot_name)
    finally:
        _cleanup_importer_resources(
            dataset_name=context.dataset_name,
            robot_name=context.robot_name,
            is_shared=shared,
        )


def test_max_workers_import(
    tmp_path: Path,
    case: dict,
    kind: ImporterIntegrationKind,
    *,
    max_workers: int = 2,
) -> None:
    """Import with multiple workers and verify the dataset is populated."""
    skip_without_configured_org()
    expected_episodes = count_source_episodes(case, kind)
    context = prepare_importer_run(tmp_path, case, kind, name_suffix="max_workers_it")
    try:
        with online_daemon_running():
            run_checked(
                build_importer_command(
                    context,
                    case,
                    ["--overwrite"],
                    max_workers=max_workers,
                )
            )
            wait_for_dataset_ready(
                context.dataset_name,
                expected_recording_count=expected_episodes,
                timeout_s=IMPORTER_DATASET_READY_TIMEOUT_S,
            )
            imported_dataset = nc.get_dataset(context.dataset_name)
            assert (
                len(imported_dataset) == expected_episodes
            ), f"Expected {expected_episodes} recordings, got {len(imported_dataset)}."
            archived_robot = find_org_robot_by_name(context.robot_name, mode="archived")
            assert (
                archived_robot is not None
            ), f"Expected imported robot '{context.robot_name}' to be archived."
            assert archived_robot.get("archived") is True
            assert_no_high_robot_instances(context.robot_name)
    finally:
        _cleanup_importer_resources(
            dataset_name=context.dataset_name,
            robot_name=context.robot_name,
        )


def test_random_sample_import(
    tmp_path: Path,
    case: dict,
    kind: ImporterIntegrationKind,
    *,
    random_sample: int = 2,
) -> None:
    """Import with --random-sample and verify the episode count."""
    skip_without_configured_org()
    source_episodes = count_source_episodes(case, kind)
    assert (
        random_sample <= source_episodes
    ), f"random_sample={random_sample} exceeds source episodes ({source_episodes})."
    context = prepare_importer_run(tmp_path, case, kind, name_suffix="random_sample_it")
    try:
        with online_daemon_running():
            run_checked(
                build_importer_command(
                    context,
                    case,
                    ["--overwrite", "--random-sample", str(random_sample)],
                )
            )
            wait_for_dataset_ready(
                context.dataset_name,
                expected_recording_count=random_sample,
                timeout_s=IMPORTER_DATASET_READY_TIMEOUT_S,
            )
            imported_dataset = nc.get_dataset(context.dataset_name)
            assert len(imported_dataset) == random_sample
            archived_robot = find_org_robot_by_name(context.robot_name, mode="archived")
            assert (
                archived_robot is not None
            ), f"Expected imported robot '{context.robot_name}' to be archived."
            assert archived_robot.get("archived") is True
            assert_no_high_robot_instances(context.robot_name)
    finally:
        _cleanup_importer_resources(
            dataset_name=context.dataset_name,
            robot_name=context.robot_name,
        )
