"""Shared helpers for importer integration tests."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests.integration.importer.conftest import ROBOTS_REPO_COMMIT, ROBOTS_REPO_URL
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    has_configured_org,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ImporterIntegrationKind:
    """Per-importer-type settings for integration tests."""

    format_tag: str
    cache_dir_env: str
    dataset_type_label: str
    seed_workflow: str
    require_directory: bool = True
    patch_config: Callable[[dict], None] | None = None


LEROBOT_KIND = ImporterIntegrationKind(
    format_tag="lerobot",
    cache_dir_env="IMPORTER_LEROBOT_CACHE_DIR",
    dataset_type_label="LeRobot",
    seed_workflow="Seed Importer Cache",
    patch_config=lambda config: config.update({"dataset_type": "LEROBOT"}),
)
RLDS_KIND = ImporterIntegrationKind(
    format_tag="rlds",
    cache_dir_env="IMPORTER_RLDS_CACHE_DIR",
    dataset_type_label="RLDS",
    seed_workflow="Seed Importer RLDS Cache",
)
MCAP_KIND = ImporterIntegrationKind(
    format_tag="mcap",
    cache_dir_env="IMPORTER_MCAP_CACHE_DIR",
    dataset_type_label="MCAP",
    seed_workflow="Seed Importer MCAP Cache",
    require_directory=False,
)


@dataclass
class ImporterRunContext:
    dataset_path: Path
    robot_dir: Path
    patched_config_path: Path
    dataset_name: str


def skip_without_configured_org() -> None:
    if not has_configured_org():
        pytest.skip(
            "Importer integration test requires NEURACORE_ORG_ID "
            "or a saved current organization."
        )


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def find_robot_urdf(robot_repo_dir: Path, robot_keyword: str) -> Path:
    candidates = sorted(robot_repo_dir.rglob("*.urdf"))
    for candidate in candidates:
        candidate_parts_lower = [part.lower() for part in candidate.parts]
        if robot_keyword.lower() in candidate_parts_lower:
            return candidate
    raise AssertionError(
        f"Unable to find urdf for robot keyword '{robot_keyword}' in {robot_repo_dir}"
    )


def clone_robots_repo(tmp_path: Path) -> Path:
    robots_repo_dir = tmp_path / "neuracore_robots"
    run_checked(["git", "clone", ROBOTS_REPO_URL, str(robots_repo_dir)])
    run_checked(["git", "-C", str(robots_repo_dir), "checkout", ROBOTS_REPO_COMMIT])
    return robots_repo_dir


def resolve_source_config_path(dataset_config: str) -> Path:
    return REPO_ROOT / dataset_config


def load_importer_config(case: dict) -> dict:
    """Load the importer YAML config referenced by an integration test case."""
    config_path = resolve_source_config_path(case["dataset_config"])
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _resolve_tfds_builder_dir(dataset_dir: Path, dataset_name: str) -> Path:
    """Mirror TFDS/RLDS importer builder-dir resolution for episode counting."""
    if (dataset_dir / "dataset_info.json").exists():
        return dataset_dir

    version_dirs = [
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "dataset_info.json").exists()
    ]
    if version_dirs:
        return sorted(version_dirs)[-1]

    name_dir = dataset_dir / dataset_name
    if (name_dir / "dataset_info.json").exists():
        return name_dir
    nested_versions = (
        [
            path
            for path in name_dir.iterdir()
            if path.is_dir() and (path / "dataset_info.json").exists()
        ]
        if name_dir.exists()
        else []
    )
    if nested_versions:
        return sorted(nested_versions)[-1]

    raise AssertionError(
        f"Could not find dataset_info.json under {dataset_dir} for '{dataset_name}'."
    )


def _count_tfds_episodes(dataset_dir: Path, input_dataset_name: str) -> int:
    import tensorflow_datasets as tfds

    builder_dir = _resolve_tfds_builder_dir(dataset_dir, input_dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    if not builder.info.splits:
        raise AssertionError(f"No splits found in TFDS dataset at '{builder_dir}'.")
    return int(builder.info.splits.total_num_examples)


def _count_lerobot_episodes(dataset_dir: Path, input_dataset_name: str) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return int(
        LeRobotDatasetMetadata(input_dataset_name, root=dataset_dir).total_episodes
    )


def _count_mcap_episodes(dataset_dir: Path) -> int:
    if dataset_dir.is_file():
        if dataset_dir.suffix.lower() != ".mcap":
            raise AssertionError(
                f"Expected an MCAP file, got '{dataset_dir.name}' instead."
            )
        return 1

    if not dataset_dir.exists():
        raise AssertionError(f"Dataset path does not exist: {dataset_dir}")

    mcap_files = sorted(dataset_dir.rglob("*.mcap"))
    if not mcap_files:
        raise AssertionError(
            f"No MCAP files found under '{dataset_dir}'. "
            "Provide a .mcap file or a directory containing MCAP files."
        )
    return len(mcap_files)


def count_source_episodes(case: dict, kind: ImporterIntegrationKind) -> int:
    """Return the number of episodes the importer will upload for a cached case."""
    dataset_path = resolve_cached_dataset_path(
        cache_dir_env=kind.cache_dir_env,
        dataset_type=kind.dataset_type_label,
        case=case,
        seed_workflow=kind.seed_workflow,
        require_directory=kind.require_directory,
    )
    importer_config = load_importer_config(case)
    input_dataset_name = importer_config["input_dataset_name"]

    if kind == RLDS_KIND:
        return _count_tfds_episodes(dataset_path, input_dataset_name)
    if kind == LEROBOT_KIND:
        return _count_lerobot_episodes(dataset_path, input_dataset_name)
    if kind == MCAP_KIND:
        return _count_mcap_episodes(dataset_path)
    raise ValueError(f"Unsupported importer kind: {kind.format_tag}")


def resolve_cached_dataset_path(
    *,
    cache_dir_env: str,
    dataset_type: str,
    case: dict,
    seed_workflow: str,
    require_directory: bool = True,
) -> Path:
    """Resolve a cached importer dataset path from an env var and case config."""
    cache_root = os.environ.get(cache_dir_env)
    if not cache_root:
        pytest.fail(
            f"{cache_dir_env} is not set. "
            f"{dataset_type} importer integration tests require the GitHub Actions "
            "cache to be restored before running."
        )

    dataset_path = Path(cache_root) / case.get("cache_relative_path", "")
    if require_directory:
        if not dataset_path.is_dir():
            pytest.fail(
                f"{dataset_type} dataset not found in cache at {dataset_path}. "
                f"Run the '{seed_workflow}' workflow to populate the cache."
            )
    elif not dataset_path.exists():
        pytest.fail(
            f"{dataset_type} dataset not found in cache at {dataset_path}. "
            f"Run the '{seed_workflow}' workflow to populate the cache."
        )
    return dataset_path


def prepare_importer_run(
    tmp_path: Path,
    case: dict,
    kind: ImporterIntegrationKind,
    *,
    name_suffix: str,
) -> ImporterRunContext:
    dataset_case_name = case["name"]
    dataset_path = resolve_cached_dataset_path(
        cache_dir_env=kind.cache_dir_env,
        dataset_type=kind.dataset_type_label,
        case=case,
        seed_workflow=kind.seed_workflow,
        require_directory=kind.require_directory,
    )

    robots_repo_dir = clone_robots_repo(tmp_path)
    robot_urdf_path = find_robot_urdf(robots_repo_dir, case["robot_keyword"])
    robot_dir = robot_urdf_path.parent

    source_config_path = resolve_source_config_path(case["dataset_config"])
    with source_config_path.open("r", encoding="utf-8") as source_config_file:
        importer_config = yaml.safe_load(source_config_file)

    dataset_name = (
        f"{dataset_case_name}_{kind.format_tag}_{name_suffix}_{uuid.uuid4().hex[:8]}"
    )
    importer_config["output_dataset"]["name"] = dataset_name
    importer_config["robot"]["urdf_path"] = str(robot_urdf_path)
    if kind.patch_config is not None:
        kind.patch_config(importer_config)

    patched_config_path = (
        tmp_path / f"{dataset_case_name}.{kind.format_tag}.{name_suffix}.yaml"
    )
    with patched_config_path.open("w", encoding="utf-8") as patched_config_file:
        yaml.safe_dump(importer_config, patched_config_file, sort_keys=False)

    return ImporterRunContext(
        dataset_path=dataset_path,
        robot_dir=robot_dir,
        patched_config_path=patched_config_path,
        dataset_name=dataset_name,
    )


def build_importer_command(
    context: ImporterRunContext,
    case: dict,
    extra_args: list[str],
    *,
    max_workers: int | None = None,
) -> list[str]:
    worker_count = (
        max_workers if max_workers is not None else case.get("max_workers", 1)
    )
    return [
        sys.executable,
        "-m",
        "neuracore.core.cli.app",
        "importer",
        "import",
        "--dataset-config",
        str(context.patched_config_path),
        "--dataset-dir",
        str(context.dataset_path),
        "--robot-dir",
        str(context.robot_dir),
        "--max-workers",
        str(worker_count),
        "--no-validation-warnings",
        *extra_args,
    ]
