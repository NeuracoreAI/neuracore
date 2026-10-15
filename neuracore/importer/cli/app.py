"""Typer CLI for the Neuracore dataset importer."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from neuracore_types import Codec

from neuracore.importer.core.daemon_settings import ImporterDaemonConfig
from neuracore.importer.core.exceptions import (
    CLIError,
    ConfigLoadError,
    DatasetDetectionError,
    DatasetOperationError,
    ImporterError,
)
from neuracore.importer.core.utils import parse_storage_size
from neuracore.importer.importer import _run_import

app = typer.Typer(
    add_completion=False, help="Neuracore dataset import command line interface."
)


@app.command("import")
def import_dataset(
    dataset_config: Path = typer.Option(
        ...,
        "--dataset-config",
        "-c",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Path to dataset configuration file (YAML or JSON).",
    ),
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        "-d",
        exists=True,
        file_okay=False,
        help="Path to the dataset directory.",
    ),
    robot_dir: Path | None = typer.Option(
        None,
        "--robot-dir",
        "-r",
        exists=True,
        file_okay=False,
        help=(
            "Directory containing robot description files (.urdf/.xml/.mjcf). "
            "Optional: omit it for datasets with no URDF/MJCF, and robot "
            "information is left unpopulated."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Delete the dataset before importing if it already exists.",
    ),
    shared: bool = typer.Option(
        False,
        "--shared",
        help=(
            "Create the output dataset as shared "
            "(only available for administrators)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Perform a dry run without logging data to Neuracore.",
    ),
    skip_on_error: str = typer.Option(
        "episode",
        "--skip-on-error",
        case_sensitive=False,
        help=(
            "Error handling strategy: "
            "'episode' skips the failed episode; "
            "'step' skips only the failing step; "
            "'all' aborts on first error."
        ),
    ),
    no_validation_warnings: bool = typer.Option(
        False,
        "--no-validation-warnings",
        help="Suppress warning messages from data validation.",
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        help="Maximum number of worker processes to use.",
        min=1,
    ),
    random_sample: int | None = typer.Option(
        None,
        "--random-sample",
        help="If set, import only this many episodes chosen at random for sampling.",
        min=1,
    ),
    spool_limit: str = typer.Option(
        "auto",
        "--spool-limit",
        help=(
            "Cap on the daemon video spool backlog. 'auto' uses 25% of the free "
            "disk, so imports rarely stall. Accepts size with unit: kb, mb, gb "
            "(e.g. 10gb, 500mb)."
        ),
    ),
    daemon_transcode_workers: int | None = typer.Option(
        None,
        "--daemon-transcode-workers",
        help=(
            "Concurrent daemon video transcodes. Applies only when the import "
            "starts the daemon. [default: half the CPU cores]"
        ),
        min=1,
    ),
    daemon_upload_workers: int | None = typer.Option(
        None,
        "--daemon-upload-workers",
        help=(
            "Concurrent daemon uploads. Applies only when the import starts the "
            "daemon. [default: 16]"
        ),
        min=1,
    ),
    video_codec: Codec | None = typer.Option(
        None,
        "--video-codec",
        help=(
            "Daemon RGB video codec. Applies only when the import starts the "
            "daemon. [default: the daemon profile, h264_medium]"
        ),
    ),
    debug_target_ee_frame: str | None = typer.Option(
        None,
        "--debug-target-ee-frame",
        help=(
            "If provided, log JOINT_TARGET_POSITIONS as END_EFFECTOR_POSES "
            "using this end-effector frame name."
        ),
    ),
) -> None:
    """Import a dataset into Neuracore using the provided configuration."""
    try:
        spool_limit_bytes = (
            None
            if spool_limit.strip().lower() == "auto"
            else parse_storage_size(spool_limit)
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    try:
        _run_import(
            dataset_config=dataset_config,
            dataset_dir=dataset_dir,
            robot_dir=robot_dir,
            overwrite=overwrite,
            shared=shared,
            dry_run=dry_run,
            skip_on_error=skip_on_error,
            suppress_validation_warnings=no_validation_warnings,
            max_workers=max_workers,
            random_sample=random_sample,
            daemon_config=ImporterDaemonConfig(
                spool_limit=spool_limit_bytes,
                transcode_workers=daemon_transcode_workers,
                upload_workers=daemon_upload_workers,
                video_codec=video_codec,
            ),
            debug_target_ee_frame=debug_target_ee_frame,
        )
    except (
        CLIError,
        ConfigLoadError,
        DatasetOperationError,
        DatasetDetectionError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise typer.Exit(code=1) from exc
    except ImporterError as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise typer.Exit(code=1) from exc
    except Exception:
        logging.getLogger(__name__).exception("Unexpected error during dataset import.")
        raise typer.Exit(code=1) from None


def main() -> None:
    """CLI entrypoint for the dataset importer."""
    app()


if __name__ == "__main__":
    main()
