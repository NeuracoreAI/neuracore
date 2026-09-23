"""CLI for local dataset exports."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import typer

from neuracore.api.core import login
from neuracore.api.datasets import get_dataset
from neuracore.core.exceptions import (
    AuthenticationError,
    ConfigError,
    DatasetError,
    SynchronizationError,
)
from neuracore.exporter.export import DatasetExporter, export_recordings
from neuracore.exporter.lerobot import LeRobotExporter
from neuracore.exporter.mcap import McapExporter

export_app = typer.Typer(help="Export datasets to local files.")


def _export_dataset(
    output: Path,
    dataset_name: str | None,
    exporter_factory: Callable[[], DatasetExporter],
) -> None:
    """Run the shared validation, authentication and export CLI workflow."""
    if dataset_name is None:
        raise typer.BadParameter("Provide exactly one of --dataset")
    if output.exists():
        raise typer.BadParameter(f"Output directory already exists: {output}")

    def progress(completed: int, total: int, recording_name: str) -> None:
        typer.echo(f"[{completed + 1}/{total}] Exporting recording {recording_name}")

    try:
        exporter = exporter_factory()
        exporter.check_dependencies()
        login()
        dataset = get_dataset(name=dataset_name)
        recordings = list(dataset)
        manifest = export_recordings(
            dataset, recordings, output, exporter, progress=progress
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        AuthenticationError,
        ConfigError,
        DatasetError,
        SynchronizationError,
    ) as exc:
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Export complete: {manifest}")


@export_app.command("mcap")
def export_mcap(
    output: Path = typer.Option(..., "--output", "-o", help="New output directory."),
    dataset_name: str | None = typer.Option(None, "--dataset", help="Dataset name."),
) -> None:
    """Export a dataset to JSON MCAP files with original media attachments."""
    _export_dataset(output, dataset_name, McapExporter)


@export_app.command("lerobot")
def export_lerobot(
    output: Path = typer.Option(..., "--output", "-o", help="New output directory."),
    dataset_name: str | None = typer.Option(None, "--dataset", help="Dataset name."),
    fps: int = typer.Option(
        ..., "--fps", help="Frequency (Hz) to synchronize each recording at."
    ),
) -> None:
    """Export a dataset to a LeRobot v3.0 dataset directory."""
    _export_dataset(output, dataset_name, partial(LeRobotExporter, fps))
