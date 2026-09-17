"""CLI for local dataset exports."""

from pathlib import Path

import typer

from neuracore.api.core import login
from neuracore.api.datasets import get_dataset
from neuracore.core.exceptions import AuthenticationError, ConfigError, DatasetError
from neuracore.exporter.export import export_recordings
from neuracore.exporter.mcap import McapExporter

export_app = typer.Typer(help="Export datasets to local files.")


@export_app.command("mcap")
def export_mcap(
    output: Path = typer.Option(..., "--output", "-o", help="New output directory."),
    dataset_name: str | None = typer.Option(None, "--dataset", help="Dataset name."),
) -> None:
    """Export a dataset to JSON MCAP files with original media attachments."""
    if dataset_name is None:
        raise typer.BadParameter("Provide exactly one of --dataset")
    if output.exists():
        raise typer.BadParameter(f"Output directory already exists: {output}")

    def progress(completed: int, total: int, recording_name: str) -> None:
        typer.echo(f"[{completed + 1}/{total}] Exporting recording {recording_name}")

    login()

    dataset = get_dataset(name=dataset_name)
    recordings = list(dataset)

    try:
        exporter = McapExporter()
        exporter.check_dependencies()
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
    ) as exc:
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Export complete: {manifest}")
