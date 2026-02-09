"""Typer CLI for the Neuracore dataset importer."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from neuracore.importer.core.exceptions import (
    CLIError,
    ConfigLoadError,
    DatasetDetectionError,
    DatasetOperationError,
    UploaderError,
)
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
            "Optional directory containing robot description files "
            "(.urdf/.xml/.mjcf)."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Delete the dataset before importing if it already exists.",
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
) -> None:
    """Import a dataset into Neuracore using the provided configuration."""
    try:
        _run_import(
            dataset_config=dataset_config,
            dataset_dir=dataset_dir,
            robot_dir=robot_dir,
            overwrite=overwrite,
            dry_run=dry_run,
            skip_on_error=skip_on_error,
            suppress_validation_warnings=no_validation_warnings,
        )
    except (
        CLIError,
        ConfigLoadError,
        DatasetOperationError,
        DatasetDetectionError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise typer.Exit(code=1) from exc
    except UploaderError as exc:
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
