"""Cache management CLI commands."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from neuracore.core.const import DEFAULT_CACHE_DIR, DEFAULT_RECORDING_CACHE_DIR

cache_app = typer.Typer(help="Cache management utilities.")

DEFAULT_DATASET_CACHE_DIR = DEFAULT_CACHE_DIR / "dataset_cache"


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count for CLI output."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _directory_size(path: Path) -> int:
    """Return the total size of files under a directory."""
    if not path.exists():
        return 0

    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_symlink():
                continue
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _clear_directory_contents(path: Path) -> None:
    """Remove all children from a cache directory while preserving the directory."""
    if not path.exists():
        return

    for entry in path.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)


def _selected_cache_dirs(
    recording_cache: bool,
    dataset_stats: bool,
) -> list[tuple[str, Path]]:
    """Resolve which cache directories should be cleared."""
    if not any([recording_cache, dataset_stats]):
        recording_cache = True
        dataset_stats = True

    selected = []
    if recording_cache:
        selected.append(("recording cache", DEFAULT_RECORDING_CACHE_DIR))
    if dataset_stats:
        selected.append(("dataset statistics cache", DEFAULT_DATASET_CACHE_DIR))
    return selected


@cache_app.command("clear")
def clear_cache(
    recording_cache: bool = typer.Option(
        False,
        "--recording-cache",
        help="Clear downloaded and decoded recording frames.",
    ),
    dataset_stats: bool = typer.Option(
        False,
        "--dataset-stats",
        help="Clear cached synchronized dataset statistics.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm cache deletion without prompting.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be cleared without deleting anything.",
    ),
) -> None:
    """Clear disposable Neuracore cache files."""
    selected = _selected_cache_dirs(recording_cache, dataset_stats)
    existing = [(label, path) for label, path in selected if path.exists()]

    if not existing:
        typer.echo("No cache files found.")
        return

    sizes = [(label, path, _directory_size(path)) for label, path in existing]
    total_size = sum(size for _, _, size in sizes)

    typer.echo("The following disposable cache directories will be cleared:")
    for label, path, size in sizes:
        typer.echo(f"  - {label}: {path} ({_format_bytes(size)})")
    typer.echo(f"Total: {_format_bytes(total_size)}")
    typer.echo("Training runs, auth config, and daemon state will not be deleted.")

    if dry_run:
        typer.echo("Dry run: no files were deleted.")
        return

    if not yes:
        confirmed = typer.confirm("Clear these cache files?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    for _, path, _ in sizes:
        try:
            _clear_directory_contents(path)
        except OSError as exc:
            typer.echo(f"Failed to clear {path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    typer.echo(f"Cleared {_format_bytes(total_size)} of cache files.")
