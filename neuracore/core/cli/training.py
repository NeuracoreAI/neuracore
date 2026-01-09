"""Training CLI commands for local and cloud runs."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import typer
from omegaconf import OmegaConf

from neuracore.api.training import delete_training_job
from neuracore.core.cli import training_runs
from neuracore.core.data.dataset import DEFAULT_CACHE_DIR
from neuracore.core.exceptions import AuthenticationError, ConfigError, TrainingRunError

training_app = typer.Typer(help="Training utilities.")

LOCAL_RUNS_ROOT = DEFAULT_CACHE_DIR / "runs"
SUCCESS_MARKER = "Training completed successfully"

# Map for pretty status coloring on cloud runs
STATUS_COLORS = {
    "COMPLETED": typer.colors.GREEN,
    "RUNNING": typer.colors.BLUE,
    "PENDING": typer.colors.YELLOW,
    "PREPARING_DATA": typer.colors.YELLOW,
    "FAILED": typer.colors.RED,
    "CANCELLED": typer.colors.MAGENTA,
}


def _iter_local_runs(root: Path) -> Iterable[Path]:
    """Yield run directories under the provided root sorted by mtime desc."""
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _read_tail(path: Path, num_bytes: int = 4000) -> str:
    """Read the last num_bytes of a file to check for markers."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(size - num_bytes, 0))
            return handle.read().decode(errors="ignore")
    except OSError:
        return ""


def _local_run_success(run_path: Path) -> str:
    """Infer success from train.log contents."""
    log_path = run_path / "train.log"
    if not log_path.exists():
        return "unknown"

    tail = _read_tail(log_path)
    if SUCCESS_MARKER in tail:
        return "yes"

    # If we have a log but no success marker, assume incomplete/failed.
    return "no"


def _format_mtime(path: Path) -> str:
    """Format modification time for display."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "N/A"


def _load_local_metadata(run_path: Path) -> tuple[str, str]:
    """Extract algorithm and dataset info from the Hydra config if present."""
    config_path = run_path / ".hydra" / "config.yaml"
    if not config_path.exists():
        return ("unknown", "unknown")

    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return ("unknown", "unknown")

    algorithm = "unknown"
    if "algorithm_id" in cfg and cfg.algorithm_id:
        algorithm = str(cfg.algorithm_id)
    elif "algorithm" in cfg and hasattr(cfg.algorithm, "_target_"):
        algorithm = str(cfg.algorithm._target_)

    dataset = "unknown"
    if "dataset_name" in cfg and cfg.dataset_name:
        dataset = str(cfg.dataset_name)
    elif "dataset_id" in cfg and cfg.dataset_id:
        dataset = str(cfg.dataset_id)

    return (algorithm, dataset)


def _maybe_style(
    text: str, fg: typer.colors.Color | None = None, bold: bool = False
) -> str:
    """Apply color unless NO_COLOR is set."""
    if os.environ.get("NO_COLOR"):
        return text
    return typer.style(text, fg=fg, bold=bold)


def _style_success(success: str) -> str:
    """Colorize success text."""
    normalized = success.lower()
    if normalized == "yes":
        return _maybe_style(success, fg=typer.colors.GREEN, bold=True)
    if normalized == "no":
        return _maybe_style(success, fg=typer.colors.RED, bold=True)
    return _maybe_style(success, fg=typer.colors.YELLOW, bold=True)


def _style_status(status: str) -> str:
    """Colorize status text for cloud runs."""
    color = STATUS_COLORS.get(status.upper(), typer.colors.WHITE)
    return _maybe_style(status, fg=color, bold=True)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI color codes from a string for width calculations."""
    return ANSI_RE.sub("", text)


def _pad(text: str, width: int) -> str:
    """Pad text to a fixed width accounting for ANSI codes."""
    visible = _strip_ansi(text)
    padding = max(width - len(visible), 0)
    return f"{text}{' ' * padding}"


def _resolve_training_job(name_or_id: str) -> tuple[str, dict]:
    """Resolve a training job by name or id and return (id, job_dict)."""
    jobs = training_runs.list_training_runs()
    for job in jobs:
        job_id = job.get("id", "")
        job_name = job.get("name", "")
        if name_or_id == job_id or name_or_id == job_name:
            return job_id, job
    raise TrainingRunError(f"Training run not found: {name_or_id}")


def _print_local_runs(
    root: Path,
    limit: int | None,
    allow_empty_exit: bool = True,
) -> bool:
    """Print local runs; return True if runs were printed."""
    runs = list(_iter_local_runs(root))

    if not runs:
        typer.echo(f"No local training runs found under {root}.")
        if allow_empty_exit:
            raise typer.Exit(code=0)
        return False

    if limit is not None and limit > 0:
        runs = runs[:limit]

    typer.secho("Local Training Runs", bold=True)
    header = (
        "Name                         Date               "
        "Success  Algorithm          Dataset"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for run in runs:
        name = run.name
        date = _format_mtime(run)
        success = _local_run_success(run).capitalize()
        algorithm, dataset = _load_local_metadata(run)
        success_str = _style_success(success)
        typer.echo(
            " ".join([
                _pad(name, 28),
                _pad(date, 17),
                _pad(success_str, 8),
                _pad(algorithm, 18),
                _pad(dataset, 18),
            ])
        )
    return True


def _print_cloud_runs(
    status: str | None,
    limit: int | None,
    allow_empty_exit: bool = True,
) -> bool:
    """Print cloud runs; return True if runs were printed."""
    try:
        jobs = training_runs.list_training_runs(status_filter=status, limit=limit)
    except AuthenticationError:
        typer.echo(
            "Authentication failed. Please run 'neuracore login' first.", err=True
        )
        if allow_empty_exit:
            raise typer.Exit(code=1)
        return False
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        if allow_empty_exit:
            raise typer.Exit(code=1)
        return False
    except TrainingRunError as exc:
        typer.echo(f"Error: {exc}", err=True)
        if allow_empty_exit:
            raise typer.Exit(code=1)
        return False

    if not jobs:
        typer.echo("No cloud training runs found.")
        if allow_empty_exit:
            raise typer.Exit(code=0)
        return False

    typer.secho("Cloud Training Runs", bold=True)
    header = (
        "Name                         Date                 "
        "Success  Status      Algorithm        Dataset"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for job in jobs:
        name = job.get("name", "Unnamed")
        date = training_runs._format_timestamp(job.get("launch_time"))
        status_str = job.get("status", "UNKNOWN")
        success = "Yes" if status_str == "COMPLETED" else "No"
        algorithm = job.get("algorithm", "N/A")
        dataset = job.get("dataset_name") or job.get("dataset_id") or "N/A"
        success_str = _style_success(success)
        status_colored = _style_status(status_str)
        typer.echo(
            " ".join([
                _pad(name, 28),
                _pad(date, 19),
                _pad(success_str, 8),
                _pad(status_colored, 10),
                _pad(algorithm, 16),
                _pad(dataset, 18),
            ])
        )
    return True


@training_app.command("list")
def list_training(
    cloud: bool = typer.Option(False, "--cloud", help="List cloud runs."),
    local: bool = typer.Option(False, "--local", help="List local runs."),
    all_runs: bool = typer.Option(
        False,
        "--all",
        help="List both cloud and local runs. (default if no flags provided)",
    ),
    root: Path = typer.Option(
        LOCAL_RUNS_ROOT,
        "--root",
        "-r",
        help="Root directory containing local training runs.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Optional status filter for cloud runs (e.g., COMPLETED, RUNNING).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help=(
            "Maximum number of runs to display (applies separately to cloud and local)."
        ),
    ),
) -> None:
    """List training runs (cloud/local)."""
    if not any([cloud, local, all_runs]):
        all_runs = True

    show_cloud = cloud or all_runs
    show_local = local or all_runs

    printed_any = False
    if show_local:
        printed_any |= _print_local_runs(root, limit, allow_empty_exit=False)
        if printed_any and show_cloud:
            typer.echo("")  # spacer
    if show_cloud:
        printed_any |= _print_cloud_runs(status, limit, allow_empty_exit=False)

    if not printed_any:
        typer.echo("No training runs found.")


@training_app.command("inspect")
def inspect_training(
    training_name: str = typer.Option(
        ...,
        "--training-name",
        "-t",
        help="Name or ID of the training run to inspect.",
    ),
    show_config: bool = typer.Option(
        False,
        "--config",
        "-c",
        help="Show full algorithm configuration.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output in JSON format.",
    ),
) -> None:
    """Inspect a cloud training run by name or id."""
    try:
        job_id, _ = _resolve_training_job(training_name)
        training_runs.run_inspect(
            job_id=job_id, show_config=show_config, json_output=json_output
        )
    except (AuthenticationError, ConfigError, TrainingRunError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@training_app.command("delete")
def delete_training(
    training_name: str = typer.Option(
        ...,
        "--training-name",
        "-t",
        help="Name or ID of the training run to delete.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm deletion without prompting.",
    ),
) -> None:
    """Delete a cloud training run by name or id."""
    try:
        job_id, job = _resolve_training_job(training_name)
        if not yes:
            confirm = typer.confirm(
                f"Delete training run '{job.get('name', job_id)}' ({job_id})?"
            )
            if not confirm:
                typer.echo("Aborted.")
                raise typer.Exit(code=0)
        delete_training_job(job_id)
        typer.echo(f"Deleted training run {job_id}.")
    except (AuthenticationError, ConfigError, TrainingRunError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@training_app.command("start")
def start_training(
    cloud: bool = typer.Option(
        False, "--cloud", help="Start a cloud training run (placeholder)."
    ),
    local: bool = typer.Option(
        False, "--local", help="Start a local training run (placeholder)."
    ),
) -> None:
    """Placeholder for starting training runs with unified interface."""
    typer.echo(
        "Training start command is not yet implemented. "
        "Use your existing Hydra config for local runs "
        "or nc.start_training_run for cloud runs."
    )
