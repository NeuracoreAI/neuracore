"""Training CLI commands for local and cloud runs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import typer
from neuracore_types import TrainingJob, TrainingJobStatus
from omegaconf import OmegaConf
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neuracore.api.training import delete_training_job
from neuracore.core.cli.training_display import (
    STATUS_STYLES,
    RunDisplayRow,
    print_run_table,
)
from neuracore.core.const import DEFAULT_CACHE_DIR
from neuracore.core.exceptions import AuthenticationError, ConfigError, TrainingRunError
from neuracore.ml.cli import training_runs_cloud as training_runs

training_app = typer.Typer(help="Training utilities.")
console = Console()

LOCAL_RUNS_ROOT = DEFAULT_CACHE_DIR / "runs"
SUCCESS_MARKER = "Training completed successfully"


def _iter_local_runs(root: Path) -> Iterable[Path]:
    """Yield run directories under the provided root sorted by launch time desc."""
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]

    def _run_start_time(run_path: Path) -> float:
        metadata_path = run_path / "training_run.json"
        if metadata_path.exists():
            try:
                data = json.loads(metadata_path.read_text())
                launch_time = data.get("launch_time")
                if isinstance(launch_time, (int, float)):
                    return float(launch_time)
            except Exception:
                pass
        try:
            return run_path.stat().st_mtime
        except OSError:
            return 0.0

    runs.sort(key=_run_start_time, reverse=True)
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


def _load_local_hydra_metadata(run_path: Path) -> tuple[str, str]:
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


def _build_local_run_row(run: Path) -> RunDisplayRow:
    """Create a display row from a local run directory."""
    algorithm, dataset = _load_local_hydra_metadata(run)
    return RunDisplayRow(
        name=run.name,
        date=_format_mtime(run),
        success=_local_run_success(run).capitalize(),
        algorithm=algorithm,
        dataset=dataset,
    )


def _build_cloud_run_row(job: TrainingJob) -> RunDisplayRow:
    """Create a display row from a cloud training job model."""
    status_str = job.status.value
    success = "Yes" if job.status is TrainingJobStatus.COMPLETED else "No"
    return RunDisplayRow(
        name=job.name,
        date=training_runs._format_timestamp(job.launch_time),
        success=success,
        algorithm=job.algorithm,
        dataset=job.dataset_id,
        status=status_str,
    )


def _print_run_table(
    title: str, rows: list[RunDisplayRow], include_status: bool
) -> None:
    """Render tabular run data with consistent formatting."""
    print_run_table(console, title, rows, include_status)


def _resolve_cloud_training_job(name_or_id: str) -> tuple[str, TrainingJob]:
    """Resolve a training job by name or id and return (id, job_model)."""
    jobs = training_runs.list_training_runs()
    for job in jobs:
        if name_or_id == job.id or name_or_id == job.name:
            return job.id, job
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

    rows = [_build_local_run_row(run) for run in runs]
    _print_run_table("Local Training Runs", rows, include_status=False)
    return True


def _resolve_local_run_path(name_or_path: str, root: Path) -> Path:
    """Resolve a local run path from name or path input."""
    candidate = Path(name_or_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists() or not candidate.is_dir():
        raise TrainingRunError(
            f"Local training run not found at {candidate}. "
            f"Use --root to point to the correct runs directory."
        )
    return candidate


def _load_local_metadata_from_path(run_path: Path) -> dict:
    """Load locally saved training metadata."""
    metadata_path = run_path / "training_run.json"
    if not metadata_path.exists():
        raise TrainingRunError(
            f"Missing training_run.json in local run directory: {metadata_path}"
        )
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        raise TrainingRunError(f"Failed to read local metadata: {exc}") from exc


def _render_local_inspect(metadata: dict) -> None:
    """Render details for a local training run in a cloud-like layout."""
    name = metadata.get("name", "Unnamed")
    status_str = str(metadata.get("status", "UNKNOWN"))

    def _status_text(status: str) -> Text:
        style = STATUS_STYLES.get(status.upper(), "white")
        return Text(status.upper(), style=f"{style} bold")

    def _panel(title: str, rows: list[tuple[str, str | Text]]) -> None:
        grid = Table.grid(padding=(0, 1), expand=False)
        grid.add_column(style="cyan bold")
        grid.add_column()
        for label, value in rows:
            grid.add_row(label, value)
        console.print(Panel(grid, title=title, box=box.SQUARE))

    console.print(Text(f"Local Training Run: {name}", style="bold underline"))
    console.print()

    dataset_display = metadata.get("dataset_name") or metadata.get("dataset_id", "N/A")
    _panel(
        "Basic Information",
        [
            ("ID", metadata.get("id", "N/A")),
            ("Status", _status_text(status_str)),
            ("Algorithm", metadata.get("algorithm", "N/A")),
            ("Dataset", dataset_display),
        ],
    )

    _panel(
        "Timing",
        [(
            "Launched",
            training_runs._format_timestamp(metadata.get("launch_time")),
        )],
    )

    epoch = metadata.get("epoch", -1)
    step = metadata.get("step", -1)
    _panel(
        "Training Progress",
        [
            ("Epoch", str(epoch if epoch >= 0 else "N/A")),
            ("Step", str(step if step >= 0 else "N/A")),
        ],
    )

    _panel(
        "Hardware",
        [
            ("GPU Type", metadata.get("gpu_type", "N/A") or "N/A"),
            ("Num GPUs", str(metadata.get("num_gpus", 1) or "N/A")),
        ],
    )

    console.print(
        Panel(
            training_runs._format_robot_data_spec(
                metadata.get("input_robot_data_spec", {})
            ),
            title="Model Input Data Spec",
            box=box.SQUARE,
        )
    )
    console.print(
        Panel(
            training_runs._format_robot_data_spec(
                metadata.get("output_robot_data_spec", {})
            ),
            title="Model Output Data Spec",
            box=box.SQUARE,
        )
    )

    sync_details = metadata.get("synchronization_details", {})
    if sync_details or metadata.get("frequency") is not None:
        frequency = sync_details.get("frequency", metadata.get("frequency", "N/A"))
        max_delay = sync_details.get("max_delay_s")
        max_delay_display = (
            "unlimited"
            if max_delay is None or max_delay == "unlimited"
            else str(max_delay)
        )
        allow_duplicates = sync_details.get("allow_duplicates", "N/A")
        _panel(
            "Synchronization Details",
            [
                ("Frequency", f"{frequency} Hz"),
                ("Max Delay", max_delay_display),
                ("Allow Duplicates", str(allow_duplicates)),
            ],
        )

    console.print(
        Panel(
            metadata.get("local_output_dir", "N/A"),
            title="Artifact Paths",
            box=box.SQUARE,
        )
    )


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

    rows = [_build_cloud_run_row(job) for job in jobs]
    _print_run_table("Cloud Training Runs", rows, include_status=True)

    # Plain-text fallback to ensure content appears when rich output is disabled.
    for row in rows:
        typer.echo(f"{row.name} | {row.success} | {row.algorithm} | {row.dataset}")
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
    local: bool = typer.Option(
        False,
        "--local",
        help="Inspect a local training run (reads training_run.json).",
    ),
    cloud: bool = typer.Option(
        False,
        "--cloud",
        help="Inspect a cloud training run (default).",
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
) -> None:
    """Inspect a training run (cloud by default, or local with --local)."""
    # Validate mode selection
    if local and cloud:
        typer.echo("Please choose either --cloud or --local, not both.", err=True)
        raise typer.Exit(code=1)

    # Default to cloud when no flag provided
    use_cloud = cloud or not local

    try:
        if use_cloud:
            job_id, _ = _resolve_cloud_training_job(training_name)
            training_runs.run_inspect(job_id=job_id)
        else:
            run_path = _resolve_local_run_path(training_name, root)
            metadata = _load_local_metadata_from_path(run_path)
            _render_local_inspect(metadata)
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
    local: bool = typer.Option(
        False,
        "--local",
        help="Delete a local training run (directory under --root).",
    ),
    cloud: bool = typer.Option(
        False,
        "--cloud",
        help="Delete a cloud training run (default).",
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
) -> None:
    """Delete a training run (cloud by default, or local with --local)."""
    if local and cloud:
        typer.echo("Please choose either --cloud or --local, not both.", err=True)
        raise typer.Exit(code=1)

    use_cloud = cloud or not local

    try:
        if use_cloud:
            job_id, job = _resolve_cloud_training_job(training_name)
            if not yes:
                confirm = typer.confirm(
                    f"Delete cloud training run '{job.name}' ({job_id})?"
                )
                if not confirm:
                    typer.echo("Aborted.")
                    raise typer.Exit(code=0)
            delete_training_job(job_id)
            typer.echo(f"Deleted cloud training run '{job.name}' ({job_id}).")
        else:
            run_path = _resolve_local_run_path(training_name, root)
            try:
                metadata = _load_local_metadata_from_path(run_path)
                display_name = metadata.get("name") or metadata.get("id")
            except TrainingRunError:
                display_name = None
            display_name = display_name or run_path.name

            if not yes:
                confirm = typer.confirm(
                    f"Delete local training run '{display_name}' at {run_path}?"
                )
                if not confirm:
                    typer.echo("Aborted.")
                    raise typer.Exit(code=0)

            try:
                shutil.rmtree(run_path)
            except OSError as exc:
                raise TrainingRunError(f"Failed to delete local run: {exc}") from exc

            typer.echo(f"Deleted local training run '{display_name}' at {run_path}.")
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
