"""CLI commands for inspecting training runs.

This module provides CLI commands to list and inspect training runs,
displaying training parameters, model input/output ordering, and artifact paths.
"""

from datetime import datetime
from typing import Any

import requests
import typer
from neuracore_types import CrossEmbodimentDescription, TrainingJob, TrainingJobStatus
from pydantic import TypeAdapter, ValidationError
from rich.console import Console

from neuracore.core.auth import get_auth
from neuracore.core.cli.training_display import RunDisplayRow, print_run_table
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.exceptions import AuthenticationError, ConfigError, TrainingRunError

TRAINING_JOB_LIST_ADAPTER = TypeAdapter(list[TrainingJob])
console = Console()


def _format_timestamp(timestamp: float | None) -> str:
    """Format a Unix timestamp to human-readable string.

    Args:
        timestamp: Unix timestamp or None.

    Returns:
        Formatted datetime string or "N/A" if timestamp is None.
    """
    if timestamp is None:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(start_time: float | None, end_time: float | None) -> str:
    """Calculate and format duration between two timestamps.

    Args:
        start_time: Start Unix timestamp or None.
        end_time: End Unix timestamp or None.

    Returns:
        Formatted duration string or "N/A" if either timestamp is None.
    """
    if start_time is None or end_time is None:
        return "N/A"
    duration_seconds = int(end_time - start_time)
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_robot_data_spec(robot_data_spec: CrossEmbodimentDescription | None) -> str:
    """Format robot data spec for display.

    Args:
        robot_data_spec: Dictionary mapping robot IDs to data types and names.

    Returns:
        Formatted string representation of the robot data spec.
    """
    if not robot_data_spec:
        return "  (none)"

    lines: list[str] = []
    for robot_id, data_types in robot_data_spec.items():
        lines.append(f"  Robot: {robot_id}")
        for data_type, names in data_types.items():
            data_type_name = getattr(data_type, "value", str(data_type))
            names_list = names or []
            names_str = ", ".join(names_list) if names_list else "(none)"
            lines.append(f"    {data_type_name}: [{names_str}]")
    return "\n".join(lines)


def _get_model_artifact_path(org_id: str, job_id: str) -> str:
    """Get the GCS path for the model artifact.

    Args:
        org_id: Organization ID.
        job_id: Training job ID.

    Returns:
        GCS path string for the model artifact.
    """
    return f"organizations/{org_id}/training/{job_id}/model.nc.zip"


def _build_run_row(job: TrainingJob) -> RunDisplayRow:
    """Create a display row for a cloud training job."""
    status_str = job.status.value
    success = "Yes" if job.status is TrainingJobStatus.COMPLETED else "No"
    return RunDisplayRow(
        name=job.name,
        date=_format_timestamp(job.start_time or job.launch_time),
        success=success,
        algorithm=job.algorithm,
        dataset=job.dataset_id,
        status=status_str,
    )


def _fetch_training_jobs(auth: Any, org_id: str) -> list[TrainingJob]:
    """Fetch all training jobs for an organization.

    Args:
        auth: Authentication object with get_headers method.
        org_id: Organization ID.

    Returns:
        List of training job models.

    Raises:
        TrainingRunError: If the API request fails.
    """
    try:
        response = requests.get(
            f"{API_URL}/org/{org_id}/training/jobs",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        jobs = response.json()
        return TRAINING_JOB_LIST_ADAPTER.validate_python(jobs)
    except requests.exceptions.ConnectionError:
        raise TrainingRunError(
            "Failed to connect to neuracore server. "
            "Please check your internet connection and try again."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise AuthenticationError("Authentication failed. Please login again.")
        raise TrainingRunError(f"Failed to fetch training jobs: {e}")
    except requests.exceptions.RequestException as e:
        raise TrainingRunError(f"Failed to fetch training jobs: {e}")
    except ValidationError as e:
        raise TrainingRunError(f"Failed to parse training jobs: {e}") from e


def _fetch_training_job(auth: Any, org_id: str, job_id: str) -> TrainingJob:
    """Fetch a specific training job by ID.

    Args:
        auth: Authentication object with get_headers method.
        org_id: Organization ID.
        job_id: Training job ID.

    Returns:
        Training job model.

    Raises:
        TrainingRunError: If the job is not found or API request fails.
    """
    try:
        response = requests.get(
            f"{API_URL}/org/{org_id}/training/jobs/{job_id}",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        job = response.json()
        return TrainingJob.model_validate(job)
    except requests.exceptions.ConnectionError:
        raise TrainingRunError(
            "Failed to connect to neuracore server. "
            "Please check your internet connection and try again."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise TrainingRunError(f"Training job not found: {job_id}")
        if e.response.status_code == 401:
            raise AuthenticationError("Authentication failed. Please login again.")
        raise TrainingRunError(f"Failed to fetch training job: {e}")
    except requests.exceptions.RequestException as e:
        raise TrainingRunError(f"Failed to fetch training job: {e}")
    except ValidationError as e:
        raise TrainingRunError(f"Failed to parse training job: {e}") from e


def list_training_runs(
    status_filter: str | None = None,
    limit: int | None = None,
) -> list[TrainingJob]:
    """List training runs for the current organization.

    Args:
        status_filter: Optional filter by status (e.g., "COMPLETED", "RUNNING").
        limit: Optional maximum number of results to return.

    Returns:
        List of training job models.

    Raises:
        AuthenticationError: If not authenticated.
        ConfigError: If organization is not configured.
        TrainingRunError: If the API request fails.
    """
    status_value: TrainingJobStatus | None = None
    if status_filter:
        try:
            status_value = TrainingJobStatus(status_filter.upper())
        except ValueError as exc:
            raise TrainingRunError(f"Invalid status filter: {status_filter}") from exc

    auth = get_auth()
    if not auth.is_authenticated:
        auth.login()

    org_id = get_current_org()
    jobs = _fetch_training_jobs(auth, org_id)

    # Apply status filter if provided
    if status_value:
        jobs = [job for job in jobs if job.status == status_value]

    # Sort by start_time (fallback launch_time) descending (most recent first)
    jobs.sort(
        key=lambda x: (x.start_time or x.launch_time or 0),
        reverse=True,
    )

    # Apply limit if provided
    if limit is not None and limit > 0:
        jobs = jobs[:limit]

    return jobs


def get_training_run(job_id: str) -> TrainingJob:
    """Get detailed information about a specific training run.

    Args:
        job_id: The ID of the training job to inspect.

    Returns:
        Training job model with full details.

    Raises:
        AuthenticationError: If not authenticated.
        ConfigError: If organization is not configured.
        TrainingRunError: If the job is not found or API request fails.
    """
    auth = get_auth()
    if not auth.is_authenticated:
        auth.login()

    org_id = get_current_org()
    return _fetch_training_job(auth, org_id, job_id)


def run_list(
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help="Maximum number of results to display.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional details for each training run.",
    ),
) -> None:
    """List training runs for the current organization."""
    try:
        jobs = list_training_runs(status_filter=status, limit=limit)

        if not jobs:
            typer.echo("No training runs found.")
            return

        typer.echo(f"\nFound {len(jobs)} training run(s):\n")
        rows = [_build_run_row(job) for job in jobs]
        print_run_table(console, "Cloud Training Runs", rows, include_status=True)

        # Emit a plain-text list to ensure names are present in environments
        # where rich formatting is disabled (used by tests).
        for row in rows:
            typer.echo(f"{row.name} | {row.success} | {row.algorithm} | {row.dataset}")

        if verbose:
            typer.echo("")
            for job in jobs:
                epoch = job.epoch
                step = job.step
                gpu_type = getattr(job.gpu_type, "value", str(job.gpu_type))
                num_gpus = job.num_gpus
                typer.echo(f"{job.id[:8]}...  {job.name}")
                typer.echo(f"           Epoch: {epoch}, Step: {step}")
                typer.echo(f"           GPU: {gpu_type} x{num_gpus}")
                typer.echo("")

    except AuthenticationError:
        typer.echo(
            "Authentication failed. Please run 'neuracore login' first.", err=True
        )
        raise typer.Exit(code=1)
    except ConfigError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except TrainingRunError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def run_inspect(
    job_id: str = typer.Argument(
        ...,
        help="The ID of the training run to inspect.",
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
    """Inspect a specific training run with detailed information.

    Displays training parameters, model input/output ordering, and artifact paths.
    """
    try:
        auth = get_auth()
        if not auth.is_authenticated:
            auth.login()

        org_id = get_current_org()
        job = _fetch_training_job(auth, org_id, job_id)

        if json_output:
            import json

            typer.echo(json.dumps(job.model_dump(mode="json"), indent=2, default=str))
            return

        # Header
        typer.echo("\n" + "=" * 60)
        typer.secho(f"Training Run: {job.name}", bold=True)
        typer.echo("=" * 60)

        # Basic Info
        typer.echo("\n--- Basic Information ---")
        typer.echo(f"  ID:         {job.id}")
        typer.echo(f"  Status:     {job.status.value}")
        typer.echo(f"  Algorithm:  {job.algorithm}")
        if job.algorithm_id:
            typer.echo(f"  Algo ID:    {job.algorithm_id}")

        # Timing
        typer.echo("\n--- Timing ---")
        typer.echo(f"  Launched:   {_format_timestamp(job.launch_time)}")
        typer.echo(f"  Started:    {_format_timestamp(job.start_time)}")
        typer.echo(f"  Ended:      {_format_timestamp(job.end_time)}")
        typer.echo(f"  Duration:   {_format_duration(job.start_time, job.end_time)}")

        # Progress
        typer.echo("\n--- Training Progress ---")
        epoch = job.epoch
        step = job.step
        typer.echo(f"  Epoch:      {epoch if epoch >= 0 else 'N/A'}")
        typer.echo(f"  Step:       {step if step >= 0 else 'N/A'}")

        # Hardware
        typer.echo("\n--- Hardware ---")
        typer.echo(f"  GPU Type:   {getattr(job.gpu_type, 'value', str(job.gpu_type))}")
        typer.echo(f"  Num GPUs:   {job.num_gpus}")
        if job.zone:
            typer.echo(f"  Zone:       {job.zone}")

        # Data References
        typer.echo("\n--- Data References ---")
        typer.echo(f"  Dataset ID:        {job.dataset_id}")
        if job.synced_dataset_id:
            typer.echo(f"  Synced Dataset ID: {job.synced_dataset_id}")

        # Model Input/Output Ordering
        typer.echo("\n--- Model Input Data Spec ---")
        input_spec = job.input_robot_data_spec
        typer.echo(_format_robot_data_spec(input_spec))

        typer.echo("\n--- Model Output Data Spec ---")
        output_spec = job.output_robot_data_spec
        typer.echo(_format_robot_data_spec(output_spec))

        # Synchronization Details
        sync_details = job.synchronization_details
        if sync_details is not None:
            typer.echo("\n--- Synchronization Details ---")
            typer.echo(f"  Frequency:        {sync_details.frequency} Hz")
            max_delay = sync_details.max_delay_s
            if max_delay is not None and max_delay < 1e10:
                typer.echo(f"  Max Delay:        {max_delay}s")
            else:
                typer.echo("  Max Delay:        unlimited")
            typer.echo(f"  Allow Duplicates: {sync_details.allow_duplicates}")

        # Artifact Paths
        typer.echo("\n--- Artifact Paths ---")
        artifact_path = _get_model_artifact_path(org_id, job.id)
        typer.echo(f"  Model Path: gs://<bucket>/{artifact_path}")

        # Resume Points (checkpoints)
        resume_points = job.resume_points
        if resume_points:
            typer.echo("\n--- Checkpoints (Resume Points) ---")
            for i, timestamp in enumerate(resume_points):
                checkpoint_time = _format_timestamp(timestamp)
                typer.echo(f"  [{i + 1}] {checkpoint_time}")

        # Algorithm Config
        if show_config:
            typer.echo("\n--- Algorithm Configuration ---")
            config = job.algorithm_config
            if config:
                for key, value in config.items():
                    typer.echo(f"  {key}: {value}")
            else:
                typer.echo("  (no configuration)")

        # Error (if any)
        error = job.error
        if error:
            typer.echo("\n--- Error ---")
            typer.secho(f"  {error}", fg=typer.colors.RED)

        typer.echo("\n")

    except AuthenticationError:
        typer.echo(
            "Authentication failed. Please run 'neuracore login' first.", err=True
        )
        raise typer.Exit(code=1)
    except ConfigError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except TrainingRunError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def main_list() -> None:
    """CLI entrypoint for listing training runs."""
    typer.run(run_list)


def main_inspect() -> None:
    """CLI entrypoint for inspecting a training run."""
    typer.run(run_inspect)


if __name__ == "__main__":
    main_list()
