"""Algorithm validation script for neuracore ML algorithms.

This module provides a command-line tool for validating ML algorithms in an
isolated virtual environment. It creates a temporary venv, installs dependencies,
and runs validation to ensure algorithms meet neuracore requirements.
"""

import json
import logging

import typer
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.endpoint import policy_local_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(
    model_input_order: str = typer.Option(
        ...,
        "--model_input_order",
        help=(
            "Model input order consisting of json dump of "
            "dict mapping DataType to list of strings"
        ),
    ),
    model_output_order: str = typer.Option(
        ...,
        "--model_output_order",
        help=(
            "Model output order consisting of json dump of "
            "dict mapping DataType to list of strings"
        ),
    ),
    job_id: str | None = typer.Option(None, "--job_id", help="Job ID to run"),
    org_id: str | None = typer.Option(None, "--org_id", help="Organization ID"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server"),
    port: int = typer.Option(8080, "--port", help="Port to bind the server"),
) -> None:
    """Launch a local policy server."""
    try:
        input_order_raw = json.loads(model_input_order)
        output_order_raw = json.loads(model_output_order)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            "Expected JSON strings for model input/output."
        ) from exc

    model_input_order_map = {DataType(k): v for k, v in input_order_raw.items()}
    model_output_order_map = {DataType(k): v for k, v in output_order_raw.items()}

    nc.login()

    if org_id is not None:
        nc.set_organization(org_id)

    policy = policy_local_server(
        model_input_order=model_input_order_map,
        model_output_order=model_output_order_map,
        train_run_name="",  # Use job id instead
        port=port,
        host=host,
        job_id=job_id,
    )
    assert policy.server_process is not None
    policy.server_process.wait()


def main() -> None:
    """CLI entrypoint for launching the local policy server."""
    typer.run(run)


if __name__ == "__main__":
    main()
