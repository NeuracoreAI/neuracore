"""CLI entrypoint for launching a local policy server.

This module parses embodiment descriptions from JSON, authenticates with
Neuracore, starts the local endpoint server process, and forwards startup
failures to cloud endpoint error reporting when an endpoint ID is provided.
"""

import json
import logging
import traceback

import typer
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.endpoint import policy_local_server
from neuracore.ml.utils.endpoint_storage_handler import EndpointStorageHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _try_report_error_to_cloud(
    endpoint_id: str, org_id: str | None, error_msg: str
) -> None:
    """Report endpoint startup failures to cloud storage without blocking shutdown.

    This helper is called from the local launch error path after the stack trace
    has already been logged. Any failure in this reporting flow is logged and
    suppressed so the original launch exception still propagates.
    """
    try:
        nc.login()
        if org_id is not None:
            nc.set_organization(org_id)
        storage_handler = EndpointStorageHandler(endpoint_id=endpoint_id)
        storage_handler.report_endpoint_error(error_msg)
        logger.info("Successfully reported endpoint error to cloud.")
    except Exception:
        logger.error("Failed to report endpoint error to cloud.", exc_info=True)


def run(
    input_embodiment_description: str = typer.Option(
        ...,
        "--input_embodiment_description",
        help=(
            "Input embodiment description consisting of json dump of "
            "dict mapping DataType to indexed name mapping"
        ),
    ),
    output_embodiment_description: str = typer.Option(
        ...,
        "--output_embodiment_description",
        help=(
            "Output embodiment description consisting of json dump of "
            "dict mapping DataType to indexed name mapping"
        ),
    ),
    job_id: str | None = typer.Option(None, "--job_id", help="Job ID to run"),
    endpoint_id: str | None = typer.Option(
        None, "--endpoint_id", help="Endpoint ID for log streaming"
    ),
    org_id: str | None = typer.Option(None, "--org_id", help="Organization ID"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server"),
    port: int = typer.Option(8080, "--port", help="Port to bind the server"),
) -> None:
    """Launch a local policy server."""
    try:
        input_order_raw = json.loads(input_embodiment_description)
        output_order_raw = json.loads(output_embodiment_description)

        input_embodiment_description_map = {
            DataType(k): v for k, v in input_order_raw.items()
        }
        output_embodiment_description_map = {
            DataType(k): v for k, v in output_order_raw.items()
        }

        nc.login()

        if org_id is not None:
            nc.set_organization(org_id)

        policy = policy_local_server(
            input_embodiment_description=input_embodiment_description_map,
            output_embodiment_description=output_embodiment_description_map,
            train_run_name="",  # Use job id instead
            port=port,
            host=host,
            job_id=job_id,
            endpoint_id=endpoint_id,
        )
        assert policy.server_process is not None
        policy.server_process.wait()
    except Exception as exc:
        error_msg = traceback.format_exc()
        bad_json_message = "Expected JSON strings for model input/output."
        report_error_msg = (
            f"{bad_json_message}\n{error_msg}"
            if isinstance(exc, json.JSONDecodeError)
            else error_msg
        )
        logger.error("Endpoint launch failed:\n%s", error_msg)
        if endpoint_id is not None:
            _try_report_error_to_cloud(
                endpoint_id=endpoint_id, org_id=org_id, error_msg=report_error_msg
            )
        if isinstance(exc, json.JSONDecodeError):
            raise typer.BadParameter(bad_json_message) from exc
        raise


def main() -> None:
    """CLI entrypoint for launching the local policy server."""
    typer.run(run)


if __name__ == "__main__":
    main()
