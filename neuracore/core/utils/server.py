"""Lightweight FastAPI server for local model inference.

This replaces TorchServe with a more flexible, custom solution that gives us
full control over the inference pipeline while maintaining .nc.zip compatibility.
"""

import json
import logging
import time
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neuracore_types import (
    BatchedNCDataUnion,
    DataType,
    EmbodimentDescription,
    SynchronizedPoint,
)
from pydantic import BaseModel

from neuracore.core.const import (
    PING_ENDPOINT,
    PREDICT_ENDPOINT,
    SET_CHECKPOINT_ENDPOINT,
)
from neuracore.core.exceptions import InsufficientSynchronizedPointError
from neuracore.ml.logging.json_line_formatter import JsonLineLogFormatter

logger = logging.getLogger(__name__)


def _parse_embodiment_description(raw_description: str) -> EmbodimentDescription:
    """Parse a JSON CLI embodiment description, restoring typed keys."""
    return {
        DataType(data_type): {int(index): name for index, name in indexed_names.items()}
        for data_type, indexed_names in json.loads(raw_description).items()
    }


class CheckpointRequest(BaseModel):
    """Request model for setting checkpoints."""

    epoch: int


def setup_server_logging(log_level: str, log_file_path: str | None = None) -> None:
    """Configure structured logging for the server process."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file_path is not None:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        handlers.append(file_handler)

    formatter = JsonLineLogFormatter()
    for handler in handlers:
        handler.setFormatter(formatter)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def write_startup_status(
    startup_status_file_path: str | None, status: str, error: str | None = None
) -> None:
    """Persist startup status so parent process can report errors."""
    if startup_status_file_path is None:
        return
    payload: dict[str, str] = {"status": status}
    if error is not None:
        payload["error"] = error
    path = Path(startup_status_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ModelServer:
    """Lightweight model server using FastAPI."""

    def __init__(
        self,
        input_embodiment_description: EmbodimentDescription,
        output_embodiment_description: EmbodimentDescription,
        model_file: Path,
        org_id: str,
        job_id: str | None = None,
        device: str | None = None,
    ):
        """Initialize the model server.

        Args:
            input_embodiment_description: Model input data order
            output_embodiment_description: Model output data order
            model_file: Path to the .nc.zip model archive
            org_id: Organization ID for the model
            job_id: Job ID for the model
            device: Device the model loaded on
        """
        # Import here to avoid the need for pytorch unless the user uses this policy
        from neuracore.ml.utils.policy_inference import PolicyInference

        self.policy_inference = PolicyInference(
            input_embodiment_description=input_embodiment_description,
            output_embodiment_description=output_embodiment_description,
            org_id=org_id,
            job_id=job_id,
            model_file=model_file,
            device=device,
        )
        self.app = self._create_app()
        logger.info(
            "Initialized model server.",
            extra={
                "model_file": str(model_file),
                "job_id": job_id,
                "device": device,
            },
        )

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="Neuracore Model Server",
            description="Lightweight model inference server",
            version="1.0.0",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Health check endpoint
        @app.get(PING_ENDPOINT)
        async def health_check() -> dict:
            logger.debug("Health check request received.")
            return {"status": "healthy", "timestamp": time.time()}

        # Main prediction endpoint
        @app.post(
            PREDICT_ENDPOINT,
            response_model=dict[DataType, dict[str, BatchedNCDataUnion]],
        )
        async def predict(
            sync_point: SynchronizedPoint,
        ) -> dict[DataType, dict[str, BatchedNCDataUnion]]:
            try:
                logger.info("Received prediction request.")
                return self.policy_inference(sync_point)
            except InsufficientSynchronizedPointError:
                logger.error("Insufficient sync point data.")
                raise HTTPException(
                    status_code=422,
                    detail="Insufficient sync point data for inference.",
                )
            except Exception as e:
                logger.error("Prediction error: %s", str(e), exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Prediction failed: {str(e)}"
                )

        @app.post(SET_CHECKPOINT_ENDPOINT)
        async def set_checkpoint(request: CheckpointRequest) -> None:
            try:
                logger.info("Setting checkpoint to epoch=%s.", request.epoch)
                self.policy_inference.set_checkpoint(request.epoch)
                logger.info("Checkpoint set successfully.")
            except Exception as e:
                logger.error("Checkpoint loading error.", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Checkpoint loading failed: {str(e)}"
                )

        return app

    def run(
        self, host: str = "0.0.0.0", port: int = 8080, log_level: str = "info"
    ) -> None:
        """Run the server.

        Args:
            host: Host to bind to
            port: Port to bind to
            log_level: Logging level
        """
        uvicorn.run(
            self.app, host=host, port=port, log_level=log_level, access_log=True
        )


def start_server(
    input_embodiment_description: EmbodimentDescription,
    output_embodiment_description: EmbodimentDescription,
    model_file: Path,
    org_id: str,
    job_id: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
    log_file_path: str | None = None,
    startup_status_file_path: str | None = None,
    device: str | None = None,
) -> ModelServer:
    """Start a model server instance.

    Args:
        model_file: Path to the .nc.zip model archive
        org_id: Organization ID
        job_id: Job ID
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level
        log_file_path: Optional log file path for structured server logs.
        startup_status_file_path: Optional file path used to report startup status.
        device: Device model loaded on

    Returns:
        ModelServer instance
    """
    setup_server_logging(log_level=log_level, log_file_path=log_file_path)
    logger.info("Starting model server on %s:%s.", host, port)
    server = ModelServer(
        input_embodiment_description=input_embodiment_description,
        output_embodiment_description=output_embodiment_description,
        model_file=model_file,
        org_id=org_id,
        job_id=job_id,
        device=device,
    )
    write_startup_status(startup_status_file_path, status="ready")
    server.run(host, port, log_level)
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Neuracore Model Server")
    parser.add_argument(
        "--input-embodiment-description",
        required=True,
        help=(
            "Input embodiment description consisting of json dump of "
            "dict mapping DataType to list of strings"
        ),
    )
    parser.add_argument(
        "--output-embodiment-description",
        required=True,
        help=(
            "Output embodiment description consisting of json dump of "
            "dict mapping DataType to list of strings"
        ),
    )
    parser.add_argument(
        "--model-file", required=True, help="Path to .nc.zip model file"
    )
    parser.add_argument("--org-id", required=True, help="Organization ID")
    parser.add_argument("--job-id", required=False, help="Job ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Logging level")
    parser.add_argument(
        "--log-file-path",
        required=False,
        help="Optional path to write structured server logs.",
    )
    parser.add_argument(
        "--startup-status-file-path",
        required=False,
        help="Optional path to write startup status for parent process.",
    )
    parser.add_argument("--device", help="Device to load model on (cpu, cuda, etc.)")

    try:
        args = parser.parse_args()

        input_embodiment_description = _parse_embodiment_description(
            args.input_embodiment_description
        )
        output_embodiment_description = _parse_embodiment_description(
            args.output_embodiment_description
        )
        start_server(
            input_embodiment_description=input_embodiment_description,
            output_embodiment_description=output_embodiment_description,
            model_file=Path(args.model_file),
            org_id=args.org_id,
            job_id=args.job_id,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            log_file_path=args.log_file_path,
            startup_status_file_path=args.startup_status_file_path,
            device=args.device,
        )
    except Exception as exc:
        write_startup_status(
            args.startup_status_file_path,
            status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        raise
