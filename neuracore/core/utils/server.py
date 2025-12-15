"""Lightweight FastAPI server for local model inference.

This replaces TorchServe with a more flexible, custom solution that gives us
full control over the inference pipeline while maintaining .nc.zip compatibility.
"""

import json
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neuracore_types import BatchedNCDataUnion, DataType, SynchronizedPoint
from pydantic import BaseModel

from neuracore.core.exceptions import InsufficientSynchronizedPointError

logger = logging.getLogger(__name__)

PING_ENDPOINT = "/ping"
PREDICT_ENDPOINT = "/predict"
SET_CHECKPOINT_ENDPOINT = "/set_checkpoint"


class CheckpointRequest(BaseModel):
    """Request model for setting checkpoints."""

    epoch: int


class ModelServer:
    """Lightweight model server using FastAPI."""

    def __init__(
        self,
        model_input_order: dict[DataType, list[str]],
        model_output_order: dict[DataType, list[str]],
        model_file: Path,
        org_id: str,
        job_id: str | None = None,
        device: str | None = None,
    ):
        """Initialize the model server.

        Args:
            model_input_order: Model input order
            model_output_order: Model output order
            model_file: Path to the .nc.zip model archive
            org_id: Organization ID for the model
            job_id: Job ID for the model
            device: Device the model loaded on
        """
        # Import here to avoid the need for pytorch unless the user uses this policy
        from neuracore.ml.utils.policy_inference import PolicyInference

        self.policy_inference = PolicyInference(
            model_input_order=model_input_order,
            model_output_order=model_output_order,
            org_id=org_id,
            job_id=job_id,
            model_file=model_file,
            device=device,
        )
        self.app = self._create_app()

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
                return self.policy_inference(sync_point)
            except InsufficientSynchronizedPointError:
                logger.error("Insufficient sync point data.")
                raise HTTPException(
                    status_code=422,
                    detail="Insufficient sync point data for inference.",
                )
            except Exception as e:
                logger.error("Prediction error.", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Prediction failed: {str(e)}"
                )

        @app.post(SET_CHECKPOINT_ENDPOINT)
        async def set_checkpoint(request: CheckpointRequest) -> None:
            try:
                self.policy_inference.set_checkpoint(request.epoch)
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
    model_input_order: dict[DataType, list[str]],
    model_output_order: dict[DataType, list[str]],
    model_file: Path,
    org_id: str,
    job_id: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
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
        device: Device model loaded on

    Returns:
        ModelServer instance
    """
    server = ModelServer(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_file=model_file,
        org_id=org_id,
        job_id=job_id,
        device=device,
    )
    server.run(host, port, log_level)
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Neuracore Model Server")
    parser.add_argument(
        "--model-input-order",
        required=True,
        help=(
            "Model input order consisting of json dump of "
            "dict mapping DataType to list of strings"
        ),
    )
    parser.add_argument(
        "--model-output-order",
        required=True,
        help=(
            "Model output order consisting of json dump of "
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
    parser.add_argument("--device", help="Device to load model on (cpu, cuda, etc.)")

    args = parser.parse_args()

    model_input_order = {
        DataType(k): v for k, v in json.loads(str(args.model_input_order)).items()
    }
    model_output_order = {
        DataType(k): v for k, v in json.loads(str(args.model_output_order)).items()
    }

    start_server(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_file=Path(args.model_file),
        org_id=args.org_id,
        job_id=args.job_id,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        device=args.device,
    )
