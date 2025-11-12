"""Lightweight FastAPI server for local model inference.

This replaces TorchServe with a more flexible, custom solution that gives us
full control over the inference pipeline while maintaining .nc.zip compatibility.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neuracore_types import SyncPoint
from pydantic import BaseModel

from neuracore.core.exceptions import InsufficientSyncPointError
from neuracore.core.utils.image_string_encoder import ImageStringEncoder

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
        model_file: Path,
        org_id: str,
        job_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the model server.

        Args:
            model_file: Path to the .nc.zip model archive
            org_id: Organization ID for the model
            job_id: Job ID for the model
            device: Device the model loaded on
        """
        # Import here to avoid the need for pytorch unless the user uses this policy
        from neuracore.ml.utils.policy_inference import PolicyInference

        # Only pass the device argument if it's not None, for compatibility
        if device is not None:
            self.policy_inference = PolicyInference(
                org_id=org_id, job_id=job_id, model_file=model_file, device=device
            )
        else:
            self.policy_inference = PolicyInference(
                org_id=org_id, job_id=job_id, model_file=model_file
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
        @app.post(PREDICT_ENDPOINT, response_model=list[SyncPoint])
        async def predict(sync_point: SyncPoint) -> list[SyncPoint]:
            try:
                # Decode base64 images before inference
                sync_point = self._decode_images(sync_point)

                # Run inference
                try:
                    prediction = self.policy_inference(sync_point)
                except InsufficientSyncPointError:
                    logger.error("Insufficient sync point data.")
                    raise HTTPException(
                        status_code=422,
                        detail="Insufficient sync point data for inference.",
                    )

                # Encode images in response if needed
                return [
                    self._encode_outputs(pred_sync_point)
                    for pred_sync_point in prediction
                ]

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

    def _decode_images(self, sync_point: SyncPoint) -> SyncPoint:
        """Decode base64 images in sync point to numpy arrays.

        Args:
            sync_point: SyncPoint with potentially base64-encoded images

        Returns:
            SyncPoint with decoded numpy array images
        """
        # Decode RGB images
        if sync_point.rgb_images:
            for camera_name, camera_data in sync_point.rgb_images.items():
                if isinstance(camera_data.frame, str):
                    # It's a base64 string, decode it
                    camera_data.frame = ImageStringEncoder.decode_image(
                        camera_data.frame
                    )

        # Decode depth images
        if sync_point.depth_images:
            for camera_name, camera_data in sync_point.depth_images.items():
                if isinstance(camera_data.frame, str):
                    # It's a base64 string, decode it
                    camera_data.frame = ImageStringEncoder.decode_image(
                        camera_data.frame
                    )

        return sync_point

    def _encode_outputs(self, prediction: SyncPoint) -> SyncPoint:
        """Encode output images to base64 for response.

        Args:
            prediction: SyncPoint with potentially numpy array images

        Returns:
            SyncPoint with base64-encoded images
        """
        # Handle RGB image outputs
        if prediction.rgb_images:
            rgbs = prediction.rgb_images
            for camera_name, camera_data in rgbs.items():
                assert isinstance(camera_data.frame, np.ndarray)
                image = camera_data.frame
                assert len(image.shape) == 3  # [H, W, C]
                if image.shape[0] == 3:  # CHW format
                    image = np.transpose(image, (1, 2, 0))  # Convert to HWC
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                camera_data.frame = ImageStringEncoder.encode_image(image)

        if prediction.depth_images:
            depths = prediction.depth_images
            for camera_name, camera_data in depths.items():
                assert isinstance(camera_data.frame, np.ndarray)
                depth = camera_data.frame
                assert len(depth.shape) == 3  # [H, W, C]
                if depth.shape[0] == 1:  # Remove channel dimension
                    depth = depth[0]
                # Normalize depth to 0-255 range
                depth_norm = (
                    (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
                )
                depth_norm = depth_norm.astype(np.uint8)
                camera_data.frame = ImageStringEncoder.encode_image(depth_norm)

        # Convert all SyncPoint attributes to lists if they are numpy arrays
        for attr in prediction.__dict__:
            value = getattr(prediction, attr)
            if isinstance(value, np.ndarray):
                # Convert numpy array to list
                setattr(prediction, attr, value.tolist())
            elif isinstance(value, dict):
                # Convert numpy arrays in dicts to lists
                for key, item in value.items():
                    if isinstance(item, np.ndarray):
                        value[key] = item.tolist()

        return prediction

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
    model_file: Path,
    org_id: str,
    job_id: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
    device: Optional[str] = None,
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
    server = ModelServer(model_file, org_id, job_id, device)
    server.run(host, port, log_level)
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Neuracore Model Server")
    parser.add_argument(
        "--model_file", required=True, help="Path to .nc.zip model file"
    )
    parser.add_argument("--org-id", required=True, help="Organization ID")
    parser.add_argument("--job-id", required=False, help="Job ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Logging level")
    parser.add_argument("--device", help="Device to load model on (cpu, cuda, etc.)")

    args = parser.parse_args()

    start_server(
        model_file=Path(args.model_file),
        org_id=args.org_id,
        job_id=args.job_id,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        device=args.device,
    )
