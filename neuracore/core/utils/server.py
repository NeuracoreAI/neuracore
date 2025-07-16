"""Lightweight FastAPI server for local model inference.

This replaces TorchServe with a more flexible, custom solution that gives us
full control over the inference pipeline while maintaining .nc.zip compatibility.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from neuracore.core.nc_types import DataType, ModelPrediction, SyncPoint

logger = logging.getLogger(__name__)

PING_ENDPOINT = "/ping"
PREDICT_ENDPOINT = "/predict"
SET_CHECKPOINT_ENDPOINT = "/set_checkpoint"


class CheckpointRequest(BaseModel):
    """Request model for setting checkpoints."""

    epoch: int


class ModelServer:
    """Lightweight model server using FastAPI."""

    def __init__(self, model_file: Path, org_id: str, job_id: Optional[str] = None):
        """Initialize the model server.

        Args:
            model_file: Path to the .nc.zip model archive
            org_id: Organization ID for the model
            job_id: Job ID for the model
        """
        # Import here to avoid the need for pytorch unless the user uses this policy
        from neuracore.ml.utils.policy_inference import PolicyInference

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
        @app.post(PREDICT_ENDPOINT, response_model=ModelPrediction)
        async def predict(sync_point: SyncPoint) -> ModelPrediction:
            try:
                # Decode base64 images before inference
                sync_point = self._decode_images(sync_point)

                # Run inference
                prediction = self.policy_inference(sync_point)

                # Encode images in response if needed
                return self._encode_output_images(prediction)

            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Prediction failed: {str(e)}"
                )

        @app.post(SET_CHECKPOINT_ENDPOINT)
        async def set_checkpoint(request: CheckpointRequest) -> None:
            try:
                self.policy_inference.set_checkpoint(request.epoch)
            except Exception as e:
                logger.error(f"Checkpoint loading error: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Checkpoint loading failed: {str(e)}"
                )

        return app

    def _decode_image(self, encoded_image: str) -> np.ndarray:
        """Decode base64-encoded image string to numpy array.

        Args:
            encoded_image: Base64-encoded image string from client requests.

        Returns:
            Numpy array representing the decoded image in RGB format.
        """
        img_bytes = base64.b64decode(encoded_image)
        buffer = io.BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        return np.array(pil_image)

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image array to base64 string for response.

        Args:
            image: Numpy array representing an image to encode.

        Returns:
            Base64-encoded string representation of the image.
        """
        pil_image = Image.fromarray(image.astype("uint8"))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
                    camera_data.frame = self._decode_image(camera_data.frame)

        # Decode depth images
        if sync_point.depth_images:
            for camera_name, camera_data in sync_point.depth_images.items():
                if isinstance(camera_data.frame, str):
                    # It's a base64 string, decode it
                    camera_data.frame = self._decode_image(camera_data.frame)

        return sync_point

    def _encode_output_images(self, prediction: ModelPrediction) -> ModelPrediction:
        """Encode output images to base64 for response.

        Args:
            prediction: ModelPrediction with potentially numpy array images

        Returns:
            ModelPrediction with base64-encoded images
        """
        # Handle RGB image outputs
        if DataType.RGB_IMAGE in prediction.outputs:
            rgbs = prediction.outputs[DataType.RGB_IMAGE]
            assert isinstance(rgbs, np.ndarray)
            # Convert numpy arrays to base64 strings
            str_rets = []
            assert len(rgbs.shape) == 6  # [B, T, CAMs, H, W, C]
            for b_idx in range(rgbs.shape[0]):
                batch_images = []
                for t_idx in range(rgbs.shape[1]):
                    time_images = []
                    for cam_idx in range(rgbs.shape[2]):
                        image = rgbs[b_idx, t_idx, cam_idx]
                        if image.shape[0] == 3:  # CHW format
                            image = np.transpose(image, (1, 2, 0))  # Convert to HWC
                        if image.dtype != np.uint8:
                            image = np.clip(image, 0, 255).astype(np.uint8)
                        time_images.append(self._encode_image(image))
                    batch_images.append(time_images)
                str_rets.append(batch_images)
            prediction.outputs[DataType.RGB_IMAGE] = str_rets

        # Handle depth image outputs
        if DataType.DEPTH_IMAGE in prediction.outputs:
            depths = prediction.outputs[DataType.DEPTH_IMAGE]
            assert isinstance(depths, np.ndarray)
            str_rets = []
            assert len(depths.shape) == 5  # [B, T, CAMs, H, W, C]
            for b_idx in range(depths.shape[0]):
                batch_images = []
                for t_idx in range(depths.shape[1]):
                    time_images = []
                    for cam_idx in range(depths.shape[2]):
                        depth = depths[b_idx, t_idx, cam_idx]
                        if depth.shape[0] == 1:  # Remove channel dimension
                            depth = depth[0]
                        # Normalize depth to 0-255 range
                        depth_norm = (
                            (depth - depth.min())
                            / (depth.max() - depth.min() + 1e-8)
                            * 255
                        )
                        depth_norm = depth_norm.astype(np.uint8)
                        time_images.append(self._encode_image(depth_norm))
                    batch_images.append(time_images)
                str_rets.append(batch_images)
                prediction.outputs[DataType.DEPTH_IMAGE] = str_rets

        # Handle other outputs (convert numpy arrays to lists)
        for data_type in [
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.END_EFFECTORS,
            DataType.POSES,
            DataType.POINT_CLOUD,
        ]:
            if data_type in prediction.outputs:
                output_data = prediction.outputs[data_type]
                if isinstance(output_data, np.ndarray):
                    prediction.outputs[data_type] = output_data.tolist()

        # Handle custom data outputs
        if DataType.CUSTOM in prediction.outputs:
            custom_data = prediction.outputs[DataType.CUSTOM]
            if isinstance(custom_data, dict):
                for key, value in custom_data.items():
                    if isinstance(value, np.ndarray):
                        custom_data[key] = value.tolist()
            prediction.outputs[DataType.CUSTOM] = custom_data

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
) -> ModelServer:
    """Start a model server instance.

    Args:
        model_file: Path to the .nc.zip model archive
        org_id: Organization ID
        job_id: Job ID
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level

    Returns:
        ModelServer instance
    """
    server = ModelServer(model_file, org_id, job_id)
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

    args = parser.parse_args()

    start_server(
        model_file=Path(args.model_file),
        org_id=args.org_id,
        job_id=args.job_id,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
