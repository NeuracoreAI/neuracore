"""TorchServe handler for robot control model inference and deployment.

This module provides a TorchServe-compatible handler for deploying Neuracore
robot control models. It handles model loading, data preprocessing, inference
execution, and response formatting for robot control applications.
"""

import subprocess
import sys

# Ensure neuracore is installed when running on cloud
# ruff: noqa: E402
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "neuracore",
])

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

from neuracore.core.nc_types import LanguageData  # Import LanguageData
from neuracore.core.nc_types import (
    CameraData,
    DataType,
    JointData,
    ModelInitDescription,
    ModelPrediction,
    SyncPoint,
)
from neuracore.ml import BatchedInferenceSamples, MaskableData, NeuracoreModel
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader

logger = logging.getLogger(__name__)


class RobotModelHandler(BaseHandler):
    """TorchServe handler for robot control model inference.

    This handler extends TorchServe's BaseHandler to provide specialized
    functionality for robot control models including multi-modal data
    preprocessing, model loading via AlgorithmLoader, and structured
    output formatting for robot control applications.
    """

    def __init__(self) -> None:
        """Initialize the robot model handler with default configuration."""
        super().__init__()
        self.initialized = False
        self.normalization_stats = None

    def _load_pickled_model(
        self, model_dir: str, model_file: str, model_pt_path: Optional[str]
    ) -> NeuracoreModel:
        """Load a Neuracore model using the AlgorithmLoader.

        Dynamically loads the model class from the algorithm directory and
        initializes it with the appropriate configuration. Supports loading
        pre-trained weights if available.

        Args:
            model_dir: Directory containing the model artifacts and algorithm code.
            model_file: Name of the Python file containing the model class.
            model_pt_path: Path to the pre-trained model weights file, or None for
            randomly initialized models.

        Returns:
            The loaded and initialized model instance ready for inference.

        Raises:
            RuntimeError: If the model.py file is missing (legacy check).
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        algorithm_loader = AlgorithmLoader(Path(model_dir))
        model_class = algorithm_loader.load_model()
        model = model_class(self.model_init_description, **self.algorithm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_pt_path:
            model.load_state_dict(
                torch.load(model_pt_path, map_location=self.device, weights_only=True),
            )
        return model

    def initialize(self, context: Context) -> None:
        """Initialize the model handler with TorchServe context.

        Loads model configuration, initializes the model using the parent class
        infrastructure, and prepares the handler for inference requests.

        Args:
            context: TorchServe context object containing system properties
                and configuration information.
        """
        # Get model configuration from dataset description
        model_init_description_path = os.path.join(
            context.system_properties.get("model_dir"), "model_init_description.json"
        )
        with open(model_init_description_path) as f:
            data = json.load(f)
        # Get custom algorithm configuration if available
        algorithm_config_path = os.path.join(
            context.system_properties.get("model_dir"), "algorithm_config.json"
        )
        if os.path.isfile(algorithm_config_path):
            with open(algorithm_config_path) as f:
                self.algorithm_config = json.load(f)
        else:
            self.algorithm_config = {}
        self.model_init_description = ModelInitDescription.model_validate(data)
        self.dataset_description = self.model_init_description.dataset_description

        super().initialize(context)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True
        logger.info("Model initialized!")

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

    def _process_joint_data(
        self, joint_data: list[JointData], max_len: int
    ) -> MaskableData:
        """Process joint state data into batched tensor format.

        Converts joint data from multiple samples into a batched tensor with
        appropriate padding and masking for variable-length joint configurations.

        Args:
            joint_data: List of JointData objects from the batch.
            max_len: Maximum joint dimension for padding.

        Returns:
            MaskableData containing batched joint values and attention masks.
        """
        values = np.zeros((len(joint_data), max_len))
        mask = np.zeros((len(joint_data), max_len))
        for i, jd in enumerate(joint_data):
            v = list(jd.values.values())
            values[i, : len(v)] = v
            mask[i, : len(v)] = 1.0
        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_image_data(
        self, image_data: list[dict[str, CameraData]], max_len: int
    ) -> MaskableData:
        """Process camera image data into batched tensor format.

        Decodes base64 images, applies standard preprocessing transforms,
        and creates batched tensors with masking for variable numbers of cameras.

        Args:
            image_data: List of dictionaries mapping camera names to CameraData.
            max_len: Maximum number of cameras to support with padding.

        Returns:
            MaskableData containing batched image tensors and attention masks.
        """
        values = np.zeros((len(image_data), max_len, 3, 224, 224))
        mask = np.zeros((len(image_data), max_len))
        for i, images in enumerate(image_data):
            for j, (camera_name, camera_data) in enumerate(images.items()):
                assert isinstance(
                    camera_data.frame, str
                ), f"Expected string fram data, got {type(camera_data.frame)}"
                image = self._decode_image(camera_data.frame)
                image = Image.fromarray(image)
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                ])
                values[i, j] = transform(image)
                mask[i, j] = 1.0
        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_language_data(self, language_data: list[LanguageData]) -> MaskableData:
        """Process natural language instruction data using model tokenizer.

        Tokenizes text instructions into input IDs and attention masks using
        the model's built-in tokenization functionality.

        Args:
            language_data: List of LanguageData objects containing text instructions.

        Returns:
            MaskableData containing tokenized text and attention masks.
        """
        # Tokenize all texts in the batch
        texts = [ld.text for ld in language_data]
        input_ids, attention_mask = self.model.tokenize_text(texts)
        return MaskableData(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
        )

    def preprocess(self, requests: list[dict]) -> BatchedInferenceSamples:
        """Preprocess incoming requests into model-compatible format.

        Converts raw HTTP requests containing SyncPoint data into batched
        tensors suitable for model inference. Handles multiple data modalities
        including joint states, images, and language instructions.

        Args:
            requests: List of HTTP requests containing JSON-encoded SyncPoint data.

        Returns:
            BatchedInferenceSamples object ready for model inference.
        """
        batch = BatchedInferenceSamples()
        sync_points: list[SyncPoint] = []
        for req in requests:
            data = req.get("data") or req.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            if isinstance(data, str):
                data = json.loads(data)
            sync_points.append(SyncPoint.model_validate(data))

        if sync_points[0].joint_positions:
            batch.joint_positions = self._process_joint_data(
                [
                    sp.joint_positions
                    for sp in sync_points
                    if sp.joint_positions is not None
                ],
                self.dataset_description.joint_positions.max_len,
            )
        if sync_points[0].joint_torques:
            batch.joint_torques = self._process_joint_data(
                [
                    sp.joint_torques
                    for sp in sync_points
                    if sp.joint_torques is not None
                ],
                self.dataset_description.joint_torques.max_len,
            )
        if sync_points[0].joint_velocities:
            batch.joint_velocities = self._process_joint_data(
                [
                    sp.joint_velocities
                    for sp in sync_points
                    if sp.joint_velocities is not None
                ],
                self.dataset_description.joint_velocities.max_len,
            )
        if sync_points[0].joint_target_positions:
            batch.joint_target_positions = self._process_joint_data(
                [
                    sp.joint_target_positions
                    for sp in sync_points
                    if sp.joint_target_positions is not None
                ],
                self.dataset_description.joint_target_positions.max_len,
            )

        if sync_points[0].rgb_images:
            batch.rgb_images = self._process_image_data(
                [sp.rgb_images for sp in sync_points if sp.rgb_images is not None],
                self.dataset_description.max_num_rgb_images,
            )

        # Process language data if available
        if sync_points[0].language_data:
            batch.language_tokens = self._process_language_data(
                [
                    sp.language_data
                    for sp in sync_points
                    if sp.language_data is not None
                ],
            )

        return batch.to(self.device)

    def inference(self, data: BatchedInferenceSamples) -> ModelPrediction:
        """Execute model inference on preprocessed data.

        Runs the model forward pass with gradient computation disabled for
        efficient inference.

        Args:
            data: Preprocessed batch of inference samples.

        Returns:
            ModelPrediction containing the model's output predictions.
        """
        with torch.no_grad():
            batch_output: ModelPrediction = self.model(data)
            return batch_output

    def postprocess(self, inference_output: ModelPrediction) -> list[dict]:
        """Postprocess model predictions into client-compatible format.

        Converts model output tensors back to JSON-serializable formats,
        including base64 encoding for images and list conversion for
        numerical data.

        Args:
            inference_output: Raw model predictions from the inference step.

        Returns:
            List containing a single dictionary with the formatted model
            predictions ready for HTTP response.
        """
        if DataType.RGB_IMAGE in inference_output.outputs:
            # Shape: [B, T, CAMS, 224, 224, 3]
            rgbs = inference_output.outputs[DataType.RGB_IMAGE]
            str_rets: List[List[Any]] = [
                [[] for _ in range(rgbs.shape[1])] for _ in range(rgbs.shape[0])
            ]
            for b_idx in range(rgbs.shape[0]):
                for t_idx in range(rgbs.shape[1]):
                    for cam_idx in range(rgbs.shape[2]):
                        image = rgbs[b_idx, t_idx, cam_idx]
                        if image.shape[0] == 3:
                            image = np.transpose(image, (1, 2, 0))
                        if image.dtype != np.uint8:
                            image = np.clip(image, 0, 255).astype(np.uint8)
                        str_rets[b_idx][t_idx].append(self._encode_image(image))
            inference_output.outputs[DataType.RGB_IMAGE] = str_rets
        if DataType.JOINT_TARGET_POSITIONS in inference_output.outputs:
            # Shape: [B, T, DIM]
            joint_target_positions = inference_output.outputs[
                DataType.JOINT_TARGET_POSITIONS
            ]
            inference_output.outputs[DataType.JOINT_TARGET_POSITIONS] = (
                joint_target_positions.tolist()
            )
        return [inference_output.model_dump()]
