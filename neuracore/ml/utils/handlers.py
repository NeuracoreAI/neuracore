import subprocess
import sys
from pathlib import Path

from neuracore.ml.utils.algorithm_loader import AlgorithmLoader

# Ensure neuracore is installed
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

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from neuracore.core.nc_types import (
    CameraData,
    DatasetDescription,
    JointData,
    ModelInitDescription,
    SyncPoint,
)
from neuracore.ml import BatchedInferenceOutputs, BatchedInferenceSamples, MaskableData

logger = logging.getLogger(__name__)


class RobotModelHandler(BaseHandler):
    """Handler for robot control models in TorchServe."""

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.normalization_stats = None
        self.dataset_description: DatasetDescription = None

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.

        Args:
            model_dir (str): Points to the location of the model artifacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.

        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        algorithm_loader = AlgorithmLoader(Path(model_dir))
        model_class = algorithm_loader.load_model()
        model = model_class(self.model_init_description)
        if model_pt_path:
            model.load_state_dict(torch.load(model_pt_path, weights_only=True))
        return model

    def initialize(self, context):
        """Initialize model and preprocessing."""

        # Get model configuration from dataset description
        model_init_description_path = os.path.join(
            context.system_properties.get("model_dir"), "model_init_description.json"
        )
        with open(model_init_description_path) as f:
            data = json.load(f)
        self.model_init_description = ModelInitDescription.model_validate(data)
        self.dataset_description = self.model_init_description.dataset_description

        super().initialize(context)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        logger.info("Model initialized!")

    def _decode_image(self, encoded_image: str) -> np.ndarray:
        """Decode base64 image string to numpy array."""
        img_bytes = base64.b64decode(encoded_image)
        buffer = io.BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        return np.array(pil_image)

    def _process_joint_data(
        self, joint_data: list[JointData], max_len: int
    ) -> MaskableData:
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
        values = np.zeros((len(image_data), max_len, 3, 224, 224))
        mask = np.zeros((len(image_data), max_len))
        for i, images in enumerate(image_data):
            for j, (camera_name, camera_data) in enumerate(images.items()):
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

    def preprocess(self, requests):
        """Preprocess batch of requests."""
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
                [sp.joint_positions for sp in sync_points],
                self.dataset_description.joint_positions.max_len,
            )

        if sync_points[0].rgb_images:
            batch.rgb_images = self._process_image_data(
                [sp.rgb_images for sp in sync_points],
                self.dataset_description.max_num_rgb_images,
            )
        return batch.to(self.device)

    def inference(self, data: BatchedInferenceSamples) -> torch.Tensor:
        """Run model inference."""
        with torch.no_grad():
            batch_output: BatchedInferenceOutputs = self.model(data)
            return batch_output.action_predicitons[0]

    def postprocess(self, inference_output):
        """Postprocess model output."""
        predictions_np = inference_output.cpu().numpy()
        return [predictions_np.tolist()]
