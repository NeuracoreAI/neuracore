import base64
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from neuracore.core.nc_types import LanguageData  # Import LanguageData
from neuracore.core.nc_types import (
    CameraData,
    DatasetDescription,
    DataType,
    JointData,
    ModelInitDescription,
    ModelPrediction,
    SyncPoint,
)
from neuracore.ml import BatchedInferenceSamples, MaskableData
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_pt_path:
            model.load_state_dict(
                torch.load(model_pt_path, map_location=self.device, weights_only=True),
            )
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

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        pil_image = Image.fromarray(image.astype("uint8"))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

    def _process_language_data(self, language_data: list[LanguageData]) -> MaskableData:
        """Process language data using tokenizer."""
        # Tokenize all texts in the batch
        texts = [ld.text for ld in language_data]
        input_ids, attention_mask = self.model.tokenize_text(texts)
        return MaskableData(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
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
        if sync_points[0].joint_torques:
            batch.joint_torques = self._process_joint_data(
                [sp.joint_torques for sp in sync_points],
                self.dataset_description.joint_torques.max_len,
            )
        if sync_points[0].joint_velocities:
            batch.joint_velocities = self._process_joint_data(
                [sp.joint_velocities for sp in sync_points],
                self.dataset_description.joint_velocities.max_len,
            )
        if sync_points[0].joint_target_positions:
            batch.joint_target_positions = self._process_joint_data(
                [sp.joint_target_positions for sp in sync_points],
                self.dataset_description.joint_target_positions.max_len,
            )

        if sync_points[0].rgb_images:
            batch.rgb_images = self._process_image_data(
                [sp.rgb_images for sp in sync_points],
                self.dataset_description.max_num_rgb_images,
            )

        # Process language data if available
        if sync_points[0].language_data:
            batch.language_tokens = self._process_language_data(
                [sp.language_data for sp in sync_points],
            )

        return batch.to(self.device)

    def inference(self, data: BatchedInferenceSamples) -> ModelPrediction:
        """Run model inference."""
        with torch.no_grad():
            batch_output: ModelPrediction = self.model(data)
            return batch_output

    def postprocess(self, inference_output: ModelPrediction) -> list[dict]:
        """Postprocess model output."""
        if DataType.RGB_IMAGE in inference_output.outputs:
            # Shape: [B, T, CAMS, 224, 224, 3]
            rgbs = inference_output.outputs[DataType.RGB_IMAGE]
            str_rets = [
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
