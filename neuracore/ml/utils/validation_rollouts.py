"""Save a fixed sample of validation timesteps as trajectory rollouts.

Metric validation still walks every validation sample. This module picks a
smaller set of those timesteps, runs the inference forward on each one, and
writes the images and JSON.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from neuracore_types import BatchedDepthData, BatchedNCData, BatchedRGBData, DataType
from neuracore_types.utils.depth_utils import depth_to_rgb
from neuracore_types.utils.name_utils import to_safe_name
from PIL import Image

from neuracore.ml.core.ml_types import BatchedInferenceInputs
from neuracore.ml.core.neuracore_model import NeuracoreModel
from neuracore.ml.datasets.pytorch_neuracore_dataset import (
    PytorchNeuracoreDataset,
    SampleIdentity,
)
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.embodiment_names import assign_names_to_batches
from neuracore.ml.utils.json_serialization import JsonValue
from neuracore.ml.utils.preprocessing import apply_device_preprocessing

logger = logging.getLogger(__name__)

NamedTraces: TypeAlias = Mapping[DataType, Mapping[str, BatchedNCData]]

_IMAGE_DATA_TYPES = frozenset({DataType.RGB_IMAGES, DataType.DEPTH_IMAGES})
_IMAGE_DIR_NAMES = {
    DataType.RGB_IMAGES: "rgb",
    DataType.DEPTH_IMAGES: "depths",
}
_PAYLOAD_FIELDS = (
    "value",
    "open_amount",
    "pose",
    "data",
    "input_ids",
    "points",
)
_SKIPPED_TENSOR_FIELDS = frozenset({
    "extrinsics",
    "intrinsics",
    "attention_mask",
    "rgb_points",
})


def select_validation_rollout_indices(
    validation_indices: Sequence[int],
    num_points: int,
    seed: int,
) -> list[int]:
    """Randomly select ``num_points`` validation samples from the validation indices.

    Args:
        validation_indices: Sample indices that belong to the validation split.
        num_points: How many indices to return.
        seed: Seed for the random draw.

    Returns:
        ``num_points`` indices drawn from ``validation_indices`` without
        replacement.

    Raises:
        ValueError: If ``num_points`` is not positive, or if it is greater
            than the number of validation indices.
    """
    available = len(validation_indices)
    if num_points <= 0 or num_points > available:
        raise ValueError(
            "num_points must be positive and no greater than the number of "
            f"validation samples ({available}), got {num_points}."
        )
    generator = np.random.default_rng(seed)
    positions = generator.choice(available, size=num_points, replace=False)
    return [validation_indices[int(position)] for position in positions]


def convert_trace_value(data: BatchedNCData, *, horizon: bool) -> JsonValue:
    """Convert one named trace into a JSON value.

    A state input is the value at the single input timestep. A horizon is the
    same value at every predicted timestep, as a list. A trailing length-1
    dimension, used by joints and grippers, is removed so those traces are
    plain numbers.

    Args:
        data: Batched trace. The batch dimension is taken at index 0.
        horizon: When True, keep the time axis. When False, keep timestep 0.

    Returns:
        A JSON number, or a list of them for vectors and horizons.
    """
    tensor = _payload_tensor(data)[0].detach().cpu()
    if not horizon:
        tensor = tensor[0]
    if tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim == 0:
        number = tensor.item()
        if isinstance(number, float):
            return float(number)
        return int(number)
    return tensor.tolist()


def save_validation_rollouts(
    model: NeuracoreModel,
    dataset: PytorchNeuracoreDataset,
    validation_indices: Sequence[int],
    device: torch.device,
    inference_device_preprocessing: tuple[
        PreprocessingConfiguration, PreprocessingConfiguration
    ],
    output_dir: Path,
    epoch: int,
    num_points: int,
    seed: int,
) -> Path:
    """Run inference on the chosen validation points and write their files.

    Each index is loaded on its own. Samples are already batch size 1, which
    is what ``forward`` expects, so this does not go through the validation
    loader. The caller decides which epochs to call this on.

    Args:
        model: Unwrapped model. Called in eval mode for this function only.
        dataset: Dataset that owns ``validation_indices``.
        validation_indices: Every sample in the validation split.
        device: Device the model is on.
        inference_device_preprocessing: Device-side input and output
            preprocessing, the same pair validation uses.
        output_dir: Training output directory.
        epoch: Epoch number used in the directory name.
        num_points: How many points to save.
        seed: Seed for which points are chosen.

    Returns:
        The epoch directory that holds the saved points.
    """
    selected = select_validation_rollout_indices(validation_indices, num_points, seed)
    epoch_dir = output_dir / "validation-rollouts" / f"epoch_{epoch:04d}"
    logger.info(
        "Saving %s validation rollout point(s) for epoch %s",
        len(selected),
        epoch,
    )
    was_training = model.training
    model.eval()
    identities: list[SampleIdentity] = []
    try:
        with torch.no_grad():
            for index in selected:
                identity = dataset.get_sample_identity(index)
                batch = dataset[index].to(device)
                apply_device_preprocessing(batch, *inference_device_preprocessing)
                predictions = model.forward(
                    BatchedInferenceInputs(
                        inputs=batch.inputs,
                        inputs_mask=batch.inputs_mask,
                        batch_size=batch.batch_size,
                    )
                )
                input_description = dataset.input_cross_embodiment_description[
                    identity.robot_id
                ]
                output_description = dataset.output_cross_embodiment_description[
                    identity.robot_id
                ]
                point_dir = (
                    epoch_dir / identity.recording_id / f"point_{identity.timestep:06d}"
                )
                inputs = assign_names_to_batches(
                    batch.inputs, input_description, masks=batch.inputs_mask
                )
                named_predictions = assign_names_to_batches(
                    predictions, output_description, masks=batch.outputs_mask
                )
                ground_truth = assign_names_to_batches(
                    batch.outputs, output_description, masks=batch.outputs_mask
                )
                _write_input_images(point_dir / "inputs" / "images", inputs)
                _write_trace_json(
                    point_dir / "inputs" / "state_input.json",
                    inputs,
                    horizon=False,
                )
                _write_trace_json(
                    point_dir / "outputs" / "prediction_horizon.json",
                    named_predictions,
                    horizon=True,
                )
                _write_trace_json(
                    point_dir / "outputs" / "ground_truth.json",
                    ground_truth,
                    horizon=True,
                )
                identities.append(identity)
    finally:
        model.train(was_training)

    write_rollout_manifest(epoch_dir, epoch, identities)
    return epoch_dir


def write_rollout_manifest(
    epoch_dir: Path,
    epoch: int,
    points: Sequence[SampleIdentity],
) -> None:
    """Write the list of points saved for one epoch.

    Args:
        epoch_dir: Epoch directory that holds the point folders.
        epoch: Epoch number being snapshotted.
        points: Identities of the points written under ``epoch_dir``.
    """
    payload: dict[str, JsonValue] = {
        "epoch": epoch,
        "points": [
            {
                "recording_id": point.recording_id,
                "timestep": point.timestep,
                "robot_id": point.robot_id,
            }
            for point in points
        ],
    }
    _write_json(epoch_dir / "manifest.json", payload)


def _payload_tensor(data: BatchedNCData) -> torch.Tensor:
    """Return the tensor that holds a trace's values."""
    for field_name in _PAYLOAD_FIELDS:
        value = getattr(data, field_name, None)
        if isinstance(value, torch.Tensor):
            return value
    for field_name in data.__class__.model_fields:
        if field_name in _SKIPPED_TENSOR_FIELDS:
            continue
        value = getattr(data, field_name)
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError(f"{type(data).__name__} has no value tensor to save.")


def _write_input_images(image_root: Path, inputs: NamedTraces) -> None:
    """Write RGB and depth JPEGs for the input timestep."""
    for data_type, traces in inputs.items():
        if data_type not in _IMAGE_DATA_TYPES:
            continue
        directory = image_root / _IMAGE_DIR_NAMES[data_type]
        directory.mkdir(parents=True, exist_ok=True)
        expected = (
            BatchedRGBData if data_type is DataType.RGB_IMAGES else BatchedDepthData
        )
        for trace_name, data in traces.items():
            if not isinstance(data, expected):
                raise TypeError(
                    f"{data_type.value} trace {trace_name!r} must be "
                    f"{expected.__name__} to save a validation rollout image, "
                    f"got {type(data).__name__}."
                )
            path = directory / f"{to_safe_name(trace_name)}.jpeg"
            frame = data.frame[0, 0].detach().cpu()
            if isinstance(data, BatchedRGBData):
                _write_rgb_jpeg(frame, path)
            else:
                _write_depth_jpeg(frame, path)


def _write_rgb_jpeg(frame: torch.Tensor, path: Path) -> None:
    """Write one RGB frame. Training frames are float pixels in ``[0, 255]``."""
    image = _channel_last_image(frame).numpy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, format="JPEG")


def _write_depth_jpeg(frame: torch.Tensor, path: Path) -> None:
    """Write one depth frame using the platform's RGB depth encoding."""
    depth = np.squeeze(frame.numpy()).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be 2D after squeezing, got {depth.shape}.")
    Image.fromarray(depth_to_rgb(depth)).save(path, format="JPEG")


def _channel_last_image(frame: torch.Tensor) -> torch.Tensor:
    """Move a channel-first frame to ``(H, W, C)``."""
    if frame.ndim == 3 and frame.shape[0] == 3:
        return frame.permute(1, 2, 0)
    return frame


def _write_trace_json(path: Path, traces: NamedTraces, *, horizon: bool) -> None:
    """Write ``data type -> trace name -> value`` JSON, skipping images."""
    document: dict[str, JsonValue] = {}
    for data_type, named_traces in traces.items():
        if data_type in _IMAGE_DATA_TYPES:
            continue
        document[data_type.value] = {
            trace_name: convert_trace_value(data, horizon=horizon)
            for trace_name, data in named_traces.items()
        }
    _write_json(path, document)


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Write JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
