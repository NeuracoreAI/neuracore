"""Save a fixed sample of validation timesteps as trajectory rollouts.

Metric validation still walks every validation sample. This module picks a
smaller set of those timesteps, runs the inference forward on each one, and
writes the images and JSON.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import cast

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
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler

logger = logging.getLogger(__name__)

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
    batch_size: int,
    storage_handler: TrainingStorageHandler,
) -> Path:
    """Run inference on the chosen validation points and write their files.

    Points are collated in chunks of ``batch_size``, and each chunk is one
    ``forward``. Each sample is then named with its own robot and written
    as images and JSON values.

    Args:
        model: Unwrapped model. Called in eval mode for this function only.
        dataset: Validation dataset.
        validation_indices: All sample indices in the validation split.
        device: Device the model is on.
        inference_device_preprocessing: Device-side input and output
            preprocessing, the same pair validation uses.
        output_dir: Training output directory.
        epoch: Epoch number used in the directory name.
        num_points: How many points to save.
        seed: Seed for which points are chosen.
        batch_size: Validation loader batch size. The last chunk may be smaller.
        storage_handler: Queues the file write. The caller returns before the
            files exist. ``wait_for_rollout_saves`` joins that write.

    Returns:
        The epoch directory the background write will fill.
    """
    selected = select_validation_rollout_indices(validation_indices, num_points, seed)
    epoch_dir = output_dir / "validation-rollouts" / f"epoch_{epoch:04d}"
    logger.info(
        "Running %s validation rollout point(s) for epoch %s in batches of %s",
        len(selected),
        epoch,
        batch_size,
    )
    was_training = model.training
    model.eval()
    identities: list[SampleIdentity] = []
    try:
        with torch.no_grad():
            for start in range(0, len(selected), batch_size):
                chunk = selected[start : start + batch_size]
                batch = dataset.collate_fn([dataset[index] for index in chunk]).to(
                    device
                )
                apply_device_preprocessing(batch, *inference_device_preprocessing)
                predictions = model.forward(
                    BatchedInferenceInputs(
                        inputs=batch.inputs,
                        inputs_mask=batch.inputs_mask,
                        batch_size=batch.batch_size,
                    )
                )
                # Move the batch and predictions to CPU before writing to files.
                cpu_device = torch.device("cpu")
                batch = batch.to(cpu_device)
                predictions = {
                    data_type: [item.to(cpu_device) for item in slots]
                    for data_type, slots in predictions.items()
                }
                for batch_index, index in enumerate(chunk):
                    identity = dataset.get_sample_identity(index)
                    identities.append(identity)
                    named_inputs = assign_names_to_batches(
                        batch.inputs,
                        dataset.input_cross_embodiment_description[identity.robot_id],
                        masks=_mask_at_index(batch.inputs_mask, batch_index),
                    )
                    named_predictions = assign_names_to_batches(
                        predictions,
                        dataset.output_cross_embodiment_description[identity.robot_id],
                        masks=_mask_at_index(batch.outputs_mask, batch_index),
                    )
                    named_ground_truth = assign_names_to_batches(
                        batch.outputs,
                        dataset.output_cross_embodiment_description[identity.robot_id],
                        masks=_mask_at_index(batch.outputs_mask, batch_index),
                    )
                    point_dir = (
                        epoch_dir
                        / identity.recording_id
                        / f"point_{identity.timestep:06d}"
                    )
                    images = _read_image_frames(named_inputs, batch_index)
                    state = _convert_traces_to_json(
                        named_inputs, batch_index, horizon=False
                    )
                    prediction_values = _convert_traces_to_json(
                        named_predictions, batch_index, horizon=True
                    )
                    ground_truth = _convert_traces_to_json(
                        named_ground_truth, batch_index, horizon=True
                    )
                    storage_handler.submit_rollout_save(
                        partial(
                            _write_rollout_point,
                            point_dir,
                            images,
                            state,
                            prediction_values,
                            ground_truth,
                        )
                    )
    finally:
        model.train(was_training)

    storage_handler.submit_rollout_save(
        partial(
            _write_manifest_and_upload,
            epoch_dir,
            epoch,
            identities,
            storage_handler,
        )
    )
    return epoch_dir


def _write_manifest_and_upload(
    epoch_dir: Path,
    epoch: int,
    identities: list[SampleIdentity],
    storage_handler: TrainingStorageHandler,
) -> None:
    """Write the epoch manifest, then enqueue the upload of that epoch."""
    write_rollout_manifest(epoch_dir, epoch, identities)
    storage_handler.upload_validation_rollouts(epoch_dir)


def _write_rollout_point(
    point_dir: Path,
    images: Mapping[DataType, Mapping[str, torch.Tensor]],
    state: Mapping[str, JsonValue],
    predictions: Mapping[str, JsonValue],
    ground_truth: Mapping[str, JsonValue],
) -> None:
    """Write one point's JPEG frames and JSON trace values."""
    _write_input_images(point_dir / "inputs" / "images", images)
    _write_json(point_dir / "inputs" / "state_input.json", state)
    _write_json(point_dir / "outputs" / "prediction_horizon.json", predictions)
    _write_json(point_dir / "outputs" / "ground_truth.json", ground_truth)


def _mask_at_index(
    masks: Mapping[DataType, torch.Tensor],
    batch_index: int,
) -> dict[DataType, torch.Tensor]:
    """Return the slot mask for one sample in a collated batch."""
    return {data_type: mask[batch_index] for data_type, mask in masks.items()}


def _read_image_frames(
    named_inputs: Mapping[DataType, Mapping[str, BatchedNCData]],
    batch_index: int,
) -> dict[DataType, dict[str, torch.Tensor]]:
    """Read each camera frame for one sample. Shape is channel-first."""
    frames: dict[DataType, dict[str, torch.Tensor]] = {}
    for data_type, traces in named_inputs.items():
        if data_type not in _IMAGE_DATA_TYPES:
            continue
        frames[data_type] = {}
        for trace_name, data in traces.items():
            if not isinstance(data, (BatchedRGBData, BatchedDepthData)):
                raise TypeError(
                    f"{data_type.value} trace {trace_name!r} must be camera data "
                    f"to save a validation rollout image, got {type(data).__name__}."
                )
            frames[data_type][trace_name] = data.frame[batch_index, 0]
    return frames


def _convert_traces_to_json(
    named_traces: Mapping[DataType, Mapping[str, BatchedNCData]],
    batch_index: int,
    *,
    horizon: bool,
) -> dict[str, JsonValue]:
    """Build ``data type -> trace name -> JSON value`` for one sample.

    ``horizon`` false keeps the single input timestep. ``horizon`` true keeps
    every timestep, as a list.
    """
    document: dict[str, JsonValue] = {}
    for data_type, traces in named_traces.items():
        if data_type in _IMAGE_DATA_TYPES:
            continue
        values: dict[str, JsonValue] = {}
        for trace_name, data in traces.items():
            tensor = _payload_tensor(data)[batch_index]
            if not horizon:
                tensor = tensor[0]
            if tensor.ndim > 0 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            if tensor.ndim == 0:
                number = tensor.item()
                values[trace_name] = (
                    float(number) if isinstance(number, float) else int(number)
                )
            else:
                values[trace_name] = tensor.tolist()
        document[data_type.value] = cast(JsonValue, values)
    return document


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
    """Return the tensor that holds a trace's values.

    Raises:
        ValueError: If this data type has none of the known value fields.
    """
    for field_name in _PAYLOAD_FIELDS:
        value = getattr(data, field_name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError(
        f"{type(data).__name__} has no value field among {', '.join(_PAYLOAD_FIELDS)}."
    )


def _write_input_images(
    image_root: Path,
    images: Mapping[DataType, Mapping[str, torch.Tensor]],
) -> None:
    """Write one sample's RGB and depth frames as JPEGs."""
    for data_type, frames in images.items():
        directory = image_root / _IMAGE_DIR_NAMES[data_type]
        directory.mkdir(parents=True, exist_ok=True)
        for trace_name, frame in frames.items():
            path = directory / f"{to_safe_name(trace_name)}.jpeg"
            if data_type is DataType.RGB_IMAGES:
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


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Write JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
