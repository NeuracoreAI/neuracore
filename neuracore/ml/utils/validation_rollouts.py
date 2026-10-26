"""Save a fixed sample of validation timesteps as trajectory rollouts."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import torch
from neuracore_types import BatchedDepthData, BatchedNCData, BatchedRGBData, DataType
from neuracore_types.utils.name_utils import to_safe_name
from PIL import Image
from torch.utils.data import DataLoader

from neuracore.core.utils.depth_utils import depth_to_log_gray
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

ROLLOUT_ROOT_DIR = "validation_rollouts"
INPUT_POINTS_DIR = "input_points"
OUTPUTS_DIR = "outputs"
MANIFEST_NAME = "manifest.json"

_IMAGE_TYPES_DIR_NAMES = {
    DataType.RGB_IMAGES: "rgb",
    DataType.DEPTH_IMAGES: "depth",
}
_PAYLOAD_FIELDS = (
    "value",
    "open_amount",
    "pose",
    "data",
    "input_ids",
    "points",
)


@dataclass(frozen=True)
class ValidationRolloutConfig:
    """How many validation samples to save as rollouts, and how often.

    Attributes:
        num_points: How many validation timesteps to save each time. 0
            disables validation rollouts.
        frequency: Save rollouts when ``epoch % frequency == 0``.
        seed: Seed for which timesteps are saved. The same seed keeps those
            timesteps fixed across epochs.
    """

    num_points: int
    frequency: int
    seed: int

    def __post_init__(self) -> None:
        """Reject settings that are invalid."""
        if self.num_points < 0:
            raise ValueError(
                "Validation rollout num_points must be at least 0, "
                f"got {self.num_points}."
            )
        if self.frequency <= 0:
            raise ValueError(
                "Validation rollout frequency must be greater than 0, "
                f"got {self.frequency}."
            )


def select_validation_rollout_indices(
    validation_dataset_loader: DataLoader,
    validation_rollout_config: ValidationRolloutConfig,
) -> list[int]:
    """Randomly select validation samples from the validation loader's dataset.

    Args:
        validation_dataset_loader: Validation loader whose dataset is a
            ``Subset``; its ``indices`` are the candidates.
        validation_rollout_config: Supplies ``num_points`` (how many indices to
            draw) and ``seed`` (seed for the random draw).

    Returns:
        Indices drawn from the validation subset without replacement, at most
        ``num_points`` of them.

    Raises:
        ValueError: If ``num_points`` is not positive or the split is empty.
    """
    validation_indices = validation_dataset_loader.dataset.indices
    num_points = validation_rollout_config.num_points
    available = len(validation_indices)
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    if available == 0:
        raise ValueError(
            "Cannot select validation rollout points from an empty validation split."
        )
    if num_points > available:
        logger.warning(
            "Requested %s validation rollout point(s), but the validation split "
            "only has %s sample(s). Using all of them.",
            num_points,
            available,
        )
        num_points = available
    generator = np.random.default_rng(validation_rollout_config.seed)
    positions = generator.choice(available, size=num_points, replace=False)
    return [validation_indices[int(position)] for position in positions]


def rollout_root_dir(output_dir: Path) -> Path:
    """Return the directory that holds every validation rollout file."""
    return output_dir / ROLLOUT_ROOT_DIR


def _input_point_dir(root_dir: Path, point: SampleIdentity) -> Path:
    """Return the folder that holds one point's inputs and ground truth."""
    return (
        root_dir / INPUT_POINTS_DIR / point.recording_id / f"point_{point.timestep:06d}"
    )


def _epoch_output_dir(root_dir: Path, epoch: int) -> Path:
    """Return the folder that holds one epoch's predictions."""
    return root_dir / OUTPUTS_DIR / f"epoch_{epoch:04d}"


def _output_point_path(epoch_dir: Path, point: SampleIdentity) -> Path:
    """Return the file that holds one point's predictions for an epoch."""
    return epoch_dir / point.recording_id / f"point_{point.timestep:06d}.json"


def save_validation_rollouts(
    model: NeuracoreModel,
    dataset: PytorchNeuracoreDataset,
    selected_indices: Sequence[int],
    batch_size: int | None,
    device: torch.device,
    inference_device_preprocessing: tuple[
        PreprocessingConfiguration, PreprocessingConfiguration
    ],
    output_dir: Path,
    epoch: int,
    storage_handler: TrainingStorageHandler,
    save_input_points: bool,
) -> None:
    """Run inference on the selected points and queue their prediction files.

    File writes and uploads run on the storage handler's rollout worker.

    Args:
        model: Unwrapped model, put in eval mode for this call only.
        dataset: Validation dataset.
        selected_indices: Dataset indices to run.
        batch_size: Points per ``forward`` call.
        device: Device the model is on.
        inference_device_preprocessing: Device-side input and output
            preprocessing.
        output_dir: Training output directory.
        epoch: Epoch number used in the output directory name.
        storage_handler: Queues the file writes and uploads.
        save_input_points: Also queue each point's inputs and ground truth.
    """
    if batch_size is None:
        logger.warning(
            "Batch size passed to save_validation_rollout is None. " "Defaulting to 1."
        )
        batch_size = 1

    root_dir = rollout_root_dir(output_dir)
    epoch_dir = _epoch_output_dir(root_dir, epoch)
    logger.info(
        "Running %s validation rollout point(s) for epoch %s in batches of %s",
        len(selected_indices),
        epoch,
        batch_size,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(selected_indices), batch_size):
                chunk = selected_indices[start : start + batch_size]
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
                if save_input_points:
                    batch = batch.to(cpu_device)
                outputs_masks = {
                    data_type: mask.to(cpu_device)
                    for data_type, mask in batch.outputs_mask.items()
                }
                predictions = {
                    data_type: [item.to(cpu_device) for item in slots]
                    for data_type, slots in predictions.items()
                }
                for batch_index, index in enumerate(chunk):
                    identity = dataset.get_sample_identity(index)
                    output_description = dataset.output_cross_embodiment_description[
                        identity.robot_id
                    ]
                    outputs_mask = _mask_at_index(outputs_masks, batch_index)
                    named_predictions = assign_names_to_batches(
                        predictions, output_description, masks=outputs_mask
                    )
                    storage_handler.submit_rollout_save(
                        partial(
                            _write_json,
                            _output_point_path(epoch_dir, identity),
                            _convert_traces_to_json(
                                named_predictions, batch_index, horizon=True
                            ),
                        )
                    )
                    if save_input_points:
                        input_description = dataset.input_cross_embodiment_description[
                            identity.robot_id
                        ]
                        named_inputs = assign_names_to_batches(
                            batch.inputs,
                            input_description,
                            masks=_mask_at_index(batch.inputs_mask, batch_index),
                        )
                        named_ground_truth = assign_names_to_batches(
                            batch.outputs, output_description, masks=outputs_mask
                        )
                        storage_handler.submit_rollout_save(
                            partial(
                                _write_input_point,
                                _input_point_dir(root_dir, identity),
                                _read_image_frames(named_inputs, batch_index),
                                _convert_traces_to_json(
                                    named_inputs, batch_index, horizon=False
                                ),
                                _convert_traces_to_json(
                                    named_ground_truth, batch_index, horizon=True
                                ),
                            )
                        )
    finally:
        model.train(was_training)

    if save_input_points:
        storage_handler.submit_rollout_save(
            partial(
                storage_handler.upload_validation_rollouts,
                root_dir / INPUT_POINTS_DIR,
            )
        )
    storage_handler.submit_rollout_save(
        partial(storage_handler.upload_validation_rollouts, epoch_dir)
    )


def _write_input_point(
    point_dir: Path,
    images: Mapping[DataType, Mapping[str, torch.Tensor]],
    state: Mapping[str, JsonValue],
    ground_truth: Mapping[str, JsonValue],
) -> None:
    """Write one point's JPEG frames, state values and ground truth."""
    _write_input_images(point_dir / "inputs" / "images", images)
    _write_json(point_dir / "inputs" / "state_input.json", state)
    _write_json(point_dir / "ground_truth.json", ground_truth)


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
        if data_type not in _IMAGE_TYPES_DIR_NAMES:
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
        if data_type in _IMAGE_TYPES_DIR_NAMES:
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
    root_dir: Path,
    points: Sequence[SampleIdentity],
) -> None:
    """Write the list of points that rollouts are expected for.

    Args:
        root_dir: Directory the manifest sits in.
        points: Identities of the selected points.
    """
    payload: dict[str, JsonValue] = {
        "points": [
            {
                "recording_id": point.recording_id,
                "timestep": point.timestep,
                "robot_id": point.robot_id,
            }
            for point in points
        ],
    }
    _write_json(root_dir / MANIFEST_NAME, payload)


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
        directory = image_root / _IMAGE_TYPES_DIR_NAMES[data_type]
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
    """Write one depth frame as a viewable JPEG."""
    depth = np.squeeze(frame.numpy()).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be 2D after squeezing, got {depth.shape}.")
    Image.fromarray(depth_to_log_gray(depth)).save(path, format="JPEG")


def _channel_last_image(frame: torch.Tensor) -> torch.Tensor:
    """Move a channel-first frame to ``(H, W, C)``."""
    if frame.ndim == 3 and frame.shape[0] == 3:
        return frame.permute(1, 2, 0)
    return frame


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Write JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
