"""Save a fixed sample of validation timesteps as trajectory rollouts.

Metric validation still walks every validation sample. This module only
handles the smaller snapshot: which indices to keep, and how to write the
images and JSON for one of those points.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from neuracore_types import BatchedDepthData, BatchedNCData, BatchedRGBData, DataType
from neuracore_types.utils.depth_utils import depth_to_rgb
from neuracore_types.utils.name_utils import to_safe_name
from PIL import Image

from neuracore.ml.datasets.pytorch_neuracore_dataset import (
    PytorchNeuracoreDataset,
    SampleIdentity,
)
from neuracore.ml.utils.json_serialization import JsonValue

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
    dataset: PytorchNeuracoreDataset,
    validation_indices: Sequence[int],
    num_points: int,
    seed: int,
) -> list[int]:
    """Pick ``num_points`` validation samples, spread across episodes.

    The same seed always returns the same indices, so later epochs snapshot
    the same timesteps. Points are taken round-robin across recordings so one
    long episode does not fill the budget.

    Args:
        dataset: Dataset that owns ``validation_indices``.
        validation_indices: Sample indices that belong to the validation split.
        num_points: How many indices to return. Zero returns an empty list.
        seed: Seed for the episode order and the timestep chosen in each one.

    Returns:
        Dataset indices. Fewer than ``num_points`` when the validation split
        is smaller than the request.
    """
    if num_points <= 0 or len(validation_indices) == 0:
        return []

    indices_by_recording: dict[str, list[int]] = defaultdict(list)
    for index in validation_indices:
        identity = dataset.get_sample_identity(index)
        indices_by_recording[identity.recording_id].append(index)

    generator = np.random.default_rng(seed)
    recording_ids = sorted(indices_by_recording)
    recording_order = [
        recording_ids[int(position)]
        for position in generator.permutation(len(recording_ids))
    ]
    pools: dict[str, deque[int]] = {}
    for recording_id in recording_order:
        indices = indices_by_recording[recording_id]
        order = generator.permutation(len(indices))
        pools[recording_id] = deque(indices[int(position)] for position in order)

    selected: list[int] = []
    while len(selected) < num_points:
        took_a_point = False
        for recording_id in recording_order:
            pool = pools[recording_id]
            if not pool:
                continue
            selected.append(pool.popleft())
            took_a_point = True
            if len(selected) == num_points:
                break
        if not took_a_point:
            break
    return selected


def rollout_epoch_dir(output_dir: Path, epoch: int) -> Path:
    """Return the directory for one epoch of rollout snapshots.

    Args:
        output_dir: Training output directory.
        epoch: Epoch number being snapshotted.

    Returns:
        ``output_dir/validation-rollouts/epoch_XXXX``.
    """
    return output_dir / "validation-rollouts" / f"epoch_{epoch:04d}"


def rollout_point_dir(epoch_dir: Path, recording_id: str, timestep: int) -> Path:
    """Return the directory for one saved timestep.

    Args:
        epoch_dir: Epoch directory from ``rollout_epoch_dir``.
        recording_id: Recording that owns the timestep.
        timestep: Point index inside that recording.

    Returns:
        ``epoch_dir/{recording_id}/point_XXXXXX``.
    """
    return epoch_dir / recording_id / f"point_{timestep:06d}"


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


def write_rollout_point(
    point_dir: Path,
    inputs: NamedTraces,
    predictions: NamedTraces,
    ground_truth: NamedTraces,
) -> None:
    """Write one point's images and JSON files.

    Args:
        point_dir: Destination directory for this recording timestep.
        inputs: Named model inputs. Images are written as PNG files and the
            other traces go to ``state_input.json``.
        predictions: Named inference outputs, written as horizon lists.
        ground_truth: Named dataset outputs, written as horizon lists.
    """
    _write_input_images(point_dir / "inputs" / "images", inputs)
    _write_trace_json(
        point_dir / "inputs" / "state_input.json",
        inputs,
        horizon=False,
    )
    _write_trace_json(
        point_dir / "outputs" / "prediction_horizon.json",
        predictions,
        horizon=True,
    )
    _write_trace_json(
        point_dir / "outputs" / "ground_truth.json",
        ground_truth,
        horizon=True,
    )


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
    """Write RGB and depth PNGs for the input timestep."""
    for data_type, traces in inputs.items():
        if data_type not in _IMAGE_DATA_TYPES:
            continue
        directory = image_root / _IMAGE_DIR_NAMES[data_type]
        directory.mkdir(parents=True, exist_ok=True)
        for trace_name, data in traces.items():
            path = directory / f"{to_safe_name(trace_name)}.png"
            if isinstance(data, BatchedRGBData):
                _write_rgb_png(data.frame[0, 0].detach().cpu(), path)
            elif isinstance(data, BatchedDepthData):
                _write_depth_png(data.frame[0, 0].detach().cpu(), path)


def _write_rgb_png(frame: torch.Tensor, path: Path) -> None:
    """Write one RGB frame. Training frames are float pixels in ``[0, 255]``."""
    image = _channel_last_image(frame).numpy()
    if image.dtype != np.uint8:
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, format="PNG")


def _write_depth_png(frame: torch.Tensor, path: Path) -> None:
    """Write one depth frame using the platform's RGB depth encoding."""
    depth = np.squeeze(frame.numpy()).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be 2D after squeezing, got {depth.shape}.")
    Image.fromarray(depth_to_rgb(depth)).save(path, format="PNG")


def _channel_last_image(frame: torch.Tensor) -> torch.Tensor:
    """Move a channel-first frame to ``(H, W, C)``."""
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
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
