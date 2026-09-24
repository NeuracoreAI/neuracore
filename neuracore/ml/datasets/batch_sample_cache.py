"""On-disk cache of fully built training samples."""

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from neuracore_types import (
    BatchedNCData,
    BatchedRGBData,
    CrossEmbodimentDescription,
    DataType,
)

from neuracore.core.const import DEFAULT_CACHE_DIR
from neuracore.core.data.cache_manager import CacheManager
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.json_serialization import to_json_serializable

logger = logging.getLogger(__name__)

SAMPLE_CACHE_DIR = DEFAULT_CACHE_DIR / "sample_cache"


def _frame_to_uint8(frame: torch.Tensor) -> torch.Tensor:
    """Round an RGB frame in the range 0 to 255 to uint8."""
    return frame.round().clamp(0, 255).to(torch.uint8)


def _frame_to_float32(frame: torch.Tensor) -> torch.Tensor:
    """Convert an RGB frame to float32."""
    return frame.to(torch.float32)


def _convert_rgb_frames(
    sample: BatchedTrainingSamples,
    convert: Callable[[torch.Tensor], torch.Tensor],
) -> BatchedTrainingSamples:
    """Return a copy of the sample with every RGB frame passed through convert.

    Args:
        sample: The sample to copy. This function leaves it unchanged.
        convert: Function applied to each RGB frame tensor.

    Returns:
        A new sample sharing every tensor except the converted RGB frames.
    """

    def convert_section(
        section: dict[DataType, list[BatchedNCData]],
    ) -> dict[DataType, list[BatchedNCData]]:
        return {
            data_type: [
                (
                    item.model_copy(update={"frame": convert(item.frame)})
                    if isinstance(item, BatchedRGBData)
                    else item
                )
                for item in items
            ]
            for data_type, items in section.items()
        }

    return dataclasses.replace(
        sample,
        inputs=convert_section(sample.inputs),
        outputs=convert_section(sample.outputs),
    )


class BatchSampleCache:
    """Caches built training samples on disk, shared across epochs and runs.

    Building a sample is dominated by inflating a stored frame and then
    discarding most of it during resize, work that is identical on every epoch
    and every run of the same configuration. Caching the finished sample skips
    all of it.

    This works best when augmentation is applied on the training device,
    so the cache stores only the deterministic projection and preprocessing.
    """

    def __init__(
        self,
        synchronized_dataset_id: str,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
        output_prediction_horizon: int,
        input_preprocessing_config: PreprocessingConfiguration,
        output_preprocessing_config: PreprocessingConfiguration,
        root: Path | None = None,
    ) -> None:
        """Initialize the cache.

        Every argument feeds the key, so the call site reads as a statement of
        what makes one cached sample interchangeable with another.

        Args:
            synchronized_dataset_id: Identifies the synchronized dataset, which
                the server keys on the synchronization parameters.
            input_cross_embodiment_description: Which input sensors are
                projected in, and how they are padded.
            output_cross_embodiment_description: The same, for outputs.
            output_prediction_horizon: How many future steps outputs span.
            input_preprocessing_config: Worker-side input preprocessing. Pass
                the worker-side half only; device-side methods run after this
                cache and cannot change what is stored.
            output_preprocessing_config: The same, for outputs.
            root: Directory holding every configuration's entries. Defaults to
                the shared sample cache.
        """
        self.directory = (root or SAMPLE_CACHE_DIR) / self._spec_hash(
            synchronized_dataset_id=synchronized_dataset_id,
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            output_prediction_horizon=output_prediction_horizon,
            input_preprocessing_config=input_preprocessing_config,
            output_preprocessing_config=output_preprocessing_config,
        )
        self.cache_manager = CacheManager(self.directory)

    @classmethod
    def _spec_hash(
        cls,
        synchronized_dataset_id: str,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
        output_prediction_horizon: int,
        input_preprocessing_config: PreprocessingConfiguration,
        output_preprocessing_config: PreprocessingConfiguration,
    ) -> str:
        """Digest everything that changes what a built sample contains.

        Including a field costs at worst a cache miss. Omitting one serves
        tensors that silently disagree with the configuration, so anything
        doubtful belongs here.
        """

        def preprocessing_spec(
            config: PreprocessingConfiguration,
        ) -> dict[str, list[dict[str, object]]]:
            return {
                data_type.value: [method.to_dict() for method in methods]
                for data_type, methods in sorted(
                    config.items(), key=lambda item: item[0].value
                )
            }

        spec = {
            "synchronized_dataset_id": synchronized_dataset_id,
            "input_cross_embodiment_description": to_json_serializable(
                input_cross_embodiment_description
            ),
            "output_cross_embodiment_description": to_json_serializable(
                output_cross_embodiment_description
            ),
            "output_prediction_horizon": output_prediction_horizon,
            "input_preprocessing": preprocessing_spec(input_preprocessing_config),
            "output_preprocessing": preprocessing_spec(output_preprocessing_config),
        }
        serialized = json.dumps(
            spec, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _path(self, recording_id: str, timestep: int) -> Path:
        # Sharded by recording to keep any one directory a reasonable size,
        # mirroring the layout of the decoded frame cache.
        return self.directory / recording_id / f"{timestep}.pt"

    def load(self, recording_id: str, timestep: int) -> BatchedTrainingSamples | None:
        """Return the cached sample, or None if absent or unreadable.

        Args:
            recording_id: Recording the sample was built from.
            timestep: Timestep within that recording.

        Returns:
            The cached sample, or None if the caller should rebuild it.
        """
        path = self._path(recording_id, timestep)
        if not path.exists():
            return None
        try:
            # weights_only=False is needed to rebuild the batched pydantic
            # types. Safe here: these files are written by this process to the
            # local disk and are never fetched from anywhere.
            sample = torch.load(path, weights_only=False)
        except Exception:
            # A truncated or stale entry must cost a rebuild, never a crash.
            logger.warning("Discarding unreadable sample cache entry %s", path)
            path.unlink(missing_ok=True)
            return None
        return _convert_rgb_frames(
            cast(BatchedTrainingSamples, sample), _frame_to_float32
        )

    def store(
        self, recording_id: str, timestep: int, sample: BatchedTrainingSamples
    ) -> None:
        """Write a built sample. A failure here costs a rebuild, nothing more.

        Store RGB frames as uint8.

        Args:
            recording_id: Recording the sample was built from.
            timestep: Timestep within that recording.
            sample: The built sample to store.
        """
        path = self._path(recording_id, timestep)
        staging: Path | None = None
        try:
            self.cache_manager.ensure_space_available()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename: several workers populate this concurrently and
            # a reader must never see a half-written entry.
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                staging = Path(handle.name)
                torch.save(_convert_rgb_frames(sample, _frame_to_uint8), handle)
            os.replace(staging, path)
        except Exception:
            logger.warning("Could not write sample cache entry %s", path, exc_info=True)
            if staging is not None:
                staging.unlink(missing_ok=True)
