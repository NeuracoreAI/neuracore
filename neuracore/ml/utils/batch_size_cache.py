"""On-disk cache for batch-size autotune results.

Autotuning probes several batch sizes, and every probe spawns a subprocess that
pays a fresh CUDA context, builds the model, and runs a few training steps. The
answer only depends on the model, the shapes it is fed, and the GPU it runs on —
none of which change between runs of the same configuration — so it is worth
remembering.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch
from neuracore_types import ModelInitDescription

from neuracore.core.const import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

BATCH_SIZE_CACHE_DIR = DEFAULT_CACHE_DIR / "batch_size_cache"


def _device_fingerprint(device: torch.device) -> dict[str, Any] | None:
    """Describe the GPU well enough to know the answer does not transfer.

    Returns None if the device cannot be identified, which disables caching —
    a key that cannot distinguish two GPUs would hand back a batch size tuned
    for the wrong one.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"type": device.type}
    try:
        properties = torch.cuda.get_device_properties(device)
    except (RuntimeError, AssertionError) as exc:
        logger.warning("Could not read properties for %s: %s", device, exc)
        return None
    return {
        "type": "cuda",
        "name": properties.name,
        "total_memory": properties.total_memory,
    }


def batch_size_cache_key(
    algorithm_name: str | None,
    algorithm_id: str | None,
    algorithm_config: dict[str, Any],
    model_init_description: ModelInitDescription,
    device: torch.device,
) -> str | None:
    """Build a stable cache key for one autotune result, or None if not cacheable.

    Anything that could change the peak memory of a training step belongs in
    here. Getting this wrong in the conservative direction only costs a cache
    miss; getting it wrong in the permissive direction hands back a batch size
    that OOMs, so prefer including a field over omitting it.
    """
    device_fingerprint = _device_fingerprint(device)
    if device_fingerprint is None:
        return None
    spec = {
        "algorithm_name": algorithm_name,
        "algorithm_id": algorithm_id,
        "algorithm_config": algorithm_config,
        "input_data_types": sorted(
            data_type.value for data_type in model_init_description.input_data_types
        ),
        "output_data_types": sorted(
            data_type.value for data_type in model_init_description.output_data_types
        ),
        "input_slot_counts": {
            data_type.value: len(stats)
            for data_type, stats in sorted(
                model_init_description.input_dataset_statistics.items(),
                key=lambda item: item[0].value,
            )
        },
        "output_slot_counts": {
            data_type.value: len(stats)
            for data_type, stats in sorted(
                model_init_description.output_dataset_statistics.items(),
                key=lambda item: item[0].value,
            )
        },
        "output_prediction_horizon": (model_init_description.output_prediction_horizon),
        "device": device_fingerprint,
        "torch_version": torch.__version__,
    }
    serialized = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_key: str) -> Path:
    return BATCH_SIZE_CACHE_DIR / f"{cache_key}.json"


def load_cached_batch_size(cache_key: str) -> int | None:
    """Return the cached autotuned batch size for ``cache_key``, or None on a miss.

    A malformed or unreadable entry is treated as a miss rather than an error:
    the cost of recomputing is bounded, and failing a training run over a bad
    cache file is not a good trade.
    """
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        batch_size = int(cached["batch_size"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable batch size cache at %s: %s", path, exc)
        return None
    if batch_size <= 0:
        logger.warning("Ignoring non-positive cached batch size at %s", path)
        return None
    return batch_size


def store_cached_batch_size(cache_key: str, batch_size: int) -> None:
    """Persist an autotuned ``batch_size`` for ``cache_key``.

    Only autotune results belong here. A batch size that merely passed
    validation is a weaker claim, and storing it would make a later
    ``batch_size="auto"`` run return a value the autotuner never chose.

    Write failures are logged and ignored — a cache that cannot be written is
    not a reason to fail a training run.
    """
    path = _cache_path(cache_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"batch_size": batch_size}, handle)
    except OSError as exc:
        logger.warning("Could not write batch size cache to %s: %s", path, exc)
