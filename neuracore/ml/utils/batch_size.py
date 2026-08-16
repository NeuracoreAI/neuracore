"""Resolving the training batch size, and caching what it costs to find.

Autotuning probes several batch sizes, and every probe spawns a subprocess that
pays a fresh CUDA context, builds the model, and runs a few training steps. The
answer only depends on the model, the shapes it is fed, and the GPU it runs on —
none of which change between runs of the same configuration — so it is worth
remembering.
"""

import gc
import hashlib
import json
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from neuracore_types import CrossEmbodimentDescription, ModelInitDescription
from omegaconf import DictConfig, OmegaConf

from neuracore.core.const import DEFAULT_CACHE_DIR
from neuracore.core.utils.embodiment_description_utils import extract_data_types
from neuracore.ml import NeuracoreModel
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.trainers.batch_autotuner import (
    find_optimal_batch_size,
    is_valid_batch_size,
)
from neuracore.ml.utils.device_utils import get_default_device

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


def assert_valid_batch_size(
    batch_size: int,
    cfg: DictConfig,
    dataset: PytorchSynchronizedDataset,
    input_cross_embodiment_description: CrossEmbodimentDescription,
    output_cross_embodiment_description: CrossEmbodimentDescription,
    create_model: Callable[[ModelInitDescription], NeuracoreModel],
    device: torch.device | None = None,
) -> None:
    """Assert that a user-selected batch size fits in GPU memory.

    The check is skipped on CPU (or when CUDA is unavailable). The user-selected
    batch size is trusted in that case.

    Raises:
        ValueError: If ``batch_size`` does not fit in GPU memory.
    """
    if not torch.cuda.is_available() or (
        device is not None and "cuda" not in device.type
    ):
        logger.warning("Skipping batch size memory check: GPU not available.")
        return

    if device is None:
        device = get_default_device()

    logger.info(f"Validating batch size {batch_size} on {device}...")

    dataset_statistics_by_role = dataset.dataset_statistics
    model_init_description = ModelInitDescription(
        input_dataset_statistics=dataset_statistics_by_role["input"],
        output_dataset_statistics=dataset_statistics_by_role["output"],
        input_data_types=extract_data_types(input_cross_embodiment_description),
        output_data_types=extract_data_types(output_cross_embodiment_description),
        output_prediction_horizon=cfg.output_prediction_horizon,
    )
    model_factory = partial(create_model, model_init_description)

    try:
        # dataset safe to pass in here because the probe runs in a spawned subprocess,
        # so it works on a pickled clone and cannot reach this object.
        valid = is_valid_batch_size(
            cfg=cfg,
            model_factory=model_factory,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )
    except Exception:
        logger.error("Batch size validation failed", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not valid:
        raise ValueError(
            f"Batch size {batch_size} is not valid: it does not fit in "
            "memory for the current algorithm, dataset, and GPU type. "
            "Try a smaller batch size, or use batch_size='auto' to automatically "
            "find the largest batch size that fits."
        )

    logger.info(f"Batch size {batch_size} is valid.")


def determine_optimal_batch_size(
    cfg: DictConfig,
    dataset: PytorchSynchronizedDataset,
    input_cross_embodiment_description: CrossEmbodimentDescription,
    output_cross_embodiment_description: CrossEmbodimentDescription,
    create_model: Callable[[ModelInitDescription], NeuracoreModel],
    device: torch.device | None = None,
) -> int:
    """Run batch size autotuning on a single GPU and return the result."""
    if not torch.cuda.is_available() or (
        device is not None and "cuda" not in device.type
    ):
        raise ValueError("Autotuning is only supported on GPUs.")

    if device is None:
        device = get_default_device()

    logger.info(f"Starting batch size autotuning on {device}...")

    dataset_statistics_by_role = dataset.dataset_statistics
    model_init_description = ModelInitDescription(
        input_dataset_statistics=dataset_statistics_by_role["input"],
        output_dataset_statistics=dataset_statistics_by_role["output"],
        input_data_types=extract_data_types(input_cross_embodiment_description),
        output_data_types=extract_data_types(output_cross_embodiment_description),
        output_prediction_horizon=cfg.output_prediction_horizon,
    )
    model_factory = partial(create_model, model_init_description)

    try:
        # dataset safe to pass in here because the probe runs in a spawned subprocess,
        # so it works on a pickled clone and cannot reach this object.
        optimal_batch_size = find_optimal_batch_size(
            cfg=cfg,
            model_factory=model_factory,
            dataset=dataset,
            device=device,
        )
    except Exception:
        logger.error("Batch size autotuning failed", exc_info=True)
        raise
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logger.info(
        f"Autotuning complete. Optimal batch size per GPU: {optimal_batch_size}"
    )

    return optimal_batch_size


def resolve_batch_size(
    cfg: DictConfig,
    batch_size: Any,
    dataset: PytorchSynchronizedDataset,
    input_cross_embodiment_description: CrossEmbodimentDescription,
    output_cross_embodiment_description: CrossEmbodimentDescription,
    create_model: Callable[[ModelInitDescription], NeuracoreModel],
    device: torch.device | None,
) -> int:
    """Resolve the per-GPU batch size, reusing a cached probe result if possible.

    ``batch_size`` is either the string ``"auto"`` or an integer. Autotuning it
    means spawning several probe subprocesses; validating a user-supplied value
    means spawning one. Both results are cached against the model, its input
    shapes, and the GPU, so a repeat run of the same configuration skips
    straight to the answer.

    Args:
        cfg: Fully resolved Hydra configuration.
        batch_size: ``"auto"`` or an integer batch size.
        dataset: Dataset used to build probe batches.
        input_cross_embodiment_description: Input embodiment mapping.
        output_cross_embodiment_description: Output embodiment mapping.
        create_model: Builds a model from a description. Injected so this
            module does not depend on the training entrypoint that owns
            algorithm construction.
        device: Device training will run on.

    Returns:
        The batch size to train with.
    """
    is_auto = isinstance(batch_size, str) and batch_size.lower() == "auto"
    probe_device = device if device is not None else get_default_device()

    # The cache only describes GPU memory behaviour, and neither code path
    # probes anything off-GPU.
    can_cache = probe_device.type == "cuda" and torch.cuda.is_available()
    cache_key: str | None = None
    if can_cache and not cfg.get("force_batch_size_autotune", False):
        dataset_statistics_by_role = dataset.dataset_statistics
        cache_key = batch_size_cache_key(
            algorithm_name=cfg.get("algorithm_name"),
            algorithm_id=cfg.get("algorithm_id"),
            algorithm_config=(
                OmegaConf.to_container(cfg.algorithm_params, resolve=True)
                if cfg.get("algorithm_params") is not None
                else {}
            ),
            model_init_description=ModelInitDescription(
                input_dataset_statistics=dataset_statistics_by_role["input"],
                output_dataset_statistics=dataset_statistics_by_role["output"],
                input_data_types=extract_data_types(input_cross_embodiment_description),
                output_data_types=extract_data_types(
                    output_cross_embodiment_description
                ),
                output_prediction_horizon=cfg.output_prediction_horizon,
            ),
            device=probe_device,
        )
    cached = load_cached_batch_size(cache_key) if cache_key is not None else None
    if cached is not None:
        if is_auto:
            logger.info(f"Using cached autotuned batch size: {cached}")
            return cached
        if int(batch_size) <= cached:
            # The autotuner already found a larger batch size that fits on
            # this GPU for this model, so this one fits too.
            logger.info(
                f"Batch size {int(batch_size)} is within the cached "
                f"autotuned limit of {cached}; skipping validation."
            )
            return int(batch_size)

    if not is_auto:
        resolved = int(batch_size)
        if cfg.get("validate_batch_size", True):
            assert_valid_batch_size(
                batch_size=resolved,
                cfg=cfg,
                dataset=dataset,
                input_cross_embodiment_description=input_cross_embodiment_description,
                output_cross_embodiment_description=output_cross_embodiment_description,
                create_model=create_model,
                device=device,
            )
        # Deliberately not cached: "this size fits" is a weaker claim than the
        # autotuned optimum, and writing it here would make a later
        # batch_size="auto" run return a value the autotuner never chose.
        return resolved

    resolved = determine_optimal_batch_size(
        cfg=cfg,
        dataset=dataset,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        create_model=create_model,
        device=device,
    )
    if cache_key is not None:
        store_cached_batch_size(cache_key, resolved)
    return resolved
