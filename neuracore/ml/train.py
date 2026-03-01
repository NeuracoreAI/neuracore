"""Hydra-based training script for Neuracore models."""

import gc
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.multiprocessing as mp
from names_generator import generate_name
from neuracore_types import BatchedNCData, ModelInitDescription
from neuracore_types.nc_data import DataType
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, random_split

import neuracore as nc
from neuracore.api.training import _get_algorithms
from neuracore.core.const import DEFAULT_CACHE_DIR, DEFAULT_RECORDING_CACHE_DIR
from neuracore.core.utils.robot_data_spec_utils import (
    convert_robot_data_spec_names_to_ids,
    convert_str_to_robot_data_spec,
    extract_data_types,
    merge_robot_data_spec,
)
from neuracore.core.utils.training_input_args_validation import (
    get_algorithm_name,
    validate_training_params,
)
from neuracore.ml import BatchedTrainingSamples, NeuracoreModel
from neuracore.ml.datasets.pytorch_single_sample_dataset import SingleSampleDataset
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.logging.cloud_log_streamer import CloudLogStreamer
from neuracore.ml.logging.cloud_training_logger import CloudTrainingLogger
from neuracore.ml.logging.json_line_formatter import JsonLineLogFormatter
from neuracore.ml.logging.tensorboard_training_logger import TensorboardTrainingLogger
from neuracore.ml.trainers.batch_autotuner import find_optimal_batch_size
from neuracore.ml.trainers.distributed_trainer import (
    DistributedTrainer,
    cleanup_distributed,
    setup_distributed,
)
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.algorithm_storage_handler import AlgorithmStorageHandler
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler

# Environment setup
os.environ["PJRT_DEVICE"] = "GPU"

# Configure logging
logger = logging.getLogger(__name__)

MAX_AUTOTUNE_SAMPLE_CANDIDATES = 1000


def _resolve_recording_cache_dir(cfg: DictConfig) -> Path:
    """Resolve recording cache directory for synchronized dataset downloads."""
    configured_dir = cfg.get("recording_cache_dir")
    if configured_dir is None:
        return DEFAULT_RECORDING_CACHE_DIR
    return Path(str(configured_dir)).expanduser()


def _resolve_output_dir(
    run_name: str | None = None,
    run_name_auto_increment: bool | str = False,
) -> str:
    """Hydra resolver to generate the output directory path.

    This resolver generates a unique output directory based on run_name.
    It's called during Hydra initialization, so the directory is available before
    the main function runs.

    When run_name_auto_increment is False (default), if a directory for the given
    run name already exists, resolution fails with FileExistsError. Set
    run_name_auto_increment=true to automatically use run_name_1, run_name_2, etc.

    Args:
        run_name: Optional run name. If None or "null", a random name will be generated.
        run_name_auto_increment: If True (or "true"), append _1, _2, ... when the
            name already exists. If False (default), fail when the name exists.

    Returns:
        Full path to the output directory.

    Raises:
        FileExistsError: When run_name_auto_increment is False and a directory
            for run_name (or run_name_N) already exists.
    """
    # Handle None, empty string, or string "null"
    if (
        not run_name
        or run_name == "null"
        or (isinstance(run_name, str) and run_name.strip() == "")
    ):
        # Generate name with hyphens instead of underscores
        run_name = generate_name(style="underscore").replace("_", "-")
    else:
        run_name = _sanitize_run_name(str(run_name))

    auto_increment = (
        str(run_name_auto_increment).lower() == "true"
        if not isinstance(run_name_auto_increment, bool)
        else run_name_auto_increment
    )

    base_dir = DEFAULT_CACHE_DIR / "runs"

    final_run_name = run_name
    if base_dir.exists():
        taken = {
            p.name
            for p in base_dir.iterdir()
            if p.is_dir() and (p.name == run_name or p.name.startswith(f"{run_name}_"))
        }
        if run_name in taken:
            if not auto_increment:
                existing = base_dir / run_name
                raise FileExistsError(
                    f"A run named {run_name!r} already exists at {existing}. "
                    "Either use a different run_name, "
                    "or set run_name_auto_increment=true "
                    "to use an incremented name (e.g. run_name_1)."
                )
            suffix = 1
            while f"{run_name}_{suffix}" in taken:
                suffix += 1
            final_run_name = f"{run_name}_{suffix}"
            logger.warning(
                f"A run named {run_name} already exists. "
                f"Using {final_run_name} for this run."
            )

    # Build full path
    return str(base_dir / final_run_name)


def _sanitize_run_name(name: str) -> str:
    """Sanitize run name for use in file paths.

    Args:
        name: The run name to sanitize.

    Returns:
        A sanitized version of the name safe for file paths.
    """
    # Replace spaces, slashes, and other problematic characters with underscores
    sanitized = re.sub(r"[^\w\-]", "_", name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def _estimate_sample_tensor_bytes(sample: BatchedTrainingSamples) -> int:
    """Roughly estimate total tensor memory footprint (bytes) for a sample."""

    def _collect(obj: Any) -> int:
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, BatchedNCData):
            return _collect(obj.model_dump())
        if isinstance(obj, dict):
            return sum(_collect(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_collect(v) for v in obj)
        return 0

    return _collect({
        "inputs": sample.inputs,
        "inputs_mask": sample.inputs_mask,
        "outputs": sample.outputs,
        "outputs_mask": sample.outputs_mask,
    })


def _select_worst_case_sample(
    dataset: PytorchSynchronizedDataset, device: torch.device
) -> BatchedTrainingSamples:
    """Pick the heaviest sample (by tensor bytes) from a subset of the dataset."""
    if len(dataset) == 0:
        raise ValueError("Cannot autotune batch size with an empty dataset.")

    search_space = range(min(MAX_AUTOTUNE_SAMPLE_CANDIDATES, len(dataset) - 1))
    heaviest_sample: BatchedTrainingSamples | None = None
    heaviest_bytes = -1
    heaviest_idx = -1

    for idx in search_space:

        candidate = dataset[idx]
        candidate_bytes = _estimate_sample_tensor_bytes(candidate)
        if candidate_bytes > heaviest_bytes:
            heaviest_bytes = candidate_bytes
            heaviest_sample = candidate
            heaviest_idx = idx

    assert heaviest_sample is not None
    logger.info(
        "Selected sample %s as worst-case for autotuning (approx %.2f MB of tensors)",
        heaviest_idx,
        heaviest_bytes / (1024**2),
    )

    return heaviest_sample.to(device)


def _serialize_robot_data_spec(
    robot_data_spec: dict[str, dict[DataType, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    """Convert robot data spec to JSON-serializable form."""
    serializable: dict[str, dict[str, list[str]]] = {}
    for robot_id, data_types in robot_data_spec.items():
        serializable[robot_id] = {}
        for data_type, names in data_types.items():
            key = data_type.name if hasattr(data_type, "name") else str(data_type)
            serializable[robot_id][key] = list(names)
    return serializable


def _save_local_training_metadata(
    cfg: DictConfig,
    algorithm_name: str,
    input_robot_data_spec: dict[str, dict[DataType, list[str]]],
    output_robot_data_spec: dict[str, dict[DataType, list[str]]],
) -> None:
    """Persist basic training run metadata locally for local runs."""
    training_id = getattr(cfg, "training_id", None)
    if training_id is not None:
        # Cloud run metadata is stored remotely; skip local save.
        return

    output_dir = Path(getattr(cfg, "local_output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name

    metadata = {
        "id": run_id,
        "name": run_id,
        "status": "RUNNING",
        "algorithm": algorithm_name,
        "algorithm_id": getattr(cfg, "algorithm_id", None),
        "dataset_id": getattr(cfg, "dataset_id", None),
        "dataset_name": getattr(cfg, "dataset_name", None),
        "launch_time": time.time(),
        "local_output_dir": str(output_dir),
        "org_id": getattr(cfg, "org_id", None),
        "input_robot_data_spec": _serialize_robot_data_spec(input_robot_data_spec),
        "output_robot_data_spec": _serialize_robot_data_spec(output_robot_data_spec),
        "frequency": getattr(cfg, "frequency", None),
        "output_prediction_horizon": getattr(cfg, "output_prediction_horizon", None),
        # Align with cloud run schema for inspect output
        "epoch": -1,
        "step": -1,
        "gpu_type": None,
        "num_gpus": None,
        "synchronization_details": {
            "frequency": getattr(cfg, "frequency", None),
            "allow_duplicates": None,
            "max_delay_s": None,
        },
    }

    metadata_path = output_dir / "training_run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def setup_logging(output_dir: str, rank: int = 0) -> None:
    """Setup logging configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        file_handler = logging.FileHandler(output_path / "train.log")
        file_handler.setFormatter(JsonLineLogFormatter())
        handlers: list[logging.Handler] = [file_handler]
        logging.basicConfig(
            level=logging.INFO,
            handlers=handlers,
            force=True,
        )
    else:
        file_handler = logging.FileHandler(output_path / f"train-rank{rank}.log")
        file_handler.setFormatter(JsonLineLogFormatter())
        handlers = [file_handler]
        logging.basicConfig(
            level=logging.INFO,
            handlers=handlers,
            force=True,
        )


def get_model_and_algorithm_config(
    cfg: DictConfig,
    model_init_description: ModelInitDescription,
) -> tuple[NeuracoreModel, dict[str, Any]]:
    """Get model and algorithm configuration."""
    algorithm_config: dict[str, Any] = {}
    if "algorithm" in cfg:
        algorithm_config = OmegaConf.to_container(cfg.algorithm, resolve=True)
        algorithm_config.pop("_target_", None)
        logger.info("Using custom algorithm parameters")
        logger.info(f"Algorithm parameters: {algorithm_config}")

        model = hydra.utils.instantiate(
            cfg.algorithm,
            model_init_description=model_init_description,
            **algorithm_config,
        )
    elif cfg.algorithm_id is not None:
        # Use algorithm_params for custom parameters
        if cfg.algorithm_params is not None:
            algorithm_config = OmegaConf.to_container(
                cfg.algorithm_params, resolve=True
            )
            logger.info("Using custom algorithm parameters")
            logger.info(f"Algorithm parameters: {algorithm_config}")

        extract_dir = Path(cfg.local_output_dir) / "algorithm"
        algorithm_loader = AlgorithmLoader(extract_dir)
        model_class = algorithm_loader.load_model()
        model = model_class(
            model_init_description=model_init_description,
            **algorithm_config,
        )
    else:
        raise ValueError(
            "Either 'algorithm' or 'algorithm_id' "
            "must be provided in the configuration"
        )
    return model, algorithm_config


def determine_optimal_batch_size(
    cfg: DictConfig,
    input_robot_data_spec: dict[str, dict[DataType, list[str]]],
    output_robot_data_spec: dict[str, dict[DataType, list[str]]],
    dataset: SingleSampleDataset,
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

    model_init_description = ModelInitDescription(
        dataset_statistics=dataset.dataset_statistics,
        input_data_types=extract_data_types(input_robot_data_spec),
        output_data_types=extract_data_types(output_robot_data_spec),
        output_prediction_horizon=cfg.output_prediction_horizon,
    )

    model, algorithm_config = get_model_and_algorithm_config(
        cfg, model_init_description
    )

    model = model.to(device)

    max_batch_size = cfg.max_batch_size if "max_batch_size" in cfg else len(dataset)
    min_batch_size = cfg.min_batch_size if "min_batch_size" in cfg else 2
    num_workers = cfg.batch_size_autotuning_num_workers

    logger.info(
        f"using max_batch_size: {max_batch_size}, "
        f"min_batch_size: {min_batch_size}, "
        f"num_workers: {num_workers}"
    )

    # Determine per-GPU batch size
    optimal_batch_size = find_optimal_batch_size(
        dataset=dataset,
        model=model,
        model_kwargs=algorithm_config,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
        },
    )

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(
        f"Autotuning complete. Optimal batch size per GPU: {optimal_batch_size}"
    )

    return optimal_batch_size


def run_training(
    rank: int,
    world_size: int,
    cfg: DictConfig,
    batch_size: int,
    input_robot_data_spec: dict[str, dict[DataType, list[str]]],
    output_robot_data_spec: dict[str, dict[DataType, list[str]]],
    dataset: PytorchSynchronizedDataset,
    device: torch.device | None = None,
) -> None:
    """Run the training process for a single GPU."""
    log_streamer: CloudLogStreamer | None = None

    # Setup for distributed training
    if world_size > 1:
        nc.login()  # Ensure Neuracore is logged in on this process
        setup_distributed(rank, world_size)

    # Setup logging (different file per process)
    setup_logging(cfg.local_output_dir, rank)
    logger = logging.getLogger(__name__)

    # Set random seed (different for each process to ensure different data sampling)
    torch.manual_seed(cfg.seed + rank)

    try:
        logger.info(f"Using batch size: {batch_size}")

        # Merge data_types for synchronization
        merge_robot_data_spec(input_robot_data_spec, output_robot_data_spec)

        # Split dataset
        dataset_size = len(dataset)
        train_split = 1 - cfg.validation_split
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size

        # Use random split with fixed seed for deterministic behavior
        generator = torch.Generator().manual_seed(cfg.seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg.seed,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=cfg.seed,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=cfg.num_train_workers,
                pin_memory=True,
                persistent_workers=cfg.num_train_workers > 0,
                collate_fn=dataset.collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=cfg.num_val_workers,
                pin_memory=True,
                persistent_workers=cfg.num_val_workers > 0,
                collate_fn=dataset.collate_fn,
            )
        else:
            # Regular data loaders for single GPU training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=cfg.num_train_workers,
                pin_memory=True,
                persistent_workers=cfg.num_train_workers > 0,
                collate_fn=dataset.collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cfg.num_val_workers,
                pin_memory=True,
                persistent_workers=cfg.num_val_workers > 0,
                collate_fn=dataset.collate_fn,
            )

        # Log data loader information
        logger.info(
            f"Created data loaders with {len(train_loader.dataset)} training samples "
            f"and {len(val_loader.dataset)} validation samples"
        )

        # Model doesn't need to know about ids or names, just data types
        input_data_types = extract_data_types(input_robot_data_spec)
        output_data_types = extract_data_types(output_robot_data_spec)
        model_init_description = ModelInitDescription(
            dataset_statistics=dataset.dataset_statistics,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=cfg.output_prediction_horizon,
        )

        model, algorithm_config = get_model_and_algorithm_config(
            cfg, model_init_description
        )

        training_storage_handler = TrainingStorageHandler(
            local_dir=cfg.local_output_dir,
            training_job_id=cfg.training_id,
            algorithm_config=algorithm_config,
        )

        # Only start log streamer on rank 0 to avoid multiple processes
        # uploading the same logs
        # and only if this is a cloud run (training_id is not None)
        if rank == 0 and cfg.training_id is not None:
            log_streamer = CloudLogStreamer(
                storage_handler=training_storage_handler,
                output_dir=Path(cfg.local_output_dir),
            )
            log_streamer.start()

        logger.info(
            f"Created model with "
            f"{sum(p.numel() for p in model.parameters()):,} parameters"
        )

        training_logger: TensorboardTrainingLogger | CloudTrainingLogger
        if cfg.training_id is None:
            training_logger = TensorboardTrainingLogger(
                log_dir=Path(cfg.local_output_dir) / "tensorboard",
            )
        else:
            training_logger = CloudTrainingLogger(
                training_id=cfg.training_id,
            )

        trainer = DistributedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_logger=training_logger,
            storage_handler=training_storage_handler,
            output_dir=Path(cfg.local_output_dir),
            num_epochs=cfg.epochs,
            log_freq=cfg.logging_frequency,
            keep_last_n_checkpoints=cfg.keep_last_n_checkpoints,
            clip_grad_norm=algorithm_config.get("clip_grad_norm", None),
            rank=rank,
            world_size=world_size,
            device=device,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if cfg.resume_checkpoint_path is not None:
            try:
                checkpoint = trainer.load_checkpoint(cfg.resume_checkpoint_path)
                start_epoch = checkpoint.get("epoch", 0) + 1
                logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            except Exception:
                logger.error("Failed to load checkpoint.", exc_info=True)

        # Start training
        try:
            logger.info("Starting training...")
            trainer.train(start_epoch=start_epoch)
            logger.info("Training completed successfully!")
        except Exception:
            logger.error("Training failed.", exc_info=True)
            raise

    finally:
        if log_streamer is not None:
            log_streamer.close()
        # Clean up distributed process group
        if world_size > 1:
            cleanup_distributed()

        logger.info(f"Process {rank} completed")


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("resolve_output_dir", _resolve_output_dir)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the training script."""
    # Resolve the configuration
    OmegaConf.resolve(cfg)

    logger.info(f"Training run directory: {cfg.local_output_dir}")

    # Print configuration
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    if "algorithm" in cfg and cfg.algorithm_id is not None:
        raise ValueError(
            "Both 'algorithm' and 'algorithm_id' are provided. "
            "Please specify only one."
        )
    if "algorithm" not in cfg and cfg.algorithm_id is None:
        raise ValueError(
            "Neither 'algorithm' nor 'algorithm_id' is provided. " "Please specify one."
        )

    if cfg.dataset_id is None and cfg.dataset_name is None:
        raise ValueError("Either 'dataset_id' or 'dataset_name' must be provided.")
    if cfg.dataset_id is not None and cfg.dataset_name is not None:
        raise ValueError(
            "Both 'dataset_id' and 'dataset_name' are provided. "
            "Please specify only one."
        )

    # Login and get dataset
    nc.login()
    if cfg.org_id is not None:
        nc.set_organization(cfg.org_id)

    if cfg.dataset_id is not None:
        dataset = nc.get_dataset(id=cfg.dataset_id)
    elif cfg.dataset_name is not None:
        dataset = nc.get_dataset(name=cfg.dataset_name)
    else:
        raise ValueError("Either 'dataset_id' or 'dataset_name' must be provided.")
    dataset.cache_dir = _resolve_recording_cache_dir(cfg)
    dataset.cache_dir.mkdir(parents=True, exist_ok=True)

    # Sort out data specs
    if cfg.input_robot_data_spec is not None:
        if not isinstance(cfg.input_robot_data_spec, DictConfig):
            raise ValueError(
                "'input_robot_data_spec' must be a dictionary "
                "mapping robot names to dictionary of data types to "
                "lists of data names."
            )
        input_robot_data_spec = convert_str_to_robot_data_spec(
            cfg.input_robot_data_spec
        )
    else:
        input_data_types = [DataType(data_type) for data_type in cfg.input_data_types]
        input_robot_data_spec = {}
        for robot_id in dataset.robot_ids:
            robot_full_spec = dataset.get_full_data_spec(robot_id)
            input_robot_data_spec[robot_id] = {
                data_type: list(robot_full_spec.get(data_type.value, []))
                for data_type in input_data_types
            }
    if cfg.output_robot_data_spec is not None:
        if not isinstance(cfg.output_robot_data_spec, DictConfig):
            raise ValueError(
                "'output_robot_data_spec' must either be None or a dictionary "
                "mapping robot names to dictions of data types to lists of data names."
            )
        output_robot_data_spec = convert_str_to_robot_data_spec(
            cfg.output_robot_data_spec
        )
    else:
        output_data_types = [DataType(data_type) for data_type in cfg.output_data_types]
        output_robot_data_spec = {}
        for robot_id in dataset.robot_ids:
            robot_full_spec = dataset.get_full_data_spec(robot_id)
            output_robot_data_spec[robot_id] = {
                data_type: list(robot_full_spec.get(data_type.value, []))
                for data_type in output_data_types
            }

    input_robot_data_spec_with_id = convert_robot_data_spec_names_to_ids(
        input_robot_data_spec
    )
    output_robot_data_spec_with_id = convert_robot_data_spec_names_to_ids(
        output_robot_data_spec
    )

    # =========================================================
    # From here onwards we only deal with robot IDs, not names
    # =========================================================

    batch_size = cfg.batch_size

    algorithms_jsons = _get_algorithms()
    if cfg.algorithm_id is not None:
        algorithm_name = get_algorithm_name(
            algorithm_id=cfg.algorithm_id,
            algorithm_jsons=algorithms_jsons,
        )
    else:
        algorithm_name = cfg.algorithm._target_.rsplit(".", 1)[-1]

    validate_training_params(
        dataset,
        dataset_name=cfg.dataset_name if cfg.dataset_name is not None else "",
        algorithm_name=algorithm_name,
        input_robot_data_spec=input_robot_data_spec_with_id,
        output_robot_data_spec=output_robot_data_spec_with_id,
        algorithm_jsons=algorithms_jsons,
    )

    # Prepare data types for synchronization
    robot_data_spec = merge_robot_data_spec(
        input_robot_data_spec_with_id, output_robot_data_spec_with_id
    )

    # Save local metadata so CLI can inspect local runs without cloud access
    _save_local_training_metadata(
        cfg=cfg,
        algorithm_name=algorithm_name,
        input_robot_data_spec=input_robot_data_spec,
        output_robot_data_spec=output_robot_data_spec,
    )

    synchronized_dataset = dataset.synchronize(
        frequency=cfg.frequency,
        robot_data_spec=robot_data_spec,
        prefetch_videos=True,
        max_prefetch_workers=cfg.max_prefetch_workers,
    )

    # Setup logging for main process
    setup_logging(cfg.local_output_dir)

    # Check if distributed training is enabled and multiple GPUs are available
    world_size = torch.cuda.device_count()

    if cfg.algorithm_id is not None:
        # Download the algorithm so that it can be processed later
        logger.info(f"Downloading algorithm from cloud with ID: {cfg.algorithm_id}")
        storage_handler = AlgorithmStorageHandler(algorithm_id=cfg.algorithm_id)
        extract_dir = Path(cfg.local_output_dir) / "algorithm"
        storage_handler.download_algorithm(extract_dir=extract_dir)
        logger.info(f"Algorithm extracted to {extract_dir}")

    device = None
    if cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        device = get_default_device()

    # Create a pytorch synchronized dataset
    # NOTE: we are creating it here, and not in training to access the first sample
    # for batch size autotuning, if used.
    pytorch_dataset = PytorchSynchronizedDataset(
        synchronized_dataset=synchronized_dataset,
        input_robot_data_spec=input_robot_data_spec_with_id,
        output_robot_data_spec=output_robot_data_spec_with_id,
        output_prediction_horizon=cfg.output_prediction_horizon,
    )

    # Handle batch size configuration
    if isinstance(batch_size, str) and batch_size.lower() == "auto":
        sample = _select_worst_case_sample(
            dataset=pytorch_dataset,
            device=device,
        )
        single_sample_dataset = SingleSampleDataset(
            sample=sample,
            input_robot_data_spec=input_robot_data_spec_with_id,
            output_robot_data_spec=output_robot_data_spec_with_id,
            output_prediction_horizon=cfg.output_prediction_horizon,
            dataset_statistics=pytorch_dataset.dataset_statistics,
            num_recordings=len(pytorch_dataset),
        )

        optimal_batch_size = determine_optimal_batch_size(
            cfg=cfg,
            input_robot_data_spec=input_robot_data_spec_with_id,
            output_robot_data_spec=output_robot_data_spec_with_id,
            dataset=single_sample_dataset,
            device=device,
        )
        batch_size = optimal_batch_size

    else:
        batch_size = int(batch_size)

    if world_size > 1:
        # Use multiprocessing to launch multiple processes
        mp.spawn(
            run_training,
            args=(
                world_size,
                cfg,
                batch_size,
                input_robot_data_spec_with_id,
                output_robot_data_spec_with_id,
                pytorch_dataset,
                device,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU or CPU training
        run_training(
            0,
            1,
            cfg,
            batch_size,
            input_robot_data_spec_with_id,
            output_robot_data_spec_with_id,
            pytorch_dataset,
            device,
        )


if __name__ == "__main__":
    main()
