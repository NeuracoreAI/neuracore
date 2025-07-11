"""Hydra-based training script for Neuracore models."""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, random_split

import neuracore as nc
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.nc_types import DataType, ModelInitDescription
from neuracore.ml import NeuracoreModel
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.logging.cloud_training_logger import CloudTrainingLogger
from neuracore.ml.logging.tensorboard_training_logger import TensorboardTrainingLogger
from neuracore.ml.trainers.batch_autotuner import find_optimal_batch_size
from neuracore.ml.trainers.distributed_trainer import (
    DistributedTrainer,
    cleanup_distributed,
    setup_distributed,
)
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.algorithm_storage_handler import AlgorithmStorageHandler
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler

# Environment setup
os.environ["PJRT_DEVICE"] = "GPU"

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, rank: int = 0) -> None:
    """Setup logging configuration."""
    if rank == 0:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(output_path / "train.log"),
            ],
        )
    else:
        # For other ranks, only log to console
        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )


def get_algorithm_config_and_class(
    cfg: DictConfig,
) -> Tuple[type[NeuracoreModel], Dict[str, Any]]:
    """Get model class and algorithm configuration."""
    assert (
        cfg.algorithm_id is not None
    ), "Algorithm ID must be provided in the configuration"
    #  Assume algorithm already downloaded
    extract_dir = Path(cfg.local_output_dir) / "algorithm"
    algorithm_loader = AlgorithmLoader(extract_dir)
    model_class = algorithm_loader.load_model()

    # Use algorithm_params for custom parameters
    algorithm_config = {}
    if cfg.algorithm_params is not None:
        algorithm_config = OmegaConf.to_container(cfg.algorithm_params, resolve=True)
    logger.info("Using custom algorithm parameters")
    logger.info(f"Algorithm parameters: {algorithm_config}")
    return model_class, algorithm_config


def convert_data_types(data_types_list: list[str]) -> list[DataType]:
    """Convert string data types to DataType enums."""
    return [DataType(dt) for dt in data_types_list]


def determine_optimal_batch_size(
    cfg: DictConfig,
    synchronized_dataset: SynchronizedDataset,
) -> int:
    """Run batch size autotuning on a single GPU and return the result."""
    logger.info("Starting batch size autotuning on GPU 0...")

    input_data_types = convert_data_types(cfg.input_data_types)
    output_data_types = convert_data_types(cfg.output_data_types)

    # Setup dataset for autotuning
    dataset = PytorchSynchronizedDataset(
        synchronized_dataset=synchronized_dataset,
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        output_prediction_horizon=cfg.output_prediction_horizon,
    )

    # Create a smaller subset for autotuning
    train_size = len(dataset)
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))

    model_init_description = ModelInitDescription(
        dataset_description=dataset.dataset_description,
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        output_prediction_horizon=cfg.output_prediction_horizon,
    )

    if cfg.algorithm is not None:
        model = hydra.utils.instantiate(
            cfg.algorithm,
            model_init_description=model_init_description,
        )
    elif cfg.algorithm_id is not None:
        model_class, algorithm_config = get_algorithm_config_and_class(cfg)
        model = model_class(
            model_init_description=model_init_description,
            **algorithm_config,
        )
    else:
        raise ValueError(
            "Either 'algorithm' or 'algorithm_id' must be provided in the configuration"
        )

    # Determine per-GPU batch size
    optimal_batch_size = find_optimal_batch_size(
        dataset=train_dataset,
        model=model,
        model_kwargs=algorithm_config,
        min_batch_size=2,
        max_batch_size=4096,
        gpu_id=0,
        dataloader_kwargs={
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
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
    synchronized_dataset: SynchronizedDataset,
) -> None:
    """Run the training process for a single GPU."""
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

        input_data_types = convert_data_types(cfg.input_data_types)
        output_data_types = convert_data_types(cfg.output_data_types)

        # Setup dataset
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=synchronized_dataset,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=cfg.output_prediction_horizon,
        )

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

        model_init_description = ModelInitDescription(
            dataset_description=dataset.dataset_description,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=cfg.output_prediction_horizon,
        )

        algorithm_config: dict[str, Any] = {}
        if "algorithm" in cfg:
            model = hydra.utils.instantiate(
                cfg.algorithm,
                model_init_description=model_init_description,
            )
        elif cfg.algorithm_id is not None:
            model_class, algorithm_config = get_algorithm_config_and_class(cfg)
            model = model_class(
                model_init_description=model_init_description,
                **algorithm_config,
            )
        else:
            raise ValueError(
                "Either 'algorithm' or 'algorithm_id' "
                "must be provided in the configuration"
            )

        training_storage_handler = TrainingStorageHandler(
            local_dir=cfg.local_output_dir,
            training_job_id=cfg.training_id,
            algorithm_config=algorithm_config,
        )

        # TODO: Find a better way to handle text tokenization
        dataset.tokenize_text = model.tokenize_text

        logger.info(
            f"Created model with "
            f"{sum(p.numel() for p in model.parameters()):,} parameters"
        )

        training_logger: Union[TensorboardTrainingLogger, CloudTrainingLogger]
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
            rank=rank,
            world_size=world_size,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if cfg.resume:
            try:
                checkpoint = trainer.load_checkpoint(cfg.resume)
                start_epoch = checkpoint.get("epoch", 0) + 1
                logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")

        # Start training
        try:
            logger.info("Starting training...")
            trainer.train(start_epoch=start_epoch)
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    finally:
        # Clean up distributed process group
        if world_size > 1:
            cleanup_distributed()

        logger.info(f"Process {rank} completed")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the training script."""
    # Print configuration
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

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

    batch_size = cfg.batch_size

    # Prepare data types for synchronization
    data_types_to_sync = convert_data_types(
        cfg.input_data_types + cfg.output_data_types
    )

    # Login and get dataset
    nc.login()
    if cfg.org_id is not None:
        nc.set_organization(cfg.org_id)
    if cfg.dataset_id is not None:
        dataset = nc.get_dataset(id=cfg.dataset_id)
    elif cfg.dataset_name is not None:
        dataset = nc.get_dataset(name=cfg.dataset_name)
    synchronized_dataset = dataset.synchronize(
        frequency=cfg.frequency, data_types=data_types_to_sync
    )

    # Setup logging for main process
    setup_logging(cfg.local_output_dir)

    # Check if distributed training is enabled and multiple GPUs are available
    world_size = torch.cuda.device_count()

    # Handle batch size configuration
    if isinstance(batch_size, str) and batch_size.lower() == "auto":
        optimal_batch_size = determine_optimal_batch_size(cfg, synchronized_dataset)
        batch_size = optimal_batch_size
    else:
        batch_size = int(batch_size)

    if cfg.algorithm_id is not None:
        # Download the algorithm so that it can be processed later
        logger.info(f"Downloading algorithm from cloud with ID: {cfg.algorithm_id}")
        storage_handler = AlgorithmStorageHandler(algorithm_id=cfg.algorithm_id)
        extract_dir = Path(cfg.local_output_dir) / "algorithm"
        storage_handler.download_algorithm(extract_dir=extract_dir)
        logger.info(f"Algorithm extracted to {extract_dir}")

    if world_size > 1:
        # Use multiprocessing to launch multiple processes
        mp.spawn(
            run_training,
            args=(world_size, cfg, batch_size, synchronized_dataset),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU or CPU training
        run_training(0, 1, cfg, batch_size, synchronized_dataset)


if __name__ == "__main__":
    main()
