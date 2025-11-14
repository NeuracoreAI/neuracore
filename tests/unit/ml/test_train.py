"""Tests for train.py training script.

This module provides comprehensive testing for the training script functionality
including logging setup, model configuration, data type conversion, batch size
autotuning, and training execution.
"""

import gc
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from neuracore_types import (
    DataItemStats,
    DatasetDescription,
    DataType,
    ModelInitDescription,
)
from omegaconf import OmegaConf

from neuracore.ml import NeuracoreModel
from neuracore.ml.datasets.pytorch_single_sample_dataset import SingleSampleDataset
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.train import (
    convert_data_types,
    determine_optimal_batch_size,
    get_model_and_algorithm_config,
    main,
    setup_logging,
)


class MainTestSetup:
    """Helper class to consolidate common mock setup for main() tests."""

    def __init__(self, monkeypatch, cuda_device_count=1):
        self.monkeypatch = monkeypatch
        self.cuda_device_count = cuda_device_count

        self.mock_dataset = Mock()
        self.mock_synchronized_dataset = Mock()
        self.mock_pytorch_dataset = Mock(spec=PytorchSynchronizedDataset)
        self.mock_pytorch_dataset.dataset_description = Mock()
        self.mock_pytorch_dataset.__len__ = Mock(return_value=100)
        self.mock_pytorch_dataset.load_sample = Mock(return_value=Mock())

        self.mock_login = Mock()
        self.mock_set_organization = Mock()
        self.mock_get_dataset = Mock(return_value=self.mock_dataset)
        self.mock_dataset.synchronize = Mock(
            return_value=self.mock_synchronized_dataset
        )
        self.mock_pytorch_dataset_class = Mock(return_value=self.mock_pytorch_dataset)
        self.mock_run_training = Mock()
        self.mock_cuda_device_count = Mock(return_value=self.cuda_device_count)
        self.mock_storage_handler = Mock()
        self.mock_storage_handler.download_algorithm = Mock()
        self.mock_storage_handler_class = Mock(return_value=self.mock_storage_handler)

    def setup_mocks(
        self,
        include_set_organization=False,
        include_get_default_device=False,
        include_determine_optimal_batch_size=False,
        include_mp_spawn=False,
    ):
        """Apply all monkeypatch.setattr calls for common mocks."""
        self.monkeypatch.setattr("neuracore.ml.train.logger.info", Mock())
        self.monkeypatch.setattr("neuracore.login", self.mock_login)
        self.monkeypatch.setattr("neuracore.get_dataset", self.mock_get_dataset)
        self.monkeypatch.setattr(
            "neuracore.ml.train.PytorchSynchronizedDataset",
            self.mock_pytorch_dataset_class,
        )
        self.monkeypatch.setattr(
            "neuracore.ml.train.run_training", self.mock_run_training
        )
        self.monkeypatch.setattr("torch.cuda.device_count", self.mock_cuda_device_count)
        self.monkeypatch.setattr(
            "neuracore.ml.train.AlgorithmStorageHandler",
            self.mock_storage_handler_class,
        )

        if include_set_organization:
            self.monkeypatch.setattr(
                "neuracore.set_organization", self.mock_set_organization
            )

        if include_get_default_device:
            self.mock_get_default_device = Mock(return_value=torch.device("cuda:0"))
            self.monkeypatch.setattr(
                "neuracore.ml.train.get_default_device", self.mock_get_default_device
            )

        if include_determine_optimal_batch_size:
            self.mock_determine_optimal_batch_size = Mock()
            self.monkeypatch.setattr(
                "neuracore.ml.train.determine_optimal_batch_size",
                self.mock_determine_optimal_batch_size,
            )

        if include_mp_spawn:
            self.mock_mp_spawn = Mock()
            self.monkeypatch.setattr("torch.multiprocessing.spawn", self.mock_mp_spawn)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_description():
    """Create a sample dataset description for testing."""
    return DatasetDescription(
        joint_positions=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_velocities=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_target_positions=DataItemStats(mean=[0.0] * 7, std=[1.0] * 7, max_len=7),
    )


@pytest.fixture
def model_init_description(sample_dataset_description):
    """Create a model initialization description for testing."""
    return ModelInitDescription(
        dataset_description=sample_dataset_description,
        input_data_types=[DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES],
        output_data_types=[DataType.JOINT_TARGET_POSITIONS],
        output_prediction_horizon=5,
    )


@pytest.fixture
def mock_model_class():
    """Create a mock NeuracoreModel class for testing."""

    class MockModel(NeuracoreModel):
        def __init__(self, model_init_description, **kwargs):
            super().__init__(model_init_description)
            self.kwargs = kwargs
            # Add a dummy parameter so optimizer can be created
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            from neuracore_types import ModelPrediction

            return ModelPrediction(
                outputs={DataType.JOINT_TARGET_POSITIONS: torch.zeros(1, 5, 7)}
            )

        def training_step(self, batch):
            from neuracore.ml import BatchedTrainingOutputs

            return BatchedTrainingOutputs(
                output_predictions=torch.zeros(1, 5, 7),
                losses={"loss": torch.tensor(0.5)},
                metrics={},
            )

        def configure_optimizers(self):
            return [torch.optim.Adam(self.parameters())]

        @staticmethod
        def get_supported_input_data_types():
            return [DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES]

        @staticmethod
        def get_supported_output_data_types():
            return [DataType.JOINT_TARGET_POSITIONS]

        def tokenize_text(self, texts):
            return torch.zeros(1, 10), torch.ones(1, 10)

    return MockModel


@pytest.fixture
def mock_single_sample_dataset(model_init_description):
    """Create a mock single sample dataset."""
    sample = Mock()
    sample.inputs = Mock()
    sample.outputs = Mock()

    dataset = Mock(spec=SingleSampleDataset)
    dataset.dataset_description = model_init_description.dataset_description
    dataset.load_sample.return_value = sample
    return dataset


@pytest.fixture
def mock_cfg_batch_size(temp_output_dir):
    """Create a mock configuration for batch size autotuning."""
    return OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": str(temp_output_dir),
        "input_data_types": ["joint_positions", "joint_velocities"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "max_batch_size": 32,
        "min_batch_size": 2,
        "batch_size_autotuning_num_workers": 0,
    })


@pytest.fixture
def mock_cfg_training(temp_output_dir):
    """Create a mock configuration for training."""
    return OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": str(temp_output_dir),
        "seed": 42,
        "validation_split": 0.2,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "num_train_workers": 0,
        "num_val_workers": 0,
        "epochs": 1,
        "logging_frequency": 10,
        "keep_last_n_checkpoints": 3,
        "training_id": None,
        "resume_checkpoint_path": None,
    })


@pytest.fixture
def mock_dataset(sample_dataset_description):
    """Create a mock synchronized dataset."""
    dataset = Mock(spec=PytorchSynchronizedDataset)
    dataset.dataset_description = sample_dataset_description
    dataset.collate_fn = lambda x: x
    dataset.__len__ = Mock(return_value=100)
    dataset.tokenize_text = None
    return dataset


@pytest.mark.parametrize("rank,should_create_log_file", [(0, True), (1, False)])
def test_setup_logging(temp_output_dir, rank, should_create_log_file):
    setup_logging(str(temp_output_dir), rank=rank)

    log_file = temp_output_dir / "train.log"
    if should_create_log_file:
        assert log_file.exists()
    else:
        assert not log_file.exists()

    logger = logging.getLogger(__name__)
    assert logger.level <= logging.INFO


def test_setup_logging_creates_directory(temp_output_dir):
    new_dir = temp_output_dir / "new_dir"
    setup_logging(str(new_dir), rank=0)

    assert new_dir.exists()
    assert (new_dir / "train.log").exists()


@pytest.mark.parametrize(
    "data_types,expected_result,should_raise",
    [
        (
            ["joint_positions", "rgb_image", "joint_velocities"],
            [DataType.JOINT_POSITIONS, DataType.RGB_IMAGE, DataType.JOINT_VELOCITIES],
            False,
        ),
        ([], [], False),
        (["INVALID_DATA_TYPE"], None, True),
    ],
)
def test_convert_data_types(data_types, expected_result, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            convert_data_types(data_types)
    else:
        result = convert_data_types(data_types)
        assert result == expected_result


def test_get_model_with_algorithm_config(
    model_init_description, mock_model_class, monkeypatch
):
    cfg = OmegaConf.create({
        "algorithm": {
            "_target_": "tests.unit.ml.test_train.mock_model_class",
        },
    })

    mock_instantiate = Mock(return_value=mock_model_class(model_init_description))
    monkeypatch.setattr("neuracore.ml.train.hydra.utils.instantiate", mock_instantiate)

    model, algorithm_config = get_model_and_algorithm_config(
        cfg, model_init_description
    )

    assert isinstance(model, NeuracoreModel)
    assert isinstance(algorithm_config, dict)
    mock_instantiate.assert_called_once()


def test_get_model_with_algorithm_id(
    model_init_description, mock_model_class, temp_output_dir, monkeypatch
):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": str(temp_output_dir),
        "algorithm_params": {"param1": "value1"},
    })

    mock_loader = Mock()
    mock_loader.load_model.return_value = mock_model_class
    mock_loader_class = Mock(return_value=mock_loader)
    monkeypatch.setattr("neuracore.ml.train.AlgorithmLoader", mock_loader_class)

    model, algorithm_config = get_model_and_algorithm_config(
        cfg, model_init_description
    )

    assert isinstance(model, NeuracoreModel)
    assert algorithm_config == {"param1": "value1"}
    mock_loader.load_model.assert_called_once()


def test_get_model_with_algorithm_id_no_params(
    model_init_description, mock_model_class, temp_output_dir, monkeypatch
):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": str(temp_output_dir),
        "algorithm_params": None,
    })

    mock_loader = Mock()
    mock_loader.load_model.return_value = mock_model_class
    mock_loader_class = Mock(return_value=mock_loader)
    monkeypatch.setattr("neuracore.ml.train.AlgorithmLoader", mock_loader_class)

    model, algorithm_config = get_model_and_algorithm_config(
        cfg, model_init_description
    )

    assert isinstance(model, NeuracoreModel)
    assert algorithm_config == {}


def test_get_model_no_algorithm_or_id(model_init_description):
    cfg = OmegaConf.create({
        "algorithm_id": None,
    })

    with pytest.raises(ValueError, match="Either 'algorithm' or 'algorithm_id'"):
        get_model_and_algorithm_config(cfg, model_init_description)


def test_get_model_both_algorithm_and_id(
    model_init_description, mock_model_class, monkeypatch
):
    cfg = OmegaConf.create({
        "algorithm": {"_target_": "tests.unit.ml.test_train.mock_model_class"},
        "algorithm_id": "test-id",
    })

    mock_instantiate = Mock(return_value=mock_model_class(model_init_description))
    monkeypatch.setattr("neuracore.ml.train.hydra.utils.instantiate", mock_instantiate)

    model, algorithm_config = get_model_and_algorithm_config(
        cfg, model_init_description
    )

    assert isinstance(model, NeuracoreModel)
    mock_instantiate.assert_called_once()


def test_determine_optimal_batch_size_gpu(
    mock_cfg_batch_size,
    mock_single_sample_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_get_device = Mock(return_value=torch.device("cuda:0"))
    mock_find_optimal = Mock(return_value=16)
    mock_get_model_config = Mock(
        return_value=(mock_model_class(model_init_description), {})
    )

    monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
    monkeypatch.setattr("neuracore.ml.train.find_optimal_batch_size", mock_find_optimal)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    result = determine_optimal_batch_size(
        mock_cfg_batch_size, mock_single_sample_dataset
    )

    assert result == 16
    mock_find_optimal.assert_called_once()


def test_determine_optimal_batch_size_no_gpu(
    mock_cfg_batch_size,
    mock_single_sample_dataset,
    monkeypatch,
):
    mock_get_device = Mock(return_value=torch.device("cpu"))
    monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(ValueError, match="Autotuning is only supported on GPUs"):
        determine_optimal_batch_size(mock_cfg_batch_size, mock_single_sample_dataset)


def test_determine_optimal_batch_size_with_device(
    mock_cfg_batch_size,
    mock_single_sample_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_find_optimal = Mock(return_value=8)
    mock_get_model_config = Mock(
        return_value=(mock_model_class(model_init_description), {})
    )

    monkeypatch.setattr("neuracore.ml.train.find_optimal_batch_size", mock_find_optimal)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    device = torch.device("cuda:0")
    result = determine_optimal_batch_size(
        mock_cfg_batch_size, mock_single_sample_dataset, device=device
    )

    assert result == 8
    mock_find_optimal.assert_called_once()


def test_determine_optimal_batch_size_defaults(
    mock_single_sample_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": "/tmp/test",
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "batch_size_autotuning_num_workers": 0,
    })
    mock_get_device = Mock(return_value=torch.device("cuda:0"))
    mock_find_optimal = Mock(return_value=4)
    mock_get_model_config = Mock(
        return_value=(mock_model_class(model_init_description), {})
    )

    monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
    monkeypatch.setattr("neuracore.ml.train.find_optimal_batch_size", mock_find_optimal)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    mock_single_sample_dataset.__len__ = Mock(return_value=100)

    result = determine_optimal_batch_size(cfg, mock_single_sample_dataset)

    assert result == 4
    call_kwargs = mock_find_optimal.call_args[1]
    assert call_kwargs["min_batch_size"] == 2
    assert call_kwargs["max_batch_size"] == 100


def test_run_training_single_gpu(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.load_checkpoint.return_value = {"epoch": 0}
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_trainer_class.assert_called_once()
    mock_trainer.train.assert_called_once_with(start_epoch=0)
    mock_setup.assert_not_called()
    mock_cleanup.assert_not_called()


def test_run_training_with_training_id(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_cfg_training.training_id = "test-training-id"
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_cloud_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr("neuracore.ml.train.CloudTrainingLogger", mock_cloud_logger)
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_cloud_logger.assert_called_once_with(training_id="test-training-id")
    mock_trainer.train.assert_called_once()


def test_run_training_with_checkpoint_resume(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_cfg_training.resume_checkpoint_path = "/path/to/checkpoint.pth"
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.load_checkpoint.return_value = {"epoch": 5}
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_trainer.load_checkpoint.assert_called_once_with("/path/to/checkpoint.pth")
    mock_trainer.train.assert_called_once_with(start_epoch=6)


def test_run_training_distributed(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    world_size = 2
    rank = 1
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_setup.assert_called_once_with(rank, world_size)
    mock_cleanup.assert_called_once()
    mock_login.assert_called_once()


def test_run_training_handles_exception(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.train.side_effect = RuntimeError("Training failed")
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    with pytest.raises(RuntimeError, match="Training failed"):
        run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_cleanup.assert_not_called()


def test_run_training_checkpoint_load_failure(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_cfg_training.resume_checkpoint_path = "/path/to/checkpoint.pth"
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.load_checkpoint.side_effect = FileNotFoundError("Checkpoint not found")
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_trainer.load_checkpoint.assert_called_once_with("/path/to/checkpoint.pth")
    mock_trainer.train.assert_called_once_with(start_epoch=0)


def test_determine_optimal_batch_size_cleanup(
    mock_cfg_batch_size,
    mock_single_sample_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_get_device = Mock(return_value=torch.device("cuda:0"))
    mock_find_optimal = Mock(return_value=16)
    mock_get_model_config = Mock(
        return_value=(mock_model_class(model_init_description), {})
    )

    cleanup_called = {"gc_collect": False, "cuda_empty_cache": False}

    original_gc_collect = gc.collect
    original_cuda_empty_cache = torch.cuda.empty_cache

    def mock_gc_collect():
        cleanup_called["gc_collect"] = True
        return original_gc_collect()

    def mock_cuda_empty_cache():
        cleanup_called["cuda_empty_cache"] = True
        return original_cuda_empty_cache()

    monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
    monkeypatch.setattr("neuracore.ml.train.find_optimal_batch_size", mock_find_optimal)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("gc.collect", mock_gc_collect)
    monkeypatch.setattr("torch.cuda.empty_cache", mock_cuda_empty_cache)

    result = determine_optimal_batch_size(
        mock_cfg_batch_size, mock_single_sample_dataset
    )

    assert result == 16
    assert cleanup_called["gc_collect"]
    assert cleanup_called["cuda_empty_cache"]


def test_get_model_with_algorithm_config_removes_target(
    model_init_description, mock_model_class, monkeypatch
):
    cfg = OmegaConf.create({
        "algorithm": {
            "_target_": "tests.unit.ml.test_train.mock_model_class",
            "param1": "value1",
            "param2": "value2",
        },
    })

    mock_instantiate = Mock(return_value=mock_model_class(model_init_description))
    monkeypatch.setattr("neuracore.ml.train.hydra.utils.instantiate", mock_instantiate)

    model, algorithm_config = get_model_and_algorithm_config(
        cfg, model_init_description
    )

    assert isinstance(model, NeuracoreModel)
    assert "_target_" not in algorithm_config
    assert algorithm_config == {"param1": "value1", "param2": "value2"}
    mock_instantiate.assert_called_once()


def test_determine_optimal_batch_size_device_none_branch(
    mock_cfg_batch_size,
    mock_single_sample_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_get_device = Mock(return_value=torch.device("cuda:0"))
    mock_find_optimal = Mock(return_value=16)
    mock_get_model_config = Mock(
        return_value=(mock_model_class(model_init_description), {})
    )

    monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
    monkeypatch.setattr("neuracore.ml.train.find_optimal_batch_size", mock_find_optimal)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    result = determine_optimal_batch_size(
        mock_cfg_batch_size, mock_single_sample_dataset
    )

    assert result == 16
    mock_get_device.assert_called_once()
    mock_find_optimal.assert_called_once()


def test_determine_optimal_batch_size_cpu_device(
    mock_cfg_batch_size,
    mock_single_sample_dataset,
    monkeypatch,
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="Autotuning is only supported on GPUs"):
        determine_optimal_batch_size(
            mock_cfg_batch_size, mock_single_sample_dataset, device=device
        )


def test_run_training_checkpoint_no_epoch_key(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_cfg_training.resume_checkpoint_path = "/path/to/checkpoint.pth"
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.load_checkpoint.return_value = {}  # No epoch key
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_trainer.train.assert_called_once_with(start_epoch=1)


def test_run_training_with_workers(
    mock_cfg_training,
    mock_dataset,
    model_init_description,
    mock_model_class,
    monkeypatch,
):
    mock_cfg_training.num_train_workers = 4
    mock_cfg_training.num_val_workers = 2
    world_size = 1
    rank = 0
    batch_size = 8

    mock_model = mock_model_class(model_init_description)
    mock_get_model_config = Mock(return_value=(mock_model, {}))
    mock_setup = Mock()
    mock_cleanup = Mock()
    mock_trainer = Mock()
    mock_trainer.load_checkpoint.return_value = {"epoch": 0}
    mock_trainer.train = Mock()
    mock_trainer_class = Mock(return_value=mock_trainer)
    mock_storage_handler = Mock()
    mock_tensorboard_logger = Mock()
    mock_login = Mock()

    monkeypatch.setattr("neuracore.ml.train.setup_distributed", mock_setup)
    monkeypatch.setattr("neuracore.ml.train.cleanup_distributed", mock_cleanup)
    monkeypatch.setattr("neuracore.ml.train.DistributedTrainer", mock_trainer_class)
    monkeypatch.setattr(
        "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TrainingStorageHandler", mock_storage_handler
    )
    monkeypatch.setattr(
        "neuracore.ml.train.TensorboardTrainingLogger", mock_tensorboard_logger
    )
    monkeypatch.setattr("neuracore.login", mock_login)

    from neuracore.ml.train import run_training

    run_training(rank, world_size, mock_cfg_training, batch_size, mock_dataset)

    mock_trainer_class.assert_called_once()
    call_kwargs = mock_trainer_class.call_args[1]
    assert call_kwargs["train_loader"].dataset is not None
    mock_trainer.train.assert_called_once()


@pytest.mark.parametrize(
    "cfg_updates,expected_error_match",
    [
        (
            {"algorithm": {"_target_": "test"}, "algorithm_id": "test-id"},
            "Both 'algorithm' and 'algorithm_id' are provided",
        ),
        (
            {"algorithm_id": None},
            "Neither 'algorithm' nor 'algorithm_id' is provided",
        ),
        (
            {
                "algorithm_id": "test-algorithm-id",
                "dataset_id": None,
                "dataset_name": None,
            },
            "Either 'dataset_id' or 'dataset_name' must be provided",
        ),
        (
            {
                "algorithm_id": "test-algorithm-id",
                "dataset_id": "test-dataset-id",
                "dataset_name": "test-dataset-name",
            },
            "Both 'dataset_id' and 'dataset_name' are provided",
        ),
    ],
)
def test_main_validation_errors(monkeypatch, cfg_updates, expected_error_match):
    base_cfg = {
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "local_output_dir": "/tmp/test",
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
    }
    base_cfg.update(cfg_updates)
    cfg = OmegaConf.create(base_cfg)

    monkeypatch.setattr("neuracore.ml.train.logger.info", Mock())

    with pytest.raises(ValueError, match=expected_error_match):
        main(cfg)


def test_main_with_dataset_name(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": None,
        "dataset_name": "test-dataset-name",
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks(include_set_organization=True)

    main(cfg)

    setup.mock_get_dataset.assert_called_once_with(name="test-dataset-name")
    setup.mock_set_organization.assert_not_called()


def test_main_with_org_id(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": "test-org-id",
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks(include_set_organization=True)

    main(cfg)

    setup.mock_set_organization.assert_called_once_with("test-org-id")
    setup.mock_get_dataset.assert_called_once_with(id="test-dataset-id")


def test_main_with_algorithm_instead_of_algorithm_id(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm": {
            "_target_": "tests.unit.ml.test_train.mock_model_class",
        },
        "algorithm_id": None,
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks()

    main(cfg)

    setup.mock_storage_handler_class.assert_not_called()


def test_main_with_device_none(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks(include_get_default_device=True)

    main(cfg)

    setup.mock_get_default_device.assert_called_once()
    assert setup.mock_run_training.call_args[0][5] == torch.device("cuda:0")


def test_main_with_batch_size_not_auto(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 16,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks(include_determine_optimal_batch_size=True)

    main(cfg)

    setup.mock_determine_optimal_batch_size.assert_not_called()
    assert setup.mock_run_training.call_args[0][3] == 16


def test_main_algorithm_in_cfg_but_algorithm_id_none(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm": {
            "_target_": "tests.unit.ml.test_train.mock_model_class",
        },
        "algorithm_id": None,
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks()

    main(cfg)

    setup.mock_get_dataset.assert_called_once()


def test_main_algorithm_not_in_cfg_but_algorithm_id_not_none(
    monkeypatch, temp_output_dir
):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks()

    main(cfg)

    setup.mock_get_dataset.assert_called_once()


def test_main_dataset_id_none_but_dataset_name_not_none(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": None,
        "dataset_name": "test-dataset-name",
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks()

    main(cfg)

    setup.mock_get_dataset.assert_called_once_with(name="test-dataset-name")


def test_main_dataset_id_not_none_but_dataset_name_none(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks()

    main(cfg)

    setup.mock_get_dataset.assert_called_once_with(id="test-dataset-id")


def test_main_batch_size_string_but_not_auto(monkeypatch, temp_output_dir):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": "16",
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch)
    setup.setup_mocks(include_determine_optimal_batch_size=True)

    main(cfg)

    setup.mock_determine_optimal_batch_size.assert_not_called()
    assert setup.mock_run_training.call_args[0][3] == 16


@pytest.mark.parametrize(
    "world_size,should_use_mp_spawn",
    [
        (1, False),
        (2, True),
    ],
)
def test_main_world_size(monkeypatch, temp_output_dir, world_size, should_use_mp_spawn):
    cfg = OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "dataset_id": "test-dataset-id",
        "dataset_name": None,
        "org_id": None,
        "device": None,
        "local_output_dir": str(temp_output_dir),
        "batch_size": 8,
        "input_data_types": ["joint_positions"],
        "output_data_types": ["joint_target_positions"],
        "output_prediction_horizon": 5,
        "frequency": 30,
        "algorithm_params": None,
    })

    setup = MainTestSetup(monkeypatch, cuda_device_count=world_size)
    setup.setup_mocks(include_mp_spawn=True)

    main(cfg)

    if should_use_mp_spawn:
        setup.mock_mp_spawn.assert_called_once()
        call_args = setup.mock_mp_spawn.call_args
        assert len(call_args[0]) >= 1
        assert call_args[1]["nprocs"] == world_size
        assert call_args[1]["join"] is True
        args_tuple = call_args[1]["args"]
        assert args_tuple[0] == world_size
        assert args_tuple[1] == cfg
        assert args_tuple[2] == 8
        assert args_tuple[3] == setup.mock_pytorch_dataset
        setup.mock_run_training.assert_not_called()
    else:
        setup.mock_mp_spawn.assert_not_called()
        setup.mock_run_training.assert_called_once()
        assert setup.mock_run_training.call_args[0][0] == 0
        assert setup.mock_run_training.call_args[0][1] == 1
