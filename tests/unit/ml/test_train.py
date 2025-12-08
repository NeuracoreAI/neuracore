"""Tests for train.py training script.

This module provides comprehensive testing for the training script functionality
including logging setup, model configuration, data type conversion, batch size
autotuning, and training execution.
"""

import gc
import logging
import os
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
    ModelPrediction,
)
from omegaconf import OmegaConf

from neuracore.ml import BatchedTrainingOutputs, NeuracoreModel
from neuracore.ml.datasets.pytorch_single_sample_dataset import SingleSampleDataset
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.train import (
    convert_data_types,
    determine_optimal_batch_size,
    get_model_and_algorithm_config,
    main,
    run_training,
    setup_logging,
)

SKIP_TEST = os.environ.get("CI", "false").lower() == "true"


class MainTestSetup:
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


class RunTrainingTestSetup:
    def __init__(
        self,
        monkeypatch,
        model_init_description,
        mock_model_class,
        world_size=1,
        rank=0,
        batch_size=8,
        checkpoint_epoch=None,
        use_cloud_logger=False,
    ):
        self.monkeypatch = monkeypatch
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size

        # Create mock model and config
        self.mock_model = mock_model_class(model_init_description)
        self.mock_get_model_config = Mock(return_value=(self.mock_model, {}))

        # Create distributed training mocks
        self.mock_setup = Mock()
        self.mock_cleanup = Mock()

        # Create trainer mocks
        self.mock_trainer = Mock()
        if checkpoint_epoch is not None:
            self.mock_trainer.load_checkpoint.return_value = {"epoch": checkpoint_epoch}
        self.mock_trainer.train = Mock()
        self.mock_trainer_class = Mock(return_value=self.mock_trainer)

        # Create storage and logging mocks
        self.mock_storage_handler = Mock()
        if use_cloud_logger:
            self.mock_cloud_logger = Mock()
            self.mock_tensorboard_logger = None
        else:
            self.mock_tensorboard_logger = Mock()
            self.mock_cloud_logger = None
        self.mock_login = Mock()

    def setup_mocks(self):
        """Apply all monkeypatch.setattr calls for run_training mocks."""
        self.monkeypatch.setattr(
            "neuracore.ml.train.setup_distributed", self.mock_setup
        )
        self.monkeypatch.setattr(
            "neuracore.ml.train.cleanup_distributed", self.mock_cleanup
        )
        self.monkeypatch.setattr(
            "neuracore.ml.train.DistributedTrainer", self.mock_trainer_class
        )
        self.monkeypatch.setattr(
            "neuracore.ml.train.get_model_and_algorithm_config",
            self.mock_get_model_config,
        )
        self.monkeypatch.setattr(
            "neuracore.ml.train.TrainingStorageHandler", self.mock_storage_handler
        )
        self.monkeypatch.setattr("neuracore.login", self.mock_login)

        if self.mock_cloud_logger is not None:
            self.monkeypatch.setattr(
                "neuracore.ml.train.CloudTrainingLogger", self.mock_cloud_logger
            )
        else:
            self.monkeypatch.setattr(
                "neuracore.ml.train.TensorboardTrainingLogger",
                self.mock_tensorboard_logger,
            )

    def call_run_training(self, cfg, dataset):
        """Call run_training with the configured parameters."""
        return run_training(self.rank, self.world_size, cfg, self.batch_size, dataset)


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_description():
    return DatasetDescription(
        joint_positions=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_velocities=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_target_positions=DataItemStats(mean=[0.0] * 7, std=[1.0] * 7, max_len=7),
    )


@pytest.fixture
def model_init_description(sample_dataset_description):
    return ModelInitDescription(
        dataset_description=sample_dataset_description,
        input_data_types=[DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES],
        output_data_types=[DataType.JOINT_TARGET_POSITIONS],
        output_prediction_horizon=5,
    )


@pytest.fixture
def mock_model_class():
    class MockModel(NeuracoreModel):
        def __init__(self, model_init_description, **kwargs):
            super().__init__(model_init_description)
            self.kwargs = kwargs
            # Add a dummy parameter so optimizer can be created
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            return ModelPrediction(
                outputs={DataType.JOINT_TARGET_POSITIONS: torch.zeros(1, 5, 7)}
            )

        def training_step(self, batch):
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
    sample = Mock()
    sample.inputs = Mock()
    sample.outputs = Mock()

    dataset = Mock(spec=SingleSampleDataset)
    dataset.dataset_description = model_init_description.dataset_description
    dataset.load_sample.return_value = sample
    dataset.__len__ = Mock(return_value=100)
    dataset.collate_fn = lambda x: x
    return dataset


@pytest.fixture
def mock_cfg_batch_size(temp_output_dir):
    return OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": str(temp_output_dir),
        "input_data_types": ["JOINT_POSITIONS", "JOINT_VELOCITIES"],
        "output_data_types": ["JOINT_TARGET_POSITIONS"],
        "output_prediction_horizon": 5,
        "max_batch_size": 32,
        "min_batch_size": 2,
        "batch_size_autotuning_num_workers": 0,
        "max_prefetch_workers": 4,
    })


@pytest.fixture
def mock_cfg_training(temp_output_dir):
    return OmegaConf.create({
        "algorithm_id": "test-algorithm-id",
        "local_output_dir": str(temp_output_dir),
        "seed": 42,
        "validation_split": 0.2,
        "input_data_types": ["JOINT_POSITIONS"],
        "output_data_types": ["JOINT_TARGET_POSITIONS"],
        "output_prediction_horizon": 5,
        "num_train_workers": 0,
        "num_val_workers": 0,
        "epochs": 1,
        "logging_frequency": 10,
        "keep_last_n_checkpoints": 3,
        "training_id": None,
        "resume_checkpoint_path": None,
        "max_prefetch_workers": 4,
    })


@pytest.fixture
def mock_dataset(sample_dataset_description):
    dataset = Mock(spec=PytorchSynchronizedDataset)
    dataset.dataset_description = sample_dataset_description
    dataset.collate_fn = lambda x: x
    dataset.__len__ = Mock(return_value=100)
    dataset.tokenize_text = None
    return dataset


class TestSetupLogging:
    """Tests for setup_logging function."""

    @pytest.mark.parametrize("rank,should_create_log_file", [(0, True), (1, False)])
    def test_setup_logging(self, temp_output_dir, rank, should_create_log_file):
        setup_logging(str(temp_output_dir), rank=rank)

        log_file = temp_output_dir / "train.log"
        if should_create_log_file:
            assert log_file.exists()
        else:
            assert not log_file.exists()

        logger = logging.getLogger(__name__)
        assert logger.level <= logging.INFO

    def test_setup_logging_creates_directory(self, temp_output_dir):
        new_dir = temp_output_dir / "new_dir"
        setup_logging(str(new_dir), rank=0)

        assert new_dir.exists()
        assert (new_dir / "train.log").exists()


class TestConvertDataTypes:
    """Tests for convert_data_types function."""

    @pytest.mark.parametrize(
        "data_types,expected_result,should_raise",
        [
            (
                ["JOINT_POSITIONS", "RGB_IMAGE", "JOINT_VELOCITIES"],
                [
                    DataType.JOINT_POSITIONS,
                    DataType.RGB_IMAGE,
                    DataType.JOINT_VELOCITIES,
                ],
                False,
            ),
            ([], [], False),
            (["INVALID_DATA_TYPE"], None, True),
        ],
    )
    def test_convert_data_types_valid_and_invalid_cases(
        self, data_types, expected_result, should_raise
    ):
        if should_raise:
            with pytest.raises(ValueError):
                convert_data_types(data_types)
        else:
            result = convert_data_types(data_types)
            assert result == expected_result


class TestGetModelAndAlgorithmConfig:
    """Tests for get_model_and_algorithm_config function."""

    def test_get_model_and_algorithm_config_with_algorithm_config_dict(
        self, model_init_description, mock_model_class, monkeypatch
    ):
        cfg = OmegaConf.create({
            "algorithm": {
                "_target_": "tests.unit.ml.test_train.mock_model_class",
            },
        })

        mock_instantiate = Mock(return_value=mock_model_class(model_init_description))
        monkeypatch.setattr(
            "neuracore.ml.train.hydra.utils.instantiate", mock_instantiate
        )

        model, algorithm_config = get_model_and_algorithm_config(
            cfg, model_init_description
        )

        assert isinstance(model, NeuracoreModel)
        assert isinstance(algorithm_config, dict)
        mock_instantiate.assert_called_once()

    def test_get_model_and_algorithm_config_with_algorithm_id_and_params(
        self, model_init_description, mock_model_class, temp_output_dir, monkeypatch
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

    def test_get_model_and_algorithm_config_with_algorithm_id_no_params(
        self, model_init_description, mock_model_class, temp_output_dir, monkeypatch
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

    def test_get_model_and_algorithm_config_raises_error_when_no_algorithm_or_id(
        self, model_init_description
    ):
        cfg = OmegaConf.create({
            "algorithm_id": None,
        })

        with pytest.raises(ValueError, match="Either 'algorithm' or 'algorithm_id'"):
            get_model_and_algorithm_config(cfg, model_init_description)

    def test_get_model_and_algorithm_config_prefers_algorithm_when_both_provided(
        self, model_init_description, mock_model_class, monkeypatch
    ):
        cfg = OmegaConf.create({
            "algorithm": {"_target_": "tests.unit.ml.test_train.mock_model_class"},
            "algorithm_id": "test-id",
        })

        mock_instantiate = Mock(return_value=mock_model_class(model_init_description))
        monkeypatch.setattr(
            "neuracore.ml.train.hydra.utils.instantiate", mock_instantiate
        )

        model, algorithm_config = get_model_and_algorithm_config(
            cfg, model_init_description
        )

        assert isinstance(model, NeuracoreModel)
        mock_instantiate.assert_called_once()

    def test_get_model_with_algorithm_config_removes_target(
        self, model_init_description, mock_model_class, monkeypatch
    ):
        cfg = OmegaConf.create({
            "algorithm": {
                "_target_": "tests.unit.ml.test_train.mock_model_class",
                "param1": "value1",
                "param2": "value2",
            },
        })

        mock_instantiate = Mock(return_value=mock_model_class(model_init_description))
        monkeypatch.setattr(
            "neuracore.ml.train.hydra.utils.instantiate", mock_instantiate
        )

        model, algorithm_config = get_model_and_algorithm_config(
            cfg, model_init_description
        )

        assert isinstance(model, NeuracoreModel)
        assert "_target_" not in algorithm_config
        assert algorithm_config == {"param1": "value1", "param2": "value2"}
        mock_instantiate.assert_called_once()


class TestDetermineOptimalBatchSize:
    """Tests for determine_optimal_batch_size function."""

    @pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
    def test_determine_optimal_batch_size_on_gpu_returns_optimal_size(
        self,
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
        monkeypatch.setattr(
            "neuracore.ml.train.find_optimal_batch_size", mock_find_optimal
        )
        monkeypatch.setattr(
            "neuracore.ml.train.get_model_and_algorithm_config", mock_get_model_config
        )
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        result = determine_optimal_batch_size(
            mock_cfg_batch_size, mock_single_sample_dataset
        )

        assert result == 16
        mock_find_optimal.assert_called_once()

    def test_determine_optimal_batch_size_raises_error_when_no_gpu(
        self, mock_cfg_batch_size, mock_single_sample_dataset, monkeypatch
    ):
        mock_get_device = Mock(return_value=torch.device("cpu"))
        monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        with pytest.raises(ValueError, match="Autotuning is only supported on GPUs"):
            determine_optimal_batch_size(
                mock_cfg_batch_size, mock_single_sample_dataset
            )

    @pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
    def test_determine_optimal_batch_size_with_explicit_device(
        self,
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

        monkeypatch.setattr(
            "neuracore.ml.train.find_optimal_batch_size", mock_find_optimal
        )
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

    @pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
    def test_determine_optimal_batch_size_uses_default_min_max_when_not_specified(
        self,
        mock_single_sample_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "local_output_dir": "/tmp/test",
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "batch_size_autotuning_num_workers": 0,
        })
        mock_get_device = Mock(return_value=torch.device("cuda:0"))
        mock_find_optimal = Mock(return_value=4)
        mock_get_model_config = Mock(
            return_value=(mock_model_class(model_init_description), {})
        )

        monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
        monkeypatch.setattr(
            "neuracore.ml.train.find_optimal_batch_size", mock_find_optimal
        )
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

    @pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
    def test_determine_optimal_batch_size_calls_gc_and_cuda_cleanup(
        self,
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
        monkeypatch.setattr(
            "neuracore.ml.train.find_optimal_batch_size", mock_find_optimal
        )
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

    @pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
    def test_determine_optimal_batch_size_uses_default_device_when_none_provided(
        self,
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
        monkeypatch.setattr(
            "neuracore.ml.train.find_optimal_batch_size", mock_find_optimal
        )
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

    def test_determine_optimal_batch_size_raises_error_when_cpu_device_provided(
        self, mock_cfg_batch_size, mock_single_sample_dataset, monkeypatch
    ):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="Autotuning is only supported on GPUs"):
            determine_optimal_batch_size(
                mock_cfg_batch_size, mock_single_sample_dataset, device=device
            )


class TestRunTraining:
    """Tests for run_training function."""

    def test_run_training_on_single_gpu_without_distributed_setup(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_trainer_class.assert_called_once()
        # Without resume_checkpoint_path, training starts at epoch 0
        setup.mock_trainer.train.assert_called_once_with(start_epoch=0)
        setup.mock_setup.assert_not_called()
        setup.mock_cleanup.assert_not_called()

    def test_run_training_uses_cloud_logger_when_training_id_provided(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.training_id = "test-training-id"
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            use_cloud_logger=True,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_cloud_logger.assert_called_once_with(training_id="test-training-id")
        setup.mock_trainer.train.assert_called_once()

    def test_run_training_resumes_from_checkpoint_with_correct_epoch(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.resume_checkpoint_path = "/path/to/checkpoint.pth"
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            checkpoint_epoch=5,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_trainer.load_checkpoint.assert_called_once_with(
            "/path/to/checkpoint.pth"
        )
        setup.mock_trainer.train.assert_called_once_with(start_epoch=6)

    def test_run_training_starts_at_epoch_one_when_checkpoint_has_no_epoch_key(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.resume_checkpoint_path = "/path/to/checkpoint.pth"
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.mock_trainer.load_checkpoint.return_value = {}  # No epoch key
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_trainer.train.assert_called_once_with(start_epoch=1)

    def test_run_training_handles_checkpoint_load_failure_gracefully(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.resume_checkpoint_path = "/path/to/checkpoint.pth"
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.mock_trainer.load_checkpoint.side_effect = FileNotFoundError(
            "Checkpoint not found"
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_trainer.load_checkpoint.assert_called_once_with(
            "/path/to/checkpoint.pth"
        )
        setup.mock_trainer.train.assert_called_once_with(start_epoch=0)

    def test_run_training_sets_up_distributed_training_when_world_size_greater_than_one(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            world_size=2,
            rank=1,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_setup.assert_called_once_with(setup.rank, setup.world_size)
        setup.mock_cleanup.assert_called_once()
        setup.mock_login.assert_called_once()

    def test_run_training_propagates_exception_single_gpu(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            world_size=1,  # Single GPU, so cleanup shouldn't be called
        )
        setup.mock_trainer.train.side_effect = RuntimeError("Training failed")
        setup.setup_mocks()

        with pytest.raises(RuntimeError, match="Training failed"):
            setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_cleanup.assert_not_called()

    def test_run_training_cleans_up_on_exception_in_distributed_mode(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            world_size=2,
            rank=1,
        )
        setup.mock_trainer.train.side_effect = RuntimeError("Training failed")
        setup.setup_mocks()

        with pytest.raises(RuntimeError, match="Training failed"):
            setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_cleanup.assert_called_once()

    def test_run_training_creates_data_loaders_with_specified_workers(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.num_train_workers = 4
        mock_cfg_training.num_val_workers = 2
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        setup.mock_trainer_class.assert_called_once()
        call_kwargs = setup.mock_trainer_class.call_args[1]
        train_loader = call_kwargs["train_loader"]
        val_loader = call_kwargs["val_loader"]

        # Verify DataLoaders have correct number of workers
        assert train_loader.num_workers == 4
        assert val_loader.num_workers == 2
        # Verify datasets are set
        assert train_loader.dataset is not None
        assert val_loader.dataset is not None
        setup.mock_trainer.train.assert_called_once()

    def test_run_training_sets_tokenize_text_on_dataset(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        # Verify tokenize_text was assigned
        assert mock_dataset.tokenize_text == setup.mock_model.tokenize_text

    def test_run_training_creates_distributed_sampler_when_world_size_greater_than_one(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        from torch.utils.data import DistributedSampler

        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            world_size=2,
            rank=1,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        # Verify DistributedTrainer was called with DataLoaders
        assert setup.mock_trainer_class.called
        call_kwargs = setup.mock_trainer_class.call_args[1]
        train_loader = call_kwargs["train_loader"]
        val_loader = call_kwargs["val_loader"]

        # Verify DistributedSampler is used
        assert isinstance(train_loader.sampler, DistributedSampler)
        assert isinstance(val_loader.sampler, DistributedSampler)
        assert train_loader.sampler.rank == 1
        assert train_loader.sampler.num_replicas == 2
        assert train_loader.batch_size == setup.batch_size
        assert val_loader.batch_size == setup.batch_size

    def test_run_training_creates_regular_dataloader_when_world_size_is_one(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        from torch.utils.data import RandomSampler, SequentialSampler

        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
            world_size=1,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        # Verify DistributedTrainer was called with DataLoaders
        assert setup.mock_trainer_class.called
        call_kwargs = setup.mock_trainer_class.call_args[1]
        train_loader = call_kwargs["train_loader"]
        val_loader = call_kwargs["val_loader"]

        # When shuffle=True, PyTorch uses RandomSampler internally
        # When shuffle=False, PyTorch uses SequentialSampler internally
        assert isinstance(train_loader.sampler, RandomSampler)
        assert isinstance(val_loader.sampler, SequentialSampler)
        assert train_loader.batch_size == setup.batch_size
        assert val_loader.batch_size == setup.batch_size

    def test_run_training_creates_training_storage_handler_with_correct_params(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.training_id = "test-training-id"
        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        # Verify TrainingStorageHandler was instantiated correctly
        setup.mock_storage_handler.assert_called_once()
        call_kwargs = setup.mock_storage_handler.call_args[1]
        assert call_kwargs["local_dir"] == mock_cfg_training.local_output_dir
        assert call_kwargs["training_job_id"] == "test-training-id"
        # Verify algorithm_config is passed (empty dict when no custom params)
        assert "algorithm_config" in call_kwargs
        assert isinstance(call_kwargs["algorithm_config"], dict)

    def test_run_training_logs_model_parameter_count(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        # Create a mock logger that will be returned by logging.getLogger
        mock_logger = Mock()
        mock_logger.info = Mock()

        # Mock logging.getLogger to return our mock logger
        # This is necessary because run_training creates its own logger instance
        original_get_logger = logging.getLogger

        def mock_get_logger(name=None):
            if name == "neuracore.ml.train":
                return mock_logger
            return original_get_logger(name)

        monkeypatch.setattr("logging.getLogger", mock_get_logger)

        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.setup_mocks()

        setup.call_run_training(mock_cfg_training, mock_dataset)

        # Verify logger.info was called with parameter count message
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        parameter_log_calls = [
            call for call in info_calls if "parameters" in str(call).lower()
        ]
        assert len(parameter_log_calls) > 0

    def test_run_training_uses_random_split_with_seed(
        self,
        mock_cfg_training,
        mock_dataset,
        model_init_description,
        mock_model_class,
        monkeypatch,
    ):
        mock_cfg_training.seed = 42
        mock_cfg_training.validation_split = 0.2
        mock_dataset.__len__ = Mock(return_value=100)

        # Create mock datasets for random_split to return
        # Use unsafe=True to allow setting __len__
        mock_train_dataset = Mock(unsafe=True)
        mock_train_dataset.__len__ = Mock(return_value=80)
        mock_val_dataset = Mock(unsafe=True)
        mock_val_dataset.__len__ = Mock(return_value=20)

        setup = RunTrainingTestSetup(
            monkeypatch,
            model_init_description,
            mock_model_class,
        )
        setup.setup_mocks()

        # Mock random_split to capture its arguments and return mock datasets
        def mock_random_split_side_effect(dataset, lengths, generator=None):
            return (mock_train_dataset, mock_val_dataset)

        mock_random_split = Mock(side_effect=mock_random_split_side_effect)
        monkeypatch.setattr("neuracore.ml.train.random_split", mock_random_split)

        setup.call_run_training(mock_cfg_training, mock_dataset)

        # Verify random_split was called
        assert mock_random_split.called
        call_kwargs = mock_random_split.call_args[1]
        # Verify generator was created with correct seed
        generator = call_kwargs["generator"]
        assert generator.initial_seed() == mock_cfg_training.seed
        # Verify split sizes are correct
        call_args = mock_random_split.call_args[0]
        assert call_args[1] == [
            80,
            20,
        ]  # train_size=80, val_size=20 for 100 samples with 0.2 split


class TestMain:
    """Tests for main function."""

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
    def test_main_raises_validation_errors_for_invalid_configurations(
        self, monkeypatch, cfg_updates, expected_error_match
    ):
        base_cfg = {
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "local_output_dir": "/tmp/test",
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "max_prefetch_workers": 4,
        }
        base_cfg.update(cfg_updates)
        cfg = OmegaConf.create(base_cfg)

        monkeypatch.setattr("neuracore.ml.train.logger.info", Mock())

        with pytest.raises(ValueError, match=expected_error_match):
            main(cfg)

    def test_main_loads_dataset_by_name_when_dataset_name_provided(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": None,
            "dataset_name": "test-dataset-name",
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks(include_set_organization=True)

        main(cfg)

        setup.mock_get_dataset.assert_called_once_with(name="test-dataset-name")
        setup.mock_set_organization.assert_not_called()

    def test_main_sets_organization_when_org_id_provided(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": "test-org-id",
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks(include_set_organization=True)

        main(cfg)

        setup.mock_set_organization.assert_called_once_with("test-org-id")
        setup.mock_get_dataset.assert_called_once_with(id="test-dataset-id")

    def test_main_uses_algorithm_config_when_algorithm_provided_instead_of_algorithm_id(
        self, monkeypatch, temp_output_dir
    ):
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
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks()

        main(cfg)

        setup.mock_storage_handler_class.assert_not_called()

    def test_main_uses_default_device_when_device_is_none(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks(include_get_default_device=True)

        main(cfg)

        setup.mock_get_default_device.assert_called_once()
        assert setup.mock_run_training.call_args[0][5] == torch.device("cuda:0")

    def test_main_uses_explicit_device_when_device_is_provided(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": "cuda:1",
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks(include_get_default_device=True)

        main(cfg)

        # get_default_device should NOT be called when device is explicitly provided
        setup.mock_get_default_device.assert_not_called()
        # Verify the explicit device is passed to run_training
        assert setup.mock_run_training.call_args[0][5] == torch.device("cuda:1")

    def test_main_uses_provided_batch_size_when_not_auto(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 16,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks(include_determine_optimal_batch_size=True)

        main(cfg)

        setup.mock_determine_optimal_batch_size.assert_not_called()
        assert setup.mock_run_training.call_args[0][3] == 16

    def test_main_loads_algorithm_by_id_when_algorithm_not_in_cfg_but_algorithm_id_provided(  # noqa: E501
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks()

        main(cfg)

        # Verify AlgorithmStorageHandler was called to download the algorithm
        setup.mock_storage_handler_class.assert_called_once_with(
            algorithm_id="test-algorithm-id"
        )
        expected_extract_dir = Path(temp_output_dir) / "algorithm"
        setup.mock_storage_handler.download_algorithm.assert_called_once_with(
            extract_dir=expected_extract_dir
        )

    def test_main_loads_dataset_by_id_when_dataset_id_provided_but_dataset_name_none(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks()

        main(cfg)

        setup.mock_get_dataset.assert_called_once_with(id="test-dataset-id")

    def test_main_converts_string_batch_size_to_int_when_not_auto(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": "16",
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
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
    def test_main_uses_mp_spawn_for_distributed_training_when_world_size_greater_than_one(  # noqa: E501
        self, monkeypatch, temp_output_dir, world_size, should_use_mp_spawn
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
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
            # Compare OmegaConf objects using to_container for proper comparison
            assert OmegaConf.to_container(args_tuple[1]) == OmegaConf.to_container(cfg)
            assert args_tuple[2] == 8  # batch_size
            assert args_tuple[3] == setup.mock_pytorch_dataset
            setup.mock_run_training.assert_not_called()
        else:
            setup.mock_mp_spawn.assert_not_called()
            setup.mock_run_training.assert_called_once()
            assert setup.mock_run_training.call_args[0][0] == 0  # rank
            assert setup.mock_run_training.call_args[0][1] == 1  # world_size

    def test_main_calls_setup_logging(self, monkeypatch, temp_output_dir):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks()

        mock_setup_logging = Mock()
        monkeypatch.setattr("neuracore.ml.train.setup_logging", mock_setup_logging)

        main(cfg)

        mock_setup_logging.assert_called_once_with(cfg.local_output_dir)

    def test_main_calls_dataset_synchronize_with_correct_parameters(
        self, monkeypatch, temp_output_dir
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": 8,
            "input_data_types": ["JOINT_POSITIONS", "JOINT_VELOCITIES"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks()

        main(cfg)

        # Verify synchronize was called with correct parameters
        setup.mock_dataset.synchronize.assert_called_once()
        call_kwargs = setup.mock_dataset.synchronize.call_args[1]
        assert call_kwargs["frequency"] == cfg.frequency
        assert call_kwargs["prefetch_videos"] is True
        # Verify data_types includes both input and output types
        expected_data_types = [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TARGET_POSITIONS,
        ]
        assert set(call_kwargs["data_types"]) == set(expected_data_types)

    def test_main_uses_autotuning_when_batch_size_is_auto(
        self, monkeypatch, temp_output_dir, model_init_description, mock_model_class
    ):
        cfg = OmegaConf.create({
            "algorithm_id": "test-algorithm-id",
            "dataset_id": "test-dataset-id",
            "dataset_name": None,
            "org_id": None,
            "device": None,
            "local_output_dir": str(temp_output_dir),
            "batch_size": "auto",
            "input_data_types": ["JOINT_POSITIONS"],
            "output_data_types": ["JOINT_TARGET_POSITIONS"],
            "output_prediction_horizon": 5,
            "frequency": 30,
            "algorithm_params": None,
            "max_batch_size": 32,
            "min_batch_size": 2,
            "batch_size_autotuning_num_workers": 0,
            "max_prefetch_workers": 4,
        })

        setup = MainTestSetup(monkeypatch)
        setup.setup_mocks(include_determine_optimal_batch_size=True)

        # Mock SingleSampleDataset
        mock_single_sample_dataset = Mock(spec=SingleSampleDataset)
        mock_single_sample_dataset_class = Mock(return_value=mock_single_sample_dataset)
        monkeypatch.setattr(
            "neuracore.ml.train.SingleSampleDataset", mock_single_sample_dataset_class
        )

        # Mock sample for load_sample
        mock_sample = Mock()
        setup.mock_pytorch_dataset.load_sample.return_value = mock_sample
        setup.mock_pytorch_dataset.__len__ = Mock(return_value=100)

        # Mock get_default_device
        mock_get_device = Mock(return_value=torch.device("cuda:0"))
        monkeypatch.setattr("neuracore.ml.train.get_default_device", mock_get_device)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        # Mock determine_optimal_batch_size to return a value
        setup.mock_determine_optimal_batch_size.return_value = 16

        main(cfg)

        # Verify SingleSampleDataset was created
        mock_single_sample_dataset_class.assert_called_once()
        # Verify determine_optimal_batch_size was called
        setup.mock_determine_optimal_batch_size.assert_called_once()
        # Verify run_training was called with the optimal batch size
        assert setup.mock_run_training.call_args[0][3] == 16
