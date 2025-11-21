"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import logging
from typing import Callable, Dict, List, Optional, cast

import numpy as np
import torch
from neuracore_types import DataType, SynchronizedPoint

import neuracore as nc
from neuracore.core.data.synchronized_dataset import SynchronizedDataset
from neuracore.core.data.synchronized_episode import SynchronizedEpisode
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples
CHECK_MEMORY_INTERVAL = 100


class PytorchSynchronizedDataset(PytorchNeuracoreDataset):
    """Dataset for loading episodic robot data from GCS with filesystem caching.

    Enhanced to support all data types including depth images, point clouds,
    poses, end-effectors, and custom sensor data.
    """

    def __init__(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int,
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """Initialize the dataset.

        Args:
            synchronized_dataset: The synchronized dataset to load data from.
            input_data_types: List of input data types to include in the dataset.
            output_data_types: List of output data types to include in the dataset.
            output_prediction_horizon: Number of future timesteps to predict.
            tokenize_text: Optional function to tokenize text data.
        """
        if not isinstance(synchronized_dataset, SynchronizedDataset):
            raise TypeError(
                "synchronized_dataset must be an instance of SynchronizedDataset"
            )
        super().__init__(
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
            tokenize_text=tokenize_text,
            num_recordings=len(synchronized_dataset),
        )
        self.synchronized_dataset = synchronized_dataset
        self.dataset_statistics = self.synchronized_dataset.dataset_statistics

        self._max_error_count = 100
        self._error_count = 0
        self._memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=1.0, gpu_id=None
        )
        self._mem_check_counter = 0
        self._num_samples = self.synchronized_dataset.num_transitions
        self._logged_in = False

    @staticmethod
    def _get_timestep(episode_length: int) -> int:
        max_start = max(0, episode_length)
        return np.random.randint(0, max_start - 1)

    def load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> TrainingSample:
        """Load sample from cache or GCS with full data type support."""
        if not self._logged_in:
            nc.login()
            self._logged_in = True

        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1

        synced_recording = self.synchronized_dataset[episode_idx]
        synced_recording = cast(SynchronizedEpisode, synced_recording)
        episode_length = len(synced_recording)
        if timestep is None:
            timestep = self._get_timestep(episode_length)

        TrainingSample(
            output_prediction_mask=torch.ones(
                (self.output_prediction_horizon,), dtype=torch.float32
            ),
        )
        sync_point = cast(SynchronizedPoint, synced_recording[timestep])
        future_sync_points = cast(
            list[SynchronizedPoint],
            synced_recording[
                timestep + 1 : timestep + 1 + self.output_prediction_horizon
            ],
        )
        # Padding for future sync points
        for _ in range(self.output_prediction_horizon - len(future_sync_points)):
            future_sync_points.append(future_sync_points[-1])

        inputs: Dict[DataType, Dict[str, MaskableData]] = {}
        for data_type in self.input_data_types:
            inputs[data_type] = {}
            for name, nc_data in sync_point.data[data_type].items():
                tensor = torch.from_numpy(nc_data.numpy())
                num_existing = tensor.shape[0]
                stats = self.dataset_statistics.data[data_type][name]
                extra_states = stats.max_len - num_existing
                if extra_states > 0:
                    tensor = torch.cat(
                        [tensor, torch.zeros(extra_states, dtype=torch.float32)], dim=0
                    )
                tensor_mask = torch.tensor(
                    [1.0] * num_existing + [0.0] * extra_states, dtype=torch.float32
                )
                inputs[data_type][name] = MaskableData(
                    data=tensor,
                    mask=tensor_mask,
                )

        outputs: Dict[DataType, Dict[str, MaskableData]] = {}
        for data_type in self.output_data_types:
            outputs[data_type] = {}
            for name in sync_point.data[data_type].keys():
                maskable_data_for_each_t: List[MaskableData] = []
                stats = self.dataset_statistics.data[data_type][name]
                for sp in future_sync_points:
                    nc_data = sp.data[data_type][name]
                    tensor = torch.from_numpy(nc_data.numpy())
                    num_existing = tensor.shape[0]
                    extra_states = stats.max_len - num_existing
                    if extra_states > 0:
                        tensor = torch.cat(
                            [tensor, torch.zeros(extra_states, dtype=torch.float32)],
                            dim=0,
                        )
                    tensor_mask = torch.tensor(
                        [1.0] * num_existing + [0.0] * extra_states, dtype=torch.float32
                    )
                    maskable_data_for_each_t.append(
                        MaskableData(
                            data=tensor,
                            mask=tensor_mask,
                        )
                    )
                stacked_data = torch.stack([md.data for md in maskable_data_for_each_t])
                stacked_mask = torch.stack([md.mask for md in maskable_data_for_each_t])
                outputs[data_type][name] = MaskableData(
                    data=stacked_data,
                    mask=stacked_mask,
                )
        return TrainingSample(
            inputs=inputs,
            outputs=outputs,
            output_prediction_mask=self._create_output_prediction_mask(
                episode_length, timestep, self.output_prediction_horizon
            ),
        )

    def _create_output_prediction_mask(
        self, episode_length: int, timestep: int, output_prediction_horizon: int
    ) -> torch.Tensor:
        """Create mask for output predictions."""
        output_prediction_mask = torch.zeros(
            output_prediction_horizon, dtype=torch.float32
        )
        for i in range(output_prediction_horizon):
            if timestep + i >= episode_length:
                break
            else:
                output_prediction_mask[i] = 1.0
        return output_prediction_mask

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._num_samples
