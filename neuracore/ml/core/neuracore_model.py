"""Abstract base class for models in the Neuracore framework.

This module provides the foundational NeuracoreModel class that all
models must inherit from. It handles data type validation, device management,
and defines the required interface for training and inference.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from neuracore_types import DataType, ModelInitDescription, ModelPrediction

from .ml_types import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
)

logger = logging.getLogger(__name__)


class NeuracoreModel(nn.Module, ABC):
    """Abstract base class for all Neuracore models.

    Provides the foundational structure for all robot learning models in the
    Neuracore framework. Handles automatic device placement, data type validation,
    and defines the required interface for training and inference operations.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
    ):
        """Initialize the Neuracore model.

        Args:
            model_init_description: Model initialization parameters including
                input/output data types, dataset description, and prediction horizon

        Raises:
            ValueError: If requested data types are not supported by the model
                or not present in the dataset
        """
        super().__init__()
        self.model_init_description = model_init_description
        self._validate_input_output_types()
        self.dataset_description = model_init_description.dataset_description
        self.output_prediction_horizon = (
            model_init_description.output_prediction_horizon
        )
        self.robot_to_output_mapping = self._get_robot_to_output_mapping()

    @property
    def device(self) -> torch.device:
        """Get the device for the model.

        Returns:
            torch.device: The device for the model
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # No parameters, check buffers
            try:
                return next(self.buffers()).device
            except StopIteration:
                # No parameters or buffers, default to CPU
                return torch.device("cpu")

    def _validate_input_output_types(self) -> None:
        """Validate that requested data types are supported and available.

        Ensures that all requested input and output data types are both
        supported by the model implementation and present in the dataset.

        Raises:
            ValueError: If any requested data type is not supported or not
                available in the dataset
        """
        req_input_data_types = set(self.model_init_description.input_data_types)
        types_in_dataset = set(
            self.model_init_description.dataset_description.get_data_types()
        )
        input_types_supported_by_model = set(self.get_supported_input_data_types())

        # Check if the requested input data types are in the dataset description
        if not req_input_data_types.issubset(types_in_dataset):
            raise ValueError(
                "Requested input data types not in dataset: "
                f"{req_input_data_types - types_in_dataset}"
            )

        # Check if the requested input data types are supported by the model
        if not req_input_data_types.issubset(input_types_supported_by_model):
            raise ValueError(
                "Requested input data types not supported by model: "
                f"{req_input_data_types - input_types_supported_by_model}"
            )

        req_output_data_types = set(self.model_init_description.output_data_types)
        outut_types_supported_by_model = set(self.get_supported_output_data_types())

        # Check if the requested output data types are supported by the model
        if not req_output_data_types.issubset(outut_types_supported_by_model):
            raise ValueError(
                "Requested output data types not supported by model: "
                f"{req_output_data_types - outut_types_supported_by_model}"
            )
        # Check if the requested output data types are in the dataset description
        if not req_output_data_types.issubset(types_in_dataset):
            raise ValueError(
                "Requested output data types not in dataset: "
                f"{req_output_data_types - types_in_dataset}"
            )

    def _populate_output_mapping(
        self,
        robot_to_output_mapping: dict[str, dict[DataType, list[str]]],
        keys: dict[str, list[str]],
        data_type: DataType,
    ) -> dict[str, dict[DataType, list[str]]]:
        """Populate the output mapping for a given data type.

        Args:
            robot_to_output_mapping: Output mapping per robot.
            keys: Keys for each robot for the given data type.
            data_type: Data type of the mapping.

        Returns:
            dict[str, dict[DataType, list[str]]]: Output mapping per robot.
        """
        for robot_name in keys.keys():
            if robot_name not in robot_to_output_mapping:
                robot_to_output_mapping[robot_name] = {}
            robot_to_output_mapping[robot_name][data_type] = keys[robot_name]
        return robot_to_output_mapping

    def _get_robot_to_output_mapping(self) -> dict[str, dict[DataType, list[str]]]:
        """Get the output mapping from the dataset description for each robot.

        Return the output mapping of robots that have all requested output data types.

        Returns:
            dict[str, dict[DataType, list[str]]]: Output mapping per robot.
        """
        output_data_types = self.model_init_description.output_data_types
        robot_to_output_mapping: dict[str, dict[DataType, list[str]]] = {}
        if DataType.JOINT_TARGET_POSITIONS in output_data_types:
            keys = self.dataset_description.joint_target_positions.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.JOINT_TARGET_POSITIONS
            )
        if DataType.JOINT_POSITIONS in output_data_types:
            keys = self.dataset_description.joint_positions.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.JOINT_POSITIONS
            )
        if DataType.JOINT_VELOCITIES in output_data_types:
            keys = self.dataset_description.joint_velocities.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.JOINT_VELOCITIES
            )
        if DataType.JOINT_TORQUES in output_data_types:
            keys = self.dataset_description.joint_torques.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.JOINT_TORQUES
            )
        if DataType.END_EFFECTORS in output_data_types:
            keys = self.dataset_description.end_effector_states.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.END_EFFECTORS
            )
        if DataType.POSES in output_data_types:
            keys = self.dataset_description.poses.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.POSES
            )
        if DataType.END_EFFECTOR_POSES in output_data_types:
            keys = self.dataset_description.end_effector_poses.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.END_EFFECTOR_POSES
            )
        if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in output_data_types:
            keys = (
                self.dataset_description.parallel_gripper_open_amounts.robot_to_ncdata_keys
            )
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS
            )
        if DataType.RGB_IMAGE in output_data_types:
            keys = self.dataset_description.rgb_images.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.RGB_IMAGE
            )
        if DataType.DEPTH_IMAGE in output_data_types:
            keys = self.dataset_description.depth_images.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.DEPTH_IMAGE
            )
        if DataType.POINT_CLOUD in output_data_types:
            keys = self.dataset_description.point_clouds.robot_to_ncdata_keys
            robot_to_output_mapping = self._populate_output_mapping(
                robot_to_output_mapping, keys, DataType.POINT_CLOUD
            )
        if DataType.LANGUAGE in output_data_types:
            pass  # Language data typically does not require robot-specific keys
        if DataType.CUSTOM in output_data_types:
            for _, data_item_stats in self.dataset_description.custom_data.items():
                keys = data_item_stats.robot_to_ncdata_keys
                for robot_name in keys.keys():
                    if robot_name not in robot_to_output_mapping:
                        robot_to_output_mapping[robot_name] = {DataType.CUSTOM: []}
                    robot_to_output_mapping[robot_name][DataType.CUSTOM].extend(
                        keys[robot_name]
                    )

        # Remove robots that does not have all output data types
        for robot_name in robot_to_output_mapping.keys():
            for data_type in output_data_types:
                if data_type not in robot_to_output_mapping[robot_name]:
                    del robot_to_output_mapping[robot_name]

        return robot_to_output_mapping

    @abstractmethod
    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Perform inference forward pass.

        Args:
            batch: Batched input samples for inference

        Returns:
            ModelPrediction: Model predictions with appropriate structure
        """
        pass

    @abstractmethod
    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Args:
            batch: Batched training samples including inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs including loss and metrics
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure and return optimizers for the model.

        Returns:
            list[torch.optim.Optimizer]: List of optimizers for model parameters
        """
        pass

    @staticmethod
    def tokenize_text(text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text input for language processing.

        Args:
            text: List of text strings to tokenize

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tokenized text and attention masks

        Raises:
            NotImplementedError: Must be implemented by subclasses that use text
        """
        raise NotImplementedError("User needs to implement this method")

    @staticmethod
    @abstractmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Get the input data types supported by this model.

        Returns:
            list[DataType]: List of supported input data types
        """
        pass

    @staticmethod
    @abstractmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Get the output data types supported by this model.

        Returns:
            list[DataType]: List of supported output data types
        """
        pass
