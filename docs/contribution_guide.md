# Contribution Guide

We welcome contributions! Please open **github issues** or submit **pull requests** for:
- New algorithms and models
- Performance improvements
- Documentation enhancements
- Bug fixes and feature requests

## Creating Your Custom Algorithm
This guide explains how to create and add your own custom machine learning algorithms to Neuracore. You have two options:

1. **Open Source Contribution**: Submit a PR to add your algorithm to the Neuracore repository
2. **Private Algorithm**: Upload your algorithm directly to your account at neuracore.com

### Understanding Neuracore Models

All Neuracore algorithms must extend the `NeuracoreModel` class. This base class provides the foundation for creating models that can process robot data and generate actions.

#### Key Concepts

- **Data Types**: Neuracore supports various data types (joint positions, RGB images, etc.)
- **Batched Data**: Input and output data is provided in batched form
- **Model Architecture**: You define how your model processes inputs and generates outputs

### Step 1: Extend the NeuracoreModel Class

Your model must inherit from `NeuracoreModel` and implement several required methods:

```python
from typing import Any, cast

import torch
import torch.nn as nn
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    DataItemStats,
    DataType,
    JointDataStats,
    ModelInitDescription,
)

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import MeanStdNormalizer

PROPRIO_NORMALIZER = MeanStdNormalizer
ACTION_NORMALIZER = MeanStdNormalizer


class MyCustomAlgorithm(NeuracoreModel):
    """A custom algorithm for robot control."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        lr: float = 1e-4,
        # Add any other hyperparameters your model needs
    ):
        super().__init__(model_init_description)
        self.lr = lr

        # Access data types and statistics from base class
        # self.input_data_types, self.output_data_types, self.dataset_statistics
        # are available after calling super().__init__()

        # Stats for inputs
        joint_stats = DataItemStats()
        for stat in cast(
            list[JointDataStats],
            self.dataset_statistics.get(DataType.JOINT_POSITIONS, []),
        ):
            joint_stats = joint_stats.concatenate(stat.value)
        if len(joint_stats.mean) == 0:
            raise ValueError("JOINT_POSITIONS must be present in input data types")

        # Stats for outputs
        output_stats: list[DataItemStats] = []
        target_stats = DataItemStats()
        for stat in cast(
            list[JointDataStats],
            self.dataset_statistics.get(DataType.JOINT_TARGET_POSITIONS, []),
        ):
            target_stats = target_stats.concatenate(stat.value)
        if len(target_stats.mean) == 0:
            raise ValueError("JOINT_TARGET_POSITIONS must be present in output data types")
        output_stats.append(target_stats)
        self.max_output_size = len(target_stats.mean)

        # Normalizers
        self.proprio_normalizer = PROPRIO_NORMALIZER(
            name="proprioception", statistics=[joint_stats]
        )
        self.action_normalizer = ACTION_NORMALIZER(name="actions", statistics=output_stats)

        # Output layer predicts entire sequence directly from normalized joints
        self.output_size = self.max_output_size * self.output_prediction_horizon
        self.output_layer = nn.Linear(len(joint_stats.mean), self.output_size)

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Forward pass for inference.

        Args:
            batch: Input batch with observations

        Returns:
            dict[DataType, list[BatchedNCData]]: Model predictions with action sequences
        """
        # Extract and normalize joint positions
        batched_joint_data = cast(
            list[BatchedJointData], batch.inputs[DataType.JOINT_POSITIONS]
        )
        mask = batch.inputs_mask[DataType.JOINT_POSITIONS]

        # Concatenate all joint values: (B, T, num_joints)
        joint_data = torch.cat([bjd.value for bjd in batched_joint_data], dim=-1)
        last_joint = joint_data[:, -1, :]  # (B, num_joints)
        masked_joint = last_joint * mask

        # Normalize
        normalized_joint = self.proprio_normalizer.normalize(masked_joint)

        # Predict directly from normalized joints
        action_preds = self.output_layer(normalized_joint)  # (B, output_size)

        # Reshape to (B, T, action_dim)
        batch_size = len(batch)
        action_preds = action_preds.view(
            batch_size, self.output_prediction_horizon, self.max_output_size
        )

        # Unnormalize predictions
        predictions = self.action_normalizer.unnormalize(action_preds)

        # Format output as dict[DataType, list[BatchedNCData]]
        output_tensors: dict[DataType, list[BatchedNCData]] = {}
        batched_outputs = []
        for i in range(len(self.dataset_statistics[DataType.JOINT_TARGET_POSITIONS])):
            joint_preds = predictions[:, :, i : i + 1]  # (B, T, 1)
            batched_outputs.append(BatchedJointData(value=joint_preds))
        output_tensors[DataType.JOINT_TARGET_POSITIONS] = batched_outputs

        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step - forward pass with loss calculation.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        # Create inference batch from training inputs
        inference_sample = BatchedInferenceInputs(
            inputs=batch.inputs,
            inputs_mask=batch.inputs_mask,
            batch_size=batch.batch_size,
        )

        # Get predictions
        predictions_dict = self.forward(inference_sample)

        # Extract target actions
        if DataType.JOINT_TARGET_POSITIONS not in batch.outputs:
            raise ValueError("JOINT_TARGET_POSITIONS required in batch outputs")

        batched_joints = cast(
            list[BatchedJointData], batch.outputs[DataType.JOINT_TARGET_POSITIONS]
        )
        action_targets = [bjd.value for bjd in batched_joints]
        action_data = torch.cat(action_targets, dim=-1)  # (B, T, num_joints)
        action_data = action_data.view(
            batch.batch_size, self.output_prediction_horizon, -1
        )  # (B, T, action_dim)

        target_actions = self.action_normalizer.normalize(action_data)

        # Get predicted actions (already normalized in forward)
        pred_joints = cast(
            list[BatchedJointData],
            predictions_dict[DataType.JOINT_TARGET_POSITIONS],
        )
        pred_actions = torch.cat([bjd.value for bjd in pred_joints], dim=-1)
        pred_actions = pred_actions.view(
            batch.batch_size, self.output_prediction_horizon, -1
        )
        pred_actions_normalized = self.action_normalizer.normalize(pred_actions)

        # Calculate loss
        losses: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        if self.training:
            losses["mse_loss"] = nn.functional.mse_loss(
                pred_actions_normalized, target_actions
            )

        return BatchedTrainingOutputs(losses=losses, metrics=metrics)

    def configure_optimizers(
        self,
    ) -> list[torch.optim.Optimizer]:
        """Configure optimizer for training.

        Returns:
            list[torch.optim.Optimizer]: List of optimizers
        """
        return [torch.optim.Adam(self.parameters(), lr=self.lr)]

    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        """Return the data types supported by the model.

        Returns:
            set[DataType]: Set of supported input data types
        """
        return {DataType.JOINT_POSITIONS}

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        """Return the data types supported by the model.

        Returns:
            set[DataType]: Set of supported output data types
        """
        return {DataType.JOINT_TARGET_POSITIONS}
```

### Step 2: Required Methods

Your model must implement these methods:

1. **`__init__`**: Initialize your model with necessary layers and components
2. **`forward`**: Define the inference logic of your model
3. **`training_step`**: Define how your model trains on a batch of data
4. **`configure_optimizers`**: Define what optimizers to use for training
5. **`get_supported_input_data_types`**: Declare what input data types your model supports
6. **`get_supported_output_data_types`**: Declare what output data types your model can produce

### Step 3: Optional Methods

1. **`configure_schedulers`**: Define what optimization schedulers to use for training

### Step 4: File Organization

For a basic algorithm, a single Python file containing your model class is sufficient. For more complex algorithms:

```
algorithms/my_algorithm/
├── __init__.py               # Empty or importing from my_algorithm.py
├── my_algorithm.py           # Main model class definition
├── modules.py                # Helper modules and components
└── requirements.txt          # Optional: additional dependencies
```

### Step 5: Algorithm Validation
After implementation finished, you can validate the custom algorithms with:

```bash
# Validate custom algorithms before upload
neuracore-validate /path/to/your/algorithm
```
### Step 6: Test Your Algorithm

Before submitting or uploading your algorithm, you can test it locally by creating a test similar to the ones in `tests/unit/ml/algorithms`

To setup the development environment:

```bash
git clone https://github.com/neuracoreai/neuracore
cd neuracore
pip install -e .[dev,ml]
```

### Adding Your Algorithm to Neuracore

#### Option 1: Open Source Contribution

1. Fork the Neuracore repository
2. Add your algorithm to `neuracore/ml/algorithms/your_algorithm/`
3. Ensure your implementation passes all tests
4. Submit a pull request to the main repository

#### Option 2: Private Algorithm Upload

1. Go to the "Algorithms" tab on Neuracore Dashboard
2. Click the "Upload Algorithm" button
3. Either:
   - Upload a single Python file containing your `NeuracoreModel` extension
   - Upload a ZIP file containing your algorithm directory

After uploading, your algorithm will appear as a trainable option when launching training jobs.


### Tips for Algorithm Development

1. **Start Simple**: Begin with a basic model architecture and gradually add complexity
2. **Study Existing Algorithms**: Look at the algorithms in `neuracore/ml/algorithms/` for examples
3. **Mind Your Dependencies**: If your algorithm requires additional packages, include them in a `requirements.txt` file
4. **Test Thoroughly**: Ensure your model handles all the data types it claims to support
5. **Document Well**: Include docstrings and comments explaining your model's architecture and approach


### Troubleshooting

If you encounter issues with your algorithm:

- Verify that your model correctly handles the batch structure
- Check that your model returns outputs in the expected format
- Ensure all tensor dimensions match what Neuracore expects
- When uploading as a ZIP, make sure your module imports are correctly structured


## Release Process

### Creating PRs

All PRs must follow these conventions:

1. **Version Label**: Add exactly one version label to your PR:
   - `version:major` - Breaking changes
   - `version:minor` - New features
   - `version:patch` - Bug fixes
   - `version:none` - No release (docs, chores, etc.)

2. **Commit Format**: PR title and all commits must use conventional commit format:
   ```
   <prefix>: <description>
   ```
   Valid prefixes: `feat`, `fix`, `chore`, `docs`, `ci`, `test`, `refactor`, `style`, `perf`

   Examples:
   - `feat: add multi-GPU training support`
   - `fix: resolve memory leak in streaming`
   - `chore: update dependencies`

### Pending Changelog

For significant changes (`version:major` or `version:minor`), update `changelogs/pending-changelog.md`:

```markdown
## Summary

This release adds support for multi-GPU training and improves streaming performance by 40%.
```

Simply append your summary to the existing content. This will appear at the top of the release notes.

### Triggering a Release

Releases are manual and triggered via GitHub Actions:

1. Go to **Actions** → **Release Package** → **Run workflow**
2. Optional: Check **dry_run** to preview without publishing
3. The workflow will:
   - Analyze all PRs since last release
   - Determine version bump (highest priority across all PRs)
   - Generate changelog with all PRs grouped by type
   - Bump version and create git tag
   - Publish to PyPI
   - Create GitHub release

**Dry run** shows what would happen without making any changes - useful for testing before a real release.
