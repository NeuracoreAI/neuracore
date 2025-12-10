# Creating Custom Algorithms for Neuracore

This guide explains how to create and add your own custom machine learning algorithms to Neuracore. You have two options:

1. **Open Source Contribution**: Submit a PR to add your algorithm to the Neuracore repository
2. **Private Algorithm**: Upload your algorithm directly to your account at neuracore.app

## Understanding Neuracore Models

All Neuracore algorithms must extend the `NeuracoreModel` class. This base class provides the foundation for creating models that can process robot data and generate actions.

### Key Concepts

- **Data Types**: Neuracore supports various data types (joint positions, RGB images, etc.)
- **Batched Data**: Input and output data is provided in batched form
- **Model Architecture**: You define how your model processes inputs and generates outputs

## Creating Your Custom Algorithm

### Step 1: Extend the NeuracoreModel Class

Your model must inherit from `NeuracoreModel` and implement several required methods:

```python
import torch
import torch.nn as nn
from neuracore_types import DataType, ModelInitDescription, ModelPrediction
from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

class MyCustomAlgorithm(NeuracoreModel):
    """A custom algorithm for robot control."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 512,
        # Add any other hyperparameters your model needs
    ):
        super().__init__(model_init_description)
        self.hidden_dim = hidden_dim
        
        # Initialize your model architecture here
        # For example, if processing joint positions and RGB images:
        if DataType.JOINT_POSITIONS in self.model_init_description.input_data_types:
            joint_dim = self.dataset_statistics.joint_positions.max_len
            self.joint_encoder = nn.Linear(joint_dim, hidden_dim)
            
        if DataType.RGB_IMAGES in self.model_init_description.input_data_types:
            # Initialize image processing components
            self.image_encoder = self._build_image_encoder()
            
        # Initialize output layers based on what the model predicts
        if DataType.JOINT_TARGET_POSITIONS in self.model_init_description.output_data_types:
            output_dim = self.dataset_statistics.joint_target_positions.max_len * self.output_prediction_horizon
            self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def _build_image_encoder(self):
        # Custom method to build image encoder
        # This is just an example
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, self.hidden_dim)
        )
        
    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Forward pass for inference."""
        # Process your inputs here
        # This is where you implement your model's inference logic
        
        # Example for a model processing joint positions and generating joint targets
        joint_features = self.joint_encoder(batch.joint_positions.data)
        predictions = self.output_layer(joint_features)
        
        # Format the output according to Neuracore expectations
        return ModelPrediction(
            outputs={DataType.JOINT_TARGET_POSITIONS: predictions.detach().cpu().numpy()},
            prediction_time=0.0,  # You may want to time this in a real implementation
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step - forward pass with loss calculation."""
        # Create inference batch from training inputs
        inference_batch = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            rgb_images=batch.inputs.rgb_images,
            # Add other data types as needed
        )
        
        # Get the targets
        if DataType.JOINT_TARGET_POSITIONS in self.model_init_description.output_data_types:
            targets = batch.outputs.joint_target_positions.data
            
        # Get predictions
        predictions = self.forward(inference_batch)
        
        # Calculate loss
        loss = nn.functional.mse_loss(predictions.outputs[DataType.JOINT_TARGET_POSITIONS], targets)
        
        return BatchedTrainingOutputs(
            losses={"mse_loss": loss},
            metrics={},  # You can add custom metrics here
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer for training."""
        return [torch.optim.Adam(self.parameters(), lr=1e-4)]

    @staticmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.RGB_IMAGES,
            # Add other supported input types
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [DataType.JOINT_TARGET_POSITIONS]
        
```

### Step 2: Required Methods

Your model must implement these methods:

1. **`__init__`**: Initialize your model with necessary layers and components
2. **`forward`**: Define the inference logic of your model
3. **`training_step`**: Define how your model trains on a batch of data
4. **`configure_optimizers`**: Define what optimizers to use for training
5. **`get_supported_input_data_types`**: Declare what input data types your model supports
6. **`get_supported_output_data_types`**: Declare what output data types your model can produce

### Step 3: File Organization

For a basic algorithm, a single Python file containing your model class is sufficient. For more complex algorithms:

```
my_algorithm/
├── __init__.py               # Empty or importing from my_algorithm.py
├── my_algorithm.py           # Main model class definition
├── modules.py                # Helper modules and components
└── requirements.txt          # Optional: additional dependencies
```

## Adding Your Algorithm to Neuracore

### Option 1: Open Source Contribution

1. Fork the Neuracore repository
2. Add your algorithm to `neuracore/ml/algorithms/your_algorithm/`
3. Ensure your implementation passes all tests
4. Submit a pull request to the main repository

### Option 2: Private Algorithm Upload

1. Go to the "Algorithms" tab on Neuracore Dashboard
2. Click the "Upload Algorithm" button
3. Either:
   - Upload a single Python file containing your `NeuracoreModel` extension
   - Upload a ZIP file containing your algorithm directory

After uploading, your algorithm will appear as a trainable option when launching raining jobs.

## Testing Your Algorithm

Before submitting or uploading your algorithm, you can test it locally by creating a test similar to the ones in `tests/unit/ml/algorithms`


## Tips for Algorithm Development

1. **Start Simple**: Begin with a basic model architecture and gradually add complexity
2. **Study Existing Algorithms**: Look at the algorithms in `neuracore/ml/algorithms/` for examples
3. **Mind Your Dependencies**: If your algorithm requires additional packages, include them in a `requirements.txt` file
4. **Test Thoroughly**: Ensure your model handles all the data types it claims to support
5. **Document Well**: Include docstrings and comments explaining your model's architecture and approach


## Troubleshooting

If you encounter issues with your algorithm:

- Verify that your model correctly handles the batch structure
- Check that your model returns outputs in the expected format
- Ensure all tensor dimensions match what Neuracore expects
- When uploading as a ZIP, make sure your module imports are correctly structured

## Example: Testing with an Existing Algorithm

You can test the upload functionality by zipping one of the existing algorithm folders and uploading it with a new name:

```bash
cd neuracore/ml/algorithms
zip -r cnnmlp_custom.zip cnnmlp/
```

Then upload `cnnmlp_custom.zip` through the dashboard interface to test.