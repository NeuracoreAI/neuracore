# Neuracore Python Client

Neuracore is a powerful robotics and machine learning client library for seamless robot data collection, model deployment, and interaction with comprehensive support for custom data types and real-time inference.

## Features

- Easy robot initialization and connection (URDF and MuJoCo MJCF support)
- Streaming data logging with custom data types
- Model endpoint management (local and remote)
- Real-time policy inference and deployment
- Flexible dataset creation and synchronization
- Open source training infrastructure with Hydra configuration
- Custom algorithm development and upload
- Multi-modal data support (joint positions, velocities, RGB images, language, custom data, and more)

## Installation

```bash
pip install neuracore
```

For training and ML development:
```bash
pip install neuracore[ml]
```

For MuJoCo MJCF support:
```bash
pip install neuracore[mjcf]
```

## Quick Start

Ensure you have an account at [neuracore.app](https://www.neuracore.app/)

### Authentication

```python
import neuracore as nc

# This will save your API key locally
nc.login()
```

### Robot Connection

```python
# Connect to a robot with URDF
nc.connect_robot(
    robot_name="MyRobot", 
    urdf_path="/path/to/robot.urdf",
    overwrite=False  # Set to True to overwrite existing robot config
)

# Or connect using MuJoCo MJCF
nc.connect_robot(
    robot_name="MyRobot", 
    mjcf_path="/path/to/robot.xml"
)
```

### Data Collection and Logging

#### Basic Data Logging

```python
import time

# Create a dataset for recording
nc.create_dataset(
    name="My Robot Dataset",
    description="Example dataset with multiple data types"
)

# Start recording
nc.start_recording()

# Log various data types with timestamps
t = time.time()
nc.log_joint_positions({'joint1': 0.5, 'joint2': -0.3}, timestamp=t)
nc.log_joint_velocities({'joint1': 0.1, 'joint2': -0.05}, timestamp=t)
nc.log_joint_target_positions({'joint1': 0.6, 'joint2': -0.2}, timestamp=t)

# Log camera data
nc.log_rgb("top_camera", image_array, timestamp=t)

# Log language instructions
nc.log_language("Pick up the red cube", timestamp=t)

# Log custom data
custom_sensor_data = [1.2, 3.4, 5.6]
nc.log_custom_data("force_sensor", custom_sensor_data, timestamp=t)

# Stop recording
nc.stop_recording()
```

#### Live Data Control

```python
# Stop live data streaming (saves bandwidth, doesn't affect recording)
nc.stop_live_data(robot_name="MyRobot", instance=0)

# Resume live data streaming
nc.start_live_data(robot_name="MyRobot", instance=0)
```

### Dataset Access and Visualization

```python
# Load a dataset
dataset = nc.get_dataset("My Robot Dataset")

# Synchronize data types at a specific frequency
from neuracore_types import DataType

synced_dataset = dataset.synchronize(
    frequency=10,  # Hz
    data_types=[DataType.JOINT_POSITIONS, DataType.RGB_IMAGE, DataType.LANGUAGE]
)

print(f"Dataset has {len(synced_dataset)} episodes")

# Access synchronized data
for episode in synced_dataset[:5]:  # First 5 episodes
    for step in episode:
        joint_pos = step.joint_positions
        rgb_images = step.rgb_images
        language = step.language
        # Process your data
```

### Model Inference

#### Local Model Inference

```python
# Load a trained model locally
policy = nc.policy(train_run_name="MyTrainingJob")

# Or load from file path
# policy = nc.policy(model_file="/path/to/model.nc.zip")

# Set specific checkpoint (optional, defaults to last epoch)
policy.set_checkpoint(epoch=-1)

# Predict actions
predicted_sync_points = policy.predict(timeout=5)
joint_target_positions = [sp.joint_target_positions for sp in predicted_sync_points]
actions = [jtp.numpy() for jtp in joint_target_positions if jtp is not None]
```

#### Remote Model Inference

```python
# Connect to a remote endpoint
try:
    policy = nc.policy_remote_server("MyEndpointName")
    predicted_sync_points = policy.predict(timeout=5)
    # Process predictions...
except nc.EndpointError:
    print("Endpoint not available. Please start it at neuracore.app/dashboard/endpoints")
```

#### Local Server Deployment

```python
# Connect to a local policy server
policy = nc.policy_local_server(train_run_name="MyTrainingJob")
```

## Command Line Tools

Neuracore provides several command-line utilities:

### Authentication
```bash
# Interactive login to save API key
nc-login
```

Use the `--email` and `--password` option if you wish to login non-interactively.

### Organization Management
```bash
# Select your current organization
nc-select-org
```

Use the `--org-name` option if you wish to select the org non-interactively.

### Server Operations
```bash
# Launch local policy server for inference
nc-launch-server --job_id <job_id> --org_id <org_id> [--host <host>] [--port <port>]

# Example:
nc-launch-server --job_id my_job_123 --org_id my_org_456 --host 0.0.0.0 --port 8080
```

**Parameters:**
- `--job_id`: Required. The job ID to run
- `--org_id`: Required. Your organization ID
- `--host`: Optional. Host address (default: 0.0.0.0)
- `--port`: Optional. Port number (default: 8080)

### Algorithm Validation
```bash
# Validate custom algorithms before upload
neuracore-validate /path/to/your/algorithm
```

## Open Source Training

Neuracore includes a comprehensive training infrastructure with Hydra configuration management for local model development.

### Training Structure

```
neuracore/
  ml/
    train.py              # Main training script
    config/               # Hydra configuration files
      config.yaml         # Main configuration
      algorithm/          # Algorithm-specific configs
        diffusion_policy.yaml
        act.yaml
        simple_vla.yaml
        cnnmlp.yaml
        ...
      training/           # Training configurations
      dataset/            # Dataset configurations
    algorithms/           # Built-in algorithms
    datasets/             # Dataset implementations
    trainers/             # Distributed training utilities
    utils/                # Training utilities
```

### Training Examples

```bash
# Basic training with Diffusion Policy
python -m neuracore.ml.train algorithm=diffusion_policy dataset_name="my_dataset"

# Train ACT with custom hyperparameters
python -m neuracore.ml.train algorithm=act algorithm.lr=5e-4 algorithm.hidden_dim=1024 dataset_name="my_dataset"

# Auto-tune batch size
python -m neuracore.ml.train algorithm=diffusion_policy batch_size=auto dataset_name="my_dataset"

# Hyperparameter sweeps
python -m neuracore.ml.train --multirun algorithm=cnnmlp algorithm.lr=1e-4,5e-4,1e-3 algorithm.hidden_dim=256,512,1024 dataset_name="my_dataset"

# Multi-modal training with images and language
python -m neuracore.ml.train algorithm=simple_vla dataset_name="my_multimodal_dataset" input_data_types='["joint_positions","rgb_image","language"]'
```

### Configuration Management

```yaml
# config/config.yaml
defaults:
  - algorithm: diffusion_policy
  - training: default
  - dataset: default

# Core parameters
epochs: 100
batch_size: "auto"
seed: 42

# Multi-modal data support
input_data_types:
  - "joint_positions"
  - "rgb_image"
  - "language"
output_data_types:
  - "joint_target_positions"
```

### Training Features

- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Automatic Batch Size Tuning**: Find optimal batch sizes automatically
- **Memory Monitoring**: Prevent OOM errors with built-in monitoring
- **TensorBoard Integration**: Comprehensive logging and visualization
- **Checkpoint Management**: Automatic saving and resuming
- **Cloud Integration**: Seamless integration with Neuracore SaaS platform
- **Multi-modal Support**: Images, joint states, language, and custom data types

## Custom Algorithm Development

Create custom algorithms by extending the `NeuracoreModel` class:

```python
import torch
from neuracore.ml import NeuracoreModel, BatchedInferenceSamples, BatchedTrainingSamples, BatchedTrainingOutputs
from neuracore_types import DataType, ModelInitDescription, ModelPrediction

class MyCustomAlgorithm(NeuracoreModel):
    def __init__(self, model_init_description: ModelInitDescription, **kwargs):
        super().__init__(model_init_description)
        # Your model initialization here
        
    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        # Your inference logic
        pass
        
    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        # Your training logic
        pass
        
    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        # Return list of optimizers
        pass
        
    @staticmethod
    def get_supported_input_data_types() -> list[DataType]:
        return [DataType.JOINT_POSITIONS, DataType.RGB_IMAGE]
        
    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        return [DataType.JOINT_TARGET_POSITIONS]
```

### Algorithm Upload Options

1. **Open Source Contribution**: Submit a PR to the Neuracore repository
2. **Private Upload**: Upload directly at [neuracore.app](https://neuracore.app)
   - Single Python file with your `NeuracoreModel` class
   - ZIP file containing your algorithm directory with `requirements.txt`

## Environment Variables

Configure Neuracore behavior with environment variables (case insensitive, prefixed with `NEURACORE_`):

| Variable                                     | Function                                               | Valid Values   | Default                                                                 |
| -------------------------------------------- | ------------------------------------------------------ | -------------- | ----------------------------------------------------------------------- |
| `NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED` | Allow remote recording triggers                        | `true`/`false` | `true`                                                                  |
| `NEURACORE_PROVIDE_LIVE_DATA`                | Enable live data streaming from this node              | `true`/`false` | `true`                                                                  |
| `NEURACORE_CONSUME_LIVE_DATA`                | Enable live data consumption for inference             | `true`/`false` | `true`                                                                  |
| `NEURACORE_API_URL`                          | Base URL for Neuracore platform                        | URL string     | `https://api.neuracore.app/api`                                         |
| `NEURACORE_API_KEY`                          | An override to the api-key to access the neuracore     | `nrc_XXXX`     | Configured with the [`nc-login`](#authentication) command               |
| `NEURACORE_ORG_ID`                           | An override to select the organization to use.         | A valid UUID   | Configured with the [`nc-select-org`](#organization-management) command |
| `TMPDIR`                                     | Specifies a directory used for storing temporary files | Filepath       | An appropriate folder for your system                                   |


## Performance Considerations

### Bandwidth Optimization
- Use appropriate camera resolutions
- Log only necessary joint states
- Maintain consistent joint combinations (max 50 concurrent streams)
- Consider hardware-accelerated H.264 encoding for video

### Processing Optimization
- Enable hardware acceleration for video encoding
- Limit simultaneous dashboard viewers during recording
- Distribute data collection across multiple machines when needed
- Use `nc.stop_live_data()` when live monitoring isn't required

## Documentation

- [Creating Custom Algorithms](./docs/creating_custom_algorithms.md)
- [Performance Limitations](./docs/limitations.md)
- [Examples](./examples/README.md)

## Development Setup

```bash
git clone https://github.com/neuracoreai/neuracore
cd neuracore
pip install -e .[dev,ml]
```

## Testing

```bash
export NEURACORE_API_URL=http://localhost:8000/api
pytest tests/
```

## Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests for:
- New algorithms and models
- Performance improvements
- Documentation enhancements
- Bug fixes and feature requests
