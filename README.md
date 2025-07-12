# Neuracore Python Client

Neuracore is a powerful robotics and machine learning client library for seamless robot data collection, model deployment, and interaction.

## Features

- Easy robot initialization and connection
- Streaming data logging
- Model endpoint management
- Local and remote model support
- Flexible dataset creation
- Open source training infrastructure with Hydra configuration

## Installation

```bash
pip install neuracore
```

For training and ML development:
```bash
pip install neuracore[ml]
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
# Connect to a robot
nc.connect_robot(
    robot_name="MyRobot", 
    urdf_path="/path/to/robot.urdf"
)
```

You can also upload MuJoCo MJCF rather than URDF. 
For that, ensure you install extra dependencies: `pip install neuracore[mjcf]`.

```python
nc.connect_robot(
    robot_name="MyRobot", 
    mjcf_path="/path/to/robot.xml"
)
```

### Data Logging

```python
# Log joint positions
nc.log_joint_positions({
    'joint1': 0.5, 
    'joint2': -0.3
})

# Log RGB camera image
nc.log_rgb("top_camera", image_array)
```

## Command Line Commands

Neuracore provides several command-line tools for authentication, organization management, and server operations:

### Authentication
```bash
# Generate and save API key (interactive login)
nc-login
```

### Organization Management
```bash
# Select current organization (interactive selection)
nc-select-org
```

### Server Operations
```bash
# Launch local policy server
nc-launch-server --job_id <job_id> --org_id <org_id> [--host <host>] [--port <port>]

# Example:
nc-launch-server --job_id my_job_123 --org_id my_org_456 --host 0.0.0.0 --port 8080
```

**Parameters:**
- `--job_id`: Required. The job ID to run
- `--org_id`: Required. Your organization ID
- `--host`: Optional. Host address (default: 0.0.0.0)
- `--port`: Optional. Port number (default: 8080)

## Open Source Training

Neuracore includes a powerful open-source training infrastructure built with Hydra for configuration management. Train your own robot learning algorithms locally rather than using our cloud training service.

### Training Setup

The training system is located in `neuracore/ml/` and includes:

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
        ...
      training/           # Training-specific configs
      dataset/            # Dataset-specific configs
    algorithms/           # Built-in algorithms
    datasets/             # Dataset implementations
    trainers/             # Distributed training utilities
    utils/                # Training utilities
```

### Quick Training Examples

#### Local Development Training
```bash
# Train with built-in Diffusion Policy
python -m neuracore.ml.train algorithm=diffusion_policy dataset_name="my_dataset"

# Train with ACT algorithm and custom parameters
python -m neuracore.ml.train algorithm=act algorithm.lr=5e-4 algorithm.hidden_dim=1024 dataset_name="my_dataset"

# Auto-tune batch size
pythonv neuracore.ml.train algorithm=diffusion_policy batch_size=auto dataset_name="my_dataset"

# Hyperparameter sweeps
python -m neuracore.ml.train --multirun algorithm=cnnmlp algorithm.lr=1e-4,5e-4,1e-3 algorithm.hidden_dim=256,512,1024 dataset_name="my_dataset"
```


### Configuration Management

Neuracore uses Hydra for flexible configuration management:

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

# Data types
input_data_types:
  - "joint_positions"
  - "rgb_image"
output_data_types:
  - "joint_target_positions"
```

### Custom Algorithm Development

Create your own algorithms by inheriting from `NeuracoreModel`:

```python
from neuracore.ml import NeuracoreModel
from neuracore.core.nc_types import DataType

class MyCustomAlgorithm(NeuracoreModel):
    def __init__(self, model_init_description, **kwargs):
        super().__init__(model_init_description)
        # Your initialization here
        
    def forward(self, batch):
        # Your inference logic
        pass
        
    def training_step(self, batch):
        # Your training logic
        pass
        
    @staticmethod
    def get_supported_input_data_types():
        return [DataType.JOINT_POSITIONS, DataType.RGB_IMAGE]
        
    @staticmethod
    def get_supported_output_data_types():
        return [DataType.JOINT_TARGET_POSITIONS]
```

### Features

- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Automatic Batch Size Tuning**: Find optimal batch sizes automatically
- **Memory Monitoring**: Prevent OOM errors with built-in memory monitoring
- **TensorBoard Integration**: Comprehensive logging and visualization
- **Checkpoint Management**: Automatic saving and resuming
- **Cloud Integration**: Seamless integration with Neuracore SaaS platform
- **Flexible Data Types**: Support for multi-modal robot data (images, joint states, language)

### Algorithm Validation

Validate your algorithms before training:

```bash
neuracore-validate /path/to/your/algorithm
```


## Documentation

 - [Examples](./examples/README.md)
 - [Creating Custom Algorithms](./docs/creating_custom_algorithms.md)
 - [Limitations](./docs/limitations.md)

## Development

To set up for development:

```bash
git clone https://github.com/neuraco/neuracore
cd neuracore
pip install -e .[dev,ml]
```

## Environment Variables

A few environment variables effect how this library operates, they are generally prefixed with `NEURACORE_` and are case insensitive

 | Variable                                     | Function                                                                                                            | Valid Values                               | Default Value                   |
 | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------- |
 | `NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED` | Allows you to disable other machines starting a recording when logging. this does not affect the live data          | `true`/`false`                             | `true`                          |
 | `NEURACORE_LIVE_DATA_ENABLED`                | Allows you to disable the streaming of data for live visualizations from this node. This does not affect recording. | `true`/`false`                             | `true`                          |
 | `NEURACORE_API_URL`                          | The base url used to contact the neuracore platform.                                                                | A url e.g. `https://api.neuracore.app/api` | `https://api.neuracore.app/api` |

## Testing

```bash
export NEURACORE_API_URL=http://localhost:8000/api
pytest tests/
```

## Contributing

Contributions are welcome!
