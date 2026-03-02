# Training models with Neuracore

You can train robot learning models either on the **cloud** or **locally** on your own machine.

## Training Features

- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Automatic Batch Size Tuning**: Find optimal batch sizes automatically
- **Memory Monitoring**: Prevent OOM errors with built-in monitoring
- **TensorBoard Integration**: Comprehensive logging and visualization
- **Checkpoint Management**: Automatic saving and resuming
- **Cloud Integration**: Seamless integration with Neuracore SaaS platform
- **Multi-modal Support**: Images, joint states, language, and custom data types

---

## ☁️ Cloud training

Train models on the cloud using Neuracore's credits. You can start a training job in two ways.

### From the dashboard

1. Go to the **Training** page on your [web dashboard](https://www.neuracore.com/).
2. Click **+ New training job**.
3. Select your dataset, algorithm (e.g. CNNMLP, Diffusion Policy), GPU type, and resources.
4. Launch the job. Training runs on Neuracore's cloud and uses your account credits.

### From Python

Use `nc.start_training_run(...)` to start a cloud training job (see [example_launch_training.py](../examples/example_launch_training.py)):

```python
import neuracore as nc
from neuracore_types import DataType, RobotDataSpec

nc.login()
dataset = nc.get_dataset("My Dataset")
robot_id = dataset.robot_ids[0]
full_spec = dataset.get_full_data_spec(robot_id)

input_robot_data_spec: RobotDataSpec = {
    robot_id: {
        DataType.JOINT_POSITIONS: full_spec.get(DataType.JOINT_POSITIONS, []),
        DataType.RGB_IMAGES: full_spec.get(DataType.RGB_IMAGES, []),
    }
}
output_robot_data_spec: RobotDataSpec = {
    robot_id: {
        DataType.JOINT_TARGET_POSITIONS: full_spec.get(DataType.JOINT_TARGET_POSITIONS, []),
    }
}

job_data = nc.start_training_run(
    name="MyTrainingJob",
    dataset_name="My Dataset",
    algorithm_name="diffusion_policy",
    frequency=50,
    num_gpus=1,
    gpu_type="NVIDIA_TESLA_V100",
    input_robot_data_spec=input_robot_data_spec,
    output_robot_data_spec=output_robot_data_spec,
    algorithm_config={"batch_size": 32, "epochs": 100, "output_prediction_horizon": 50},
)
```

---

## 💻 Local training

Neuracore includes a comprehensive training infrastructure with Hydra configuration management for local model development.

### Installation

```bash
pip install "neuracore[ml]"
```

### Training structure

```
neuracore/
  ml/
    train.py              # Main training script
    config/               # Hydra configuration files
      config.yaml         # Main configuration
      algorithm/          # Algorithm-specific configs
        diffusion_policy.yaml
        act.yaml
        pi0.yaml
        cnnmlp.yaml
        ...
      training/           # Training configurations
      dataset/            # Dataset configurations
    algorithms/           # Built-in algorithms
    datasets/             # Dataset implementations
    trainers/             # Distributed training utilities
    utils/                # Training utilities
```

## Training Command Examples
**Note**: Passing `run_name` is optional. If you don't pass the `run_name`, the system will generate a random name. To better track the training experiment, we highly recommend you to pass a `run_name`. If a run with the same name already exists, training fails by default; set `run_name_auto_increment=true` to use an incremented name (e.g. `my_experiment_1`) instead.
```bash
# Basic training with Diffusion Policy
python -m neuracore.ml.train algorithm=diffusion_policy dataset_name="my_dataset" run_name="my_experiment"

# Train ACT with custom algorithm hyperparameters
python -m neuracore.ml.train algorithm=act algorithm.lr=5e-4 algorithm.hidden_dim=1024 dataset_name="my_dataset" run_name="my_experiment"

# Auto-tune batch size
python -m neuracore.ml.train algorithm=diffusion_policy batch_size=auto dataset_name="my_dataset" run_name="my_experiment"

# Hyperparameter sweeps
python -m neuracore.ml.train --multirun algorithm=cnnmlp algorithm.lr=1e-4,5e-4,1e-3 algorithm.hidden_dim=256,512,1024 dataset_name="my_dataset" run_name="my_experiment"

# Training with specified modalities
python -m neuracore.ml.train algorithm=pi0 dataset_name="my_multimodal_dataset" input_robot_data_spec={"my_robot": {"JOINT_POSITIONS": ["joint_1", "joint_2", ...], "RGB_IMAGES": ["wrist_camera"], "LANGUAGE": ["task_instruction"]}} output_robot_data_spec={"my_robot": {"JOINT_TARGET_POSITIONS": ["joint_1", "joint_2", ...]}} run_name="my_experiment"
```

### Configuration management

There are two configs related to training. The `config/config.yaml` provides the core training parameters:

```yaml
# config/config.yaml
defaults:
  - algorithm: diffusion_policy
  - training: default
  - dataset: default

# Core training parameters
seed: 42
epochs: 100
output_prediction_horizon: 100
validation_split: 0.2
logging_frequency: 50
keep_last_n_checkpoints: 5
device: null  # e.g., "cuda:0", "mps", "cpu"

# Batch size (can be "auto" for automatic tuning or an integer)
batch_size: "auto"

# You can either specify input_data_types/output_data_types or
# input_robot_data_spec/output_robot_data_spec
input_data_types:
  - "JOINT_POSITIONS"
  - "RGB_IMAGES"

output_data_types:
  - "JOINT_TARGET_POSITIONS"

# Dict[str, Dict[DataType, List[str]], e.g., {"my_robot": {"JOINT_POSITIONS": ["joint_1", "joint_2", ...]}}
# You can also pass in an empty dict {} to use all available data for all robots
input_robot_data_spec: null
output_robot_data_spec: null
```

The algorithm-specific config file is inside `config/algorithm`:

```yaml
# @package _global_
algorithm:
  _target_: neuracore.ml.algorithms.diffusion_policy.diffusion_policy.DiffusionPolicy
  unet_down_dims: [256, 512, 1024]
  unet_kernel_size: 5
  unet_n_groups: 8
  unet_diffusion_step_embed_dim: 128
  hidden_dim: 64
  spatial_softmax_num_keypoints: 32
  # ... (see config/algorithm/*.yaml in the repo for full options)
```