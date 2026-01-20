# Open Source Training

Neuracore includes a comprehensive training infrastructure with Hydra configuration management for local model development.

## Training Features

- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Automatic Batch Size Tuning**: Find optimal batch sizes automatically
- **Memory Monitoring**: Prevent OOM errors with built-in monitoring
- **TensorBoard Integration**: Comprehensive logging and visualization
- **Checkpoint Management**: Automatic saving and resuming
- **Cloud Integration**: Seamless integration with Neuracore SaaS platform
- **Multi-modal Support**: Images, joint states, language, and custom data types

## Training Structure

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

```bash
# Basic training with Diffusion Policy
python -m neuracore.ml.train algorithm=diffusion_policy dataset_name="my_dataset"

# Train ACT with custom algorithm hyperparameters
python -m neuracore.ml.train algorithm=act algorithm.lr=5e-4 algorithm.hidden_dim=1024 dataset_name="my_dataset"

# Auto-tune batch size
python -m neuracore.ml.train algorithm=diffusion_policy batch_size=auto dataset_name="my_dataset"

# Hyperparameter sweeps
python -m neuracore.ml.train --multirun algorithm=cnnmlp algorithm.lr=1e-4,5e-4,1e-3 algorithm.hidden_dim=256,512,1024 dataset_name="my_dataset"

# Training with specified modalities
python -m neuracore.ml.train algorithm=pi0 dataset_name="my_multimodal_dataset" input_robot_data_spec={"my_robot": {"JOINT_POSITIONS": ["joint_1", "joint_2", ...], "RGB_IMAGES": ["wrist_camera"], "LANGUAGE": ["task_instruction"]}}
output_robot_data_spec={"my_robot": {"JOINT_TARGET_POSITIONS": ["joint_1", "joint_2", ...]}}
```

## Configuration Management
There are two configs related with the training. The `config/config.yaml` provides the core training parameters:

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

The algorithm specific config file is inside `config/algorithm`:

```yaml
# @package _global_
algorithm:
  _target_: neuracore.ml.algorithms.diffusion_policy.diffusion_policy.DiffusionPolicy
  unet_down_dims: [256, 512, 1024]
  unet_kernel_size: 5
  unet_n_groups: 8
  unet_diffusion_step_embed_dim: 128
  hidden_dim: 64 # spatial_softmax_num_keypoints * 2
  spatial_softmax_num_keypoints: 32
  unet_use_film_scale_modulation: true
  use_pretrained_weights: true
  noise_scheduler_type: "DDPM"
  num_train_timesteps: 100
  num_inference_steps: 100
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "squaredcos_cap_v2"
  clip_sample: true
  clip_sample_range: 1.0
  lr: 1e-4 # default value for batch size 16
  freeze_backbone: false
  lr_backbone: 1e-4 # default value for batch size 16
  weight_decay: 2e-6
  optimizer_betas: [0.95, 0.999]
  optimizer_eps: 1e-8
  prediction_type: "epsilon"
```