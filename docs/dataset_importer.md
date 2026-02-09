# Neuracore Dataset Importer

The `neuracore.importer` module provides utilities for importing and uploading robot datasets to Neuracore. It supports multiple dataset formats and provides a flexible configuration system for mapping data from source datasets to Neuracore's data types.

## Overview

This module enables you to:
- Import datasets from RLDS, LeRobot, and TFDS formats
- Automatically detect dataset types based on file structure
- Map various data types (images, joint states, poses, language, etc.) from source datasets
- Upload datasets to Neuracore with proper robot model associations
- Process datasets in parallel using multiprocessing

## Installation

To use the importer module, you need to install `neuracore` with the `[import]` extra, which includes the required dependencies for dataset import:

```bash
pip install neuracore[import]
```

## Supported Dataset Formats

### RLDS (Reinforcement Learning Datasets)
RLDS is a standard format for reinforcement learning datasets. The importer detects RLDS datasets by looking for markers like:
- `rlds_metadata.json` or `rlds_metadata.jsonl`
- `rlds_description.pbtxt` or `rlds_description.pb`
- `episode_metadata.jsonl` or `step_metadata.jsonl`

### LeRobot
LeRobot datasets use Hugging Face's Arrow format. Detection markers include:
- `dataset_infos.json`
- `dataset_dict.json`
- `state.json`
- Arrow (`.arrow`) or Parquet (`.parquet`) data files

### TFDS (TensorFlow Datasets)
TFDS format detection looks for:
- `dataset_info.json`
- `features.json`
- `dataset_state.json`

**Note:** TFDS upload is currently not yet implemented.


## Usage

### Command-Line Interface

The main entry point is the `importer.py` script:

```bash
python -m neuracore.importer.importer \
    --dataset-config path/to/config.yaml \
    --dataset-dir path/to/dataset \
    [--robot-dir path/to/robot/description/files] \
    [--overwrite]
```

#### Required Arguments

- `--dataset-config`: Path to the dataset configuration YAML file
- `--dataset-dir`: Path to the directory containing the dataset

#### Optional Arguments

- `--robot-dir`: Path to robot description files (URDF/MJCF)
- `--overwrite`: Delete existing dataset before uploading if it already exists

### Configuration File

The dataset configuration file (YAML) defines how to map data from the source dataset to Neuracore. See `config/example.yaml` for a complete example.

#### Basic Structure

```yaml
input_dataset_name: my_dataset
dataset_type: RLDS  # Optional: RLDS | LEROBOT | TFDS (auto-detected if omitted)

output_dataset:
  name: my_output_dataset
  tags: [tag1, tag2]
  description: "Dataset description"

robot:
  name: my_robot
  urdf_path: ""  # Path to URDF file (or use mjcf_path)
  overwrite_existing: true

frequency: 50.0  # Data frequency in Hz

data_import_config:
  # Define data mappings here
  RGB_IMAGES:
    source: observation
    format:
      image_convention: CHANNELS_LAST
      order_of_channels: RGB
    mapping:
      - name: image
        source_name: image
```

#### Supported Data Types

The importer supports the following data types:

- **RGB_IMAGES**: RGB camera images
- **DEPTH_IMAGES**: Depth images
- **POINT_CLOUDS**: 3D point clouds
- **JOINT_POSITIONS**: Robot joint positions
- **JOINT_VELOCITIES**: Robot joint velocities
- **JOINT_TORQUES**: Robot joint torques
- **JOINT_TARGET_POSITIONS**: Target joint positions issued to the robot
- **VISUAL_JOINT_POSITIONS**: Joint positions for URDF visualisation but not for training (can be populated from PARALLEL_GRIPPER_OPEN_AMOUNTS)
- **PARALLEL_GRIPPER_OPEN_AMOUNTS**: Gripper open amounts
- **PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS**: Target gripper open amounts issued to the robot
- **END_EFFECTOR_POSES**: End-effector poses
- **POSES**: General 6D poses
- **LANGUAGE**: Language instructions
- **CUSTOM_1D**: Custom 1D data arrays

#### Data Mapping Options

Each data type mapping supports:

- `name`: Name in Neuracore
- `source_name`: Name in source dataset (for dictionary access)
- `index`: Index in source array (for array access)
- `index_range`: Range of indices (for array slicing)
  - `start`: Start index
  - `end`: End index
- `offset`: Offset to apply to the data
- `inverted`: Boolean to flip the sign of the data


## Example Workflow

1. **Prepare your dataset**: Ensure your dataset is in RLDS, LeRobot, or TFDS format
2. **Create configuration file**: Define data mappings in a YAML file
3. **Prepare robot description**: Have URDF or MJCF files ready
4. **Run importer**: Execute the CLI command with appropriate arguments
5. **Monitor progress**: Check logs for upload progress and any errors
