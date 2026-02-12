# Neuracore Dataset Importer

The `neuracore.importer` module provides utilities for importing and uploading robot datasets to Neuracore. It supports multiple dataset formats and provides a flexible configuration system for mapping data from source datasets to Neuracore's data types.

## Overview

This module enables you to:
- Import datasets from RLDS and LeRobot formats
- Automatically detect dataset types based on file structure
- Map various data types (images, joint states, poses, language, etc.) from source datasets
- Upload datasets to Neuracore with proper robot model associations
- Validate data against configuration and robot model constraints

## Installation

To use the importer module, install `neuracore` with the `[import]` extra.
The full installation matrix is documented in [Installation](../README.md#installation).

```bash
# UV-first
uv pip install "neuracore[import]"

# or pip
pip install neuracore[import]
```

## Usage

### Command-Line Interface

```bash
neuracore importer import \
    --dataset-config path/to/config.yaml \
    --dataset-dir path/to/dataset \
    --robot-dir path/to/robot/description/files \
    [OPTIONS]
```


#### Required Arguments

- `--dataset-config` / `-c`: Path to the dataset configuration YAML or JSON file
- `--dataset-dir` / `-d`: Path to the directory containing the dataset
- `--robot-dir` / `-r`: Directory containing robot description files (URDF/MJCF)

#### Optional Arguments

- `--overwrite`: Delete existing dataset before uploading if it already exists
- `--dry-run`: Perform a dry run without logging data to Neuracore (validation only)
- `--skip-on-error`: Error handling strategy (default: `episode`)
  - `episode`: Skip the failed episode and continue
  - `step`: Skip only the failing step and continue with the episode
  - `all`: Abort on the first error
- `--no-validation-warnings`: Suppress warning messages from data validation

### Configuration File

The dataset configuration file (YAML or JSON) defines how to map data from the source dataset to Neuracore. See `neuracore/importer/config/example.yaml` for a complete example.

#### Basic Structure

Each data type mapping supports:

- `source`: Path or key indicating where to find the relevant data in the source dataset. This can be a single key (e.g., `observation`) or a dot-separated path for nested fields (e.g., `observation.image`).
- `format`: Specifies the format of the source data. Source data will be transformed to the format accepted by Neuracore. Available options (not all apply to every data type):

  - `image_convention`: `CHANNELS_LAST` (default) | `CHANNELS_FIRST` - Image channel layout (for RGB_IMAGES)
  - `order_of_channels`: `RGB` (default) | `BGR` - Channel color order (for RGB_IMAGES)
  - `normalized_pixel_values`: `false` (default) | `true` - Whether pixels are normalized [0,1] or [0,255] (for RGB_IMAGES)
  - `angle_units`: `RADIANS` (default) | `DEGREES` - Angle unit conversion (for joint data, poses)
  - `torque_units`: `NM` (default) | `NCM` - Torque unit conversion (for JOINT_TORQUES)
  - `distance_units`: `M` (default) | `MM` - Distance unit conversion (for DEPTH_IMAGES, POINT_CLOUDS)
  - `pose_type`: `MATRIX` (default) | `POSITION_ORIENTATION` - Pose representation format (for POSES, END_EFFECTOR_POSES, JOINT_POSITIONS with TCP)
  - `orientation`: Configuration object (required when `pose_type: POSITION_ORIENTATION`):
    - `type`: `QUATERNION` (default) | `EULER` | `MATRIX` | `AXIS_ANGLE` - Orientation representation
    - `quaternion_order`: `XYZW` (default) | `WXYZ` - Quaternion component order (when `type: QUATERNION`)
    - `euler_order`: `XYZ` (default) | `ZYX` | `YXZ` | `XZY` | `YZX` | `ZXY` - Euler angle order (when `type: EULER`)
    - `angle_units`: `RADIANS` (default) | `DEGREES` - Angle unit for orientation
  - `joint_position_type`: `CUSTOM` (default) | `TCP` - Use custom joint positions or convert from TCP pose via IK (for JOINT_POSITIONS)
  - `ik_init_config`: `list[float] | None` - Initial joint configuration for inverse kinematics (can be provided when `joint_position_type: TCP`)
  - `visual_joint_type`: `CUSTOM` (default) | `GRIPPER` - Use custom visual joints or populate from gripper open amounts (for VISUAL_JOINT_POSITIONS)
  - `invert_gripper_amount`: `false` (default) | `true` - Convert from close amount to open amount (for PARALLEL_GRIPPER_OPEN_AMOUNTS, PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS)
  - `normalize`: Configuration object (optional, for PARALLEL_GRIPPER_OPEN_AMOUNTS, PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS):
    - `min`: `float` (default: `0.0`) - Minimum value for normalization
    - `max`: `float` (default: `1.0`) - Maximum value for normalization
  - `language_type`: `STRING` (default) | `BYTES` - Language data format (for LANGUAGE)

  **Note**: Not all format options apply to all data types. See the data type examples below for specific usage. For simple data types like `CUSTOM_1D`, the format block may be omitted entirely.
- `mapping`: Configuration specific to each item of a data type
  - `name`: Name to be saved in Neuracore (required).
  - `source_name`: The key or field name in the source dataset used to extract the mapping specific data item. In hierarchical sources (like RLDS), this is usually the key within the specified `source` dictionary. For flat (column-based) datasets like LeRobot, this refers to the column name within the relevant Arrow table.

    - **RLDS:** Data is typically stored as dictionaries within each step or episode. Use `source_name` to specify the key in these dictionaries. For example, if step data looks like `{'observation': {'image': ...}}`, using `source: observation` and `source_name: image` will access the desired value.

    - **LeRobot:** LeRobot datasets use the Hugging Face Arrow format, where data is organized in columns accessible via keys. Here, `source_name` refers to the column name in the dataset table. For instance, with `source: observation` and `source_name: image`, the importer will retrieve image data from the Arrow column named "image" within the "observation" structure (if nested).

  - `index`: Index in source array (for array access, e.g., `observation.state[0]`)
  - `index_range`: Range of indices (for array slicing, e.g., `observation.state[0:7]`)
    - `start`: Start index (inclusive)
    - `end`: End index (exclusive)
  - `inverted`: Flip the sign of the data at the end of the transforms
  - `offset`: Offset to apply to the data at the end of the transforms

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
  # mjcf_path: ""  # Alternative to urdf_path
  overwrite_existing: true  # Whether to overwrite existing robot configuration

frequency: 50.0  # Data frequency in Hz (required for RLDS, optional for LeRobot)

data_import_config:
  # Define data mappings here for each data type
  RGB_IMAGES:
    source: observation  # Path to data in source dataset (supports dot notation)
    format:
      image_convention: CHANNELS_LAST  # CHANNELS_FIRST | CHANNELS_LAST
      order_of_channels: RGB  # RGB | BGR
      normalized_pixel_values: false  # Whether pixels are normalized [0,1] or [0,255]
    mapping:
      - name: image  # Name to be saved in Neuracore
        source_name: image  # Name in source dataset (for dictionary access)
```

#### Supported Data Types

The importer supports the following data types:

- **RGB_IMAGES**: RGB camera images with configurable channel order and convention
  
  ```yaml
  RGB_IMAGES:
    source: observation
    format:
      image_convention: CHANNELS_LAST  # CHANNELS_FIRST | CHANNELS_LAST
      order_of_channels: RGB  # RGB | BGR
      normalized_pixel_values: false  # true if pixels are in [0,1], false if [0,255]
    mapping:
      - name: camera_image
        source_name: image
      - name: wrist_camera
        source_name: wrist_image
  ```

- **DEPTH_IMAGES**: Depth images with distance unit configuration (M | MM)
  
  ```yaml
  DEPTH_IMAGES:
    source: observation
    format:
      distance_units: M  # M | MM
    mapping:
      - name: depth_static
        source_name: depth_static
      - name: depth_gripper
        source_name: depth_gripper
  ```
  Note: Automatically converts NaN, positive infinity, and negative infinity values to zero.

- **POINT_CLOUDS**: 3D point clouds (N×3 arrays) with distance unit configuration
  
  ```yaml
  POINT_CLOUDS:
    source: observation
    format:
      distance_units: M  # M | MM
    mapping:
      - name: point_cloud
        source_name: point_cloud
  ```

- **JOINT_POSITIONS**: Robot joint positions with angle unit conversion and IK support
  
  ```yaml
  # Example 1: Direct joint positions from state array
  JOINT_POSITIONS:
    source: observation.state
    format:
      angle_units: RADIANS  # RADIANS | DEGREES
    mapping:
      - name: joint_1
        index: 0
        offset: 0.0
        inverted: false
      - name: joint_2
        index: 1
        offset: -1.5707963267948966  # -π/2
        inverted: true
  
  # Example 2: Joint positions from TCP pose using inverse kinematics
  JOINT_POSITIONS:
    source: observation.pose
    format:
      joint_position_type: TCP  # Use IK to convert pose to joint positions
      ik_init_config: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      pose_type: POSITION_ORIENTATION
      orientation:
        type: QUATERNION
        quaternion_order: XYZW
        angle_units: RADIANS
    mapping:
      - name: end_effector
        index_range:
          start: 0
          end: 7
  ```

- **JOINT_VELOCITIES**: Robot joint velocities with angle unit conversion
  
  ```yaml
  JOINT_VELOCITIES:
    source: observation.state_vel
    format:
      angle_units: RADIANS  # RADIANS | DEGREES
    mapping:
      - name: joint_1
        index: 0
      - name: joint_2
        index: 1
        inverted: true
      - name: joint_3
        index: 2
  ```

- **JOINT_TORQUES**: Robot joint torques with unit conversion (NM | NCM)
  
  ```yaml
  JOINT_TORQUES:
    source: observation.state_torque
    format:
      torque_units: NM  # NM | NCM
    mapping:
      - name: joint_1
        index: 0
      - name: joint_2
        index: 1
        inverted: true
  ```

- **JOINT_TARGET_POSITIONS**: Target joint positions issued to the robot
  
  ```yaml
  JOINT_TARGET_POSITIONS:
    source: action.target_joints
    format:
      angle_units: RADIANS
    mapping:
      - name: joint_1
        index: 0
      - name: joint_2
        index: 1
  ```

- **VISUAL_JOINT_POSITIONS**: Joint positions for URDF visualization but not for training (can be populated from gripper open amounts or by defining custom visual joints)
  
  ```yaml
  # Example 1: Populate from gripper open amounts for standard visual joints
  VISUAL_JOINT_POSITIONS:
    source: observation.state
    format:
      visual_joint_type: GRIPPER
    mapping:
      - name: finger_joint1
        index: 6
        inverted: true
        offset: -0.7853981633974483
      - name: finger_joint2
        index: 6

  # Example 2: Logging custom visual joints using the same joint names
  VISUAL_JOINT_POSITIONS:
    source: observation.state
    format:
      visual_joint_type: CUSTOM
    mapping:
      - name: finger_joint1
        index: 10
        inverted: false
        offset: 0.0
      - name: finger_joint2
        index: 11
        inverted: true
        offset: -0.1
  ```

- **PARALLEL_GRIPPER_OPEN_AMOUNTS**: Gripper open amounts with normalization support
  
  ```yaml
  PARALLEL_GRIPPER_OPEN_AMOUNTS:
    source: observation.state
    format:
      invert_gripper_amount: true # Convert from close amount to open amount
      normalize:  # Optional: normalize the data to be between 0 and 1
        min: -3.2
        max: 3.2
    mapping:
      - name: gripper
        index: 6
  ```

- **PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS**: Target gripper open amounts issued to the robot
  
  ```yaml
  PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS:
    source: action.gripper_target
    format:
      invert_gripper_amount: true # Convert from close amount to open amount
      normalize: # Optional: normalize the data to be between 0 and 1
        min: -3.2
        max: 3.2
    mapping:
      - name: gripper_target
        index: 7
  ```

- **END_EFFECTOR_POSES**: End-effector poses with multiple representation formats
  
  ```yaml
  # Example 1: Position + Euler angles
  END_EFFECTOR_POSES:
    source: observation.state
    format:
      pose_type: POSITION_ORIENTATION
      orientation:
        type: EULER
        euler_order: XYZ  # XYZ | ZYX | YXZ | XZY | YZX | ZXY
        angle_units: RADIANS
    mapping:
      - name: end_effector_pose
        index_range:
          start: 0
          end: 6  # 3 position + 3 euler angles
  
  # Example 2: Position + Quaternion
  END_EFFECTOR_POSES:
    source: observation.pose
    format:
      pose_type: POSITION_ORIENTATION
      orientation:
        type: QUATERNION
        quaternion_order: XYZW  # XYZW | WXYZ
        angle_units: RADIANS
    mapping:
      - name: end_effector_pose
        index_range:
          start: 0
          end: 7  # 3 position + 4 quaternion
  
  # Example 3: 4×4 transformation matrix
  END_EFFECTOR_POSES:
    source: observation.transform
    format:
      pose_type: MATRIX
    mapping:
      - name: end_effector_pose
        index_range:
          start: 0
          end: 16  # 4×4 matrix flattened
  ```

- **POSES**: General 6D poses with multiple representation formats (same options as end-effector poses)
  
  ```yaml
  # Example: Position + Axis-angle representation
  POSES:
    source: observation.object_pose
    format:
      pose_type: POSITION_ORIENTATION
      orientation:
        type: AXIS_ANGLE
        angle_units: RADIANS
    mapping:
      - name: object_pose
        index_range:
          start: 0
          end: 6  # 3 position + 3 axis-angle
  ```

- **LANGUAGE**: Language instructions (STRING or BYTES format)
  
  ```yaml
  # Example 1: Single instruction
  LANGUAGE:
    source: language_instruction
    format:
      language_type: STRING
    mapping:
      - name: instruction
  
  # Example 2: Multiple instructions with different source
  LANGUAGE:
    format:
      language_type: BYTES
    mapping:
      - name: instruction # name in neuracore
        source_name: language_instruction # name in input dataset
      - name: instruction2
        source_name: language_instruction_2
      - name: instruction3
        source_name: language_instruction_3
  ```

- **CUSTOM_1D**: Custom 1D data arrays
  
  ```yaml
  CUSTOM_1D:
    source: observation.state
    mapping:
      - name: custom_1d
        index_range:
          start: 0
          end: 10
      - name: sensor_readings
        index_range:
          start: 10
          end: 20
  ```


#### Advanced Features
This section describes advanced importer configuration options for special data types and transformations.

**Pose Formats**

Poses can be represented in multiple formats:

- **MATRIX**: 4×4 transformation matrix (16 elements flattened)
- **POSITION_ORIENTATION**: Position (3D) + Orientation
  - Orientation types: `QUATERNION` (7 elements), `EULER` (6 elements), `MATRIX` (9 elements), `AXIS_ANGLE` (6 elements)
  - Quaternion order: `XYZW` or `WXYZ`
  - Euler order: `XYZ`, `ZYX`, `YXZ`, `XZY`, `YZX`, `ZXY`
  - Angle units: `RADIANS` or `DEGREES`
  
**Inverse Kinematics (IK) for Joint Positions**

When `joint_position_type: TCP` is specified, the importer uses inverse kinematics to convert end-effector poses to joint positions. This requires:

- A valid URDF or MJCF file with the robot model
- An end-effector frame name (obtained from mapping `name`)
- Optional joint configuration to initiate inverse kinematics (`ik_init_config`)

```yaml
JOINT_POSITIONS:
  source: observation.pose
  format:
    joint_position_type: TCP  # Use IK to convert pose to joint positions
    ik_init_config: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial joint config
    pose_type: POSITION_ORIENTATION
    orientation:
      type: QUATERNION
      quaternion_order: XYZW
      angle_units: RADIANS
  mapping:
    - name: end_effector
      index_range:
        start: 0
        end: 7
```


**Visual Joint Positions from Gripper**

When `visual_joint_type: GRIPPER` is specified, the importer automatically converts gripper open amounts to visual joint positions. This is useful for populating visual joint positions that are used for URDF visualization but not for training. This requires:

- A valid URDF or MJCF file with the robot model
- Joint names in the mapping that exist in the robot model
- Joint limits defined in the robot model (for automatic unnormalization)
- Gripper open amounts data (typically from `PARALLEL_GRIPPER_OPEN_AMOUNTS`)

```yaml
VISUAL_JOINT_POSITIONS:
  source: observation.state
  format:
    visual_joint_type: GRIPPER
  mapping:
    - name: finger_joint1
      index: 6
    - name: finger_joint2
      index: 6
```

## Example Workflow

1. **Prepare your dataset**: Ensure your dataset is in RLDS or LeRobot format

2. **Create configuration file**: Define data mappings in a YAML file
   - Start with `config/example.yaml` as a template
   - Specify the `input_dataset_name` matching your dataset
   - Map each data type with appropriate source paths and formats

3. **Prepare robot description**: Have URDF or MJCF files ready
   - Place robot description files in a directory
   - Specify the path in config or use `--robot-dir`
   - The importer will search for `.urdf`, `.xml`, or `.mjcf` files

4. **Run importer**: Execute the CLI command with appropriate arguments
   ```bash
   neuracore importer import \
       --dataset-config config.yaml \
       --dataset-dir /path/to/dataset \
       --robot-dir /path/to/robot \
       --overwrite
   ```

5. **Monitor progress**: 
   - Logs indicate worker status and any errors
   - Validation warnings appear for data that doesn't match expected formats

6. **Verify import**: Check the dataset in Neuracore to ensure all data was imported correctly

## Troubleshooting

### Common Issues

**Dataset type not detected**
- Ensure your dataset directory contains the expected marker files
- Manually specify `dataset_type` in the configuration file

**Robot description not found**
- Check that URDF/MJCF files are in the specified `--robot-dir`
- Verify file extensions are `.urdf`, `.xml`, or `.mjcf`
- Check that paths in config file are correct

**Joint validation errors**
- Ensure joint names in config match the robot model
- Check that joint limits are defined in the robot model
- Verify joint position data is within limits

**IK convergence failures**
- Ensure URDF is valid and complete
- Try different `ik_init_config` values
- Check that end-effector frame name is correct
