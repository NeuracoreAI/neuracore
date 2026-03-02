

## Tutorial

In the tutorial, we provide code examples on our key features.

Before start, please ensure you have an account at [neuracore.com](https://www.neuracore.com/)

### Connect to a Robot

```python
import neuracore as nc

# Remember to login first
# This will save your API key locally
nc.login()

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

#### Update Robot Name
```python
    nc.connect_robot(
        robot_name=old_robot_name,
        urdf_path="/path/to/robot.urdf",
        overwrite=False,
    )

    nc.update_robot_name(
        robot_key=old_robot_name,
        new_robot_name=new_robot_name,
        shared=False,
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
nc.log_joint_positions(positions={'joint1': 0.5, 'joint2': -0.3}, timestamp=t)
nc.log_joint_velocities(velocities={'joint1': 0.1, 'joint2': -0.05}, timestamp=t)
nc.log_joint_target_positions(target_positions={'joint1': 0.6, 'joint2': -0.2}, timestamp=t)

# Log camera data
nc.log_rgb(name="top_camera", rgb=image_array, timestamp=t)

# Log language instructions
nc.log_language(
    name="instruction",
    language="Pick up the red cube",
    timestamp=t,
)

# Log custom data
custom_sensor_data = np.array([1.2, 3.4, 5.6])
nc.log_custom_1d("force_sensor", custom_sensor_data, timestamp=t)

# Stop recording
nc.stop_recording()
```

#### Live Data Streaming Control

```python
# Stop live data streaming but still keep recording (saves bandwidth)
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

data_types_to_synchronize = [
    DataType.JOINT_POSITIONS, 
    DataType.RGB_IMAGES,
    DataType.LANGUAGES]

robot_data_spec: RobotDataSpec = {}
robot_ids_dataset = dataset.robot_ids
for robot_id in robot_ids_dataset:
    data_type_to_names = dataset.get_full_data_spec(robot_id)
    robot_data_spec[robot_id] = {
        data_type: data_type_to_names[data_type]
        for data_type in data_types_to_synchronize
    }

synced_dataset = dataset.synchronize(
    frequency=1,
    robot_data_spec=robot_data_spec,
)

print(f"Dataset has {len(synced_dataset)} episodes")

# Access synchronized data
for episode in synced_dataset[:5]:  # First 5 episodes
    for step in episode:
        step = cast(SynchronizedPoint, step)
        joint_pos = step[DataType.JOINT_POSITIONS]
        rgb_images = step[DataType.RGB_IMAGES]
        language = step[DataType.LANGUAGES]
        # Process your data
```

### Model Inference
The model inference can be done either on the local computer or remotely on the cloud.
#### Local Model Inference

```python
from typing import cast
import torch
from neuracore_types import BatchedJointData, DataSpec, DataType

# Specification of the order that will be fed into the model
MODEL_INPUT_ORDER: DataSpec = {
    DataType.JOINT_POSITIONS: ['joint1', 'joint2'],
    DataType.RGB_IMAGES: ["top_camera"],
}

MODEL_OUTPUT_ORDER: DataSpec = {
    DataType.JOINT_TARGET_POSITIONS: ['joint1', 'joint2'],
}
# Load a trained model locally
policy = nc.policy(
    train_run_name="MyTrainingJob",
    model_input_order=MODEL_INPUT_ORDER,
    model_output_order=MODEL_OUTPUT_ORDER,
)

# Or load from file path
# policy = nc.policy(model_file="/path/to/model.nc.zip")

# Set specific checkpoint indexed by epochs(optional, defaults to last epoch)
policy.set_checkpoint(epoch=-1)
# Log input data
nc.log_joint_positions(positions={'joint1': 0.5, 'joint2': -0.3}, timestamp=t)
nc.log_rgb(name="top_camera", rgb=image_array, timestamp=t)

# Predict actions
predictions = policy.predict(timeout=5)
joint_target_positions = cast(
    dict[str, BatchedJointData],
    predictions[DataType.JOINT_TARGET_POSITIONS],
)

# Concatenate joint targets and convert to numpy
joint_names = MODEL_OUTPUT_ORDER[DataType.JOINT_TARGET_POSITIONS]
batched_action = (
    torch.cat(
        [joint_target_positions[name].value for name in joint_names],
        dim=2,
    )
    .cpu()
    .numpy()
)

# Get first batch: (horizon, num_joints)
actions = batched_action[0]
```

#### Remote Model Inference

```python
# Connect to a remote endpoint on the cloud
try:
    policy = nc.policy_remote_server("MyEndpointName")
    predictions = policy.predict(timeout=5)
    # Process predictions...
except nc.EndpointError:
    print("Endpoint not available. Please start it at neuracore.com/dashboard/endpoints")
```

#### Local Server Deployment

```python
# Connect to a local policy server
policy = nc.policy_local_server(
    train_run_name="MyTrainingJob",
    model_input_order=MODEL_INPUT_ORDER,
    model_output_order=MODEL_OUTPUT_ORDER)
```