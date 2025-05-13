# Neuracore Python Client

Neuracore is a powerful robotics and machine learning client library for seamless robot data collection, model deployment, and interaction.

## Features

- Easy robot initialization and connection
- Streaming data logging
- Model endpoint management
- Local and remote model support
- Flexible dataset creation

## Installation

```bash
pip install neuracore
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

## Documentation

 - [Limitations](./docs/limitations.md)

## Development

To set up for development:

```bash
git clone https://github.com/neuraco/neuracore
cd neuracore
pip install -e .[dev]
```

## Testing

```bash
export NEURACORE_API_URL=http://localhost:8000/api
pytest tests/
```

## Contributing

Contributions are welcome!
