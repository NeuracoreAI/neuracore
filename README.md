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

 - [Creating Custom Algorithms](./docs/creating_custom_algorithms.md)
 - [Limitations](./docs/limitations.md)

## Development

To set up for development:

```bash
git clone https://github.com/neuraco/neuracore
cd neuracore
pip install -e .[dev]
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
