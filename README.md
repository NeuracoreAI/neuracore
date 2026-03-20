<div align="center">
  <img src="./docs/assets/neuracore_logo.jpg" alt="Neuracore Logo" width="100%">
</div>

<div align="center">

[![Downloads](https://static.pepy.tech/badge/neuracore)](https://pepy.tech/project/neuracore)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/neuracore)](https://pypi.org/project/neuracore/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/NeuracoreAI/neuracore)](https://github.com/NeuracoreAI/neuracore/commits/main)

</div>

<p align="center">
  Join our community!
</p>
<p align="center">
  <a target="_blank" href="https://discord.gg/DF5m8V6nbD"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord" alt="Discord" /></a>
</p>

<p align="center">
  Getting started
</p>
<p align="center">
  <a target="_blank" href="https://www.neuracore.com/try-on-colab"><img src="https://img.shields.io/badge/Try%20on%20Google%20Colab-303030?style=for-the-badge&logo=googlecolab" alt="Discord" /></a>
</p>

<div align="center">

</br> 


</div>

# 🤖 What is Neuracore
✨ **Neuracore** is a powerful robot learning library that enables **data collection and visualization, model training, deployment, and real-time inference with support for custom data types.** Get started with Neuracore today, [sign up for a Neuracore account](https://www.neuracore.com/)!
<div align="center" style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
<img src="https://github.com/user-attachments/assets/5f6e9f12-185f-4050-9e71-2e5712d49b03" alt="Data Visualization" width="150%">
</div>

# 🌟 Features

- 🚀 **Streaming data logging** with custom data types
- 📊 **Dataset visualization** and synchronization
- ☁️ Train **robot learning algorithms on cloud**
- 🤖 **Policy inference** and deployment


# 🛠️ Installation
To install the basic package for data logging and visualization:

```bash
pip install neuracore
```

**Note:** installing the `ffmpeg` binary is recommended for faster video encoding (during recording) and decoding (during playback/import). If not available, Neuracore falls back to PyAV for encoding.

Linux (Debian/Ubuntu):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

For training and ML development:
```bash
pip install neuracore[ml]
```

For bulk importing datasets:
```bash
pip install neuracore[import]
```

To run our examples:
```bash
pip install neuracore[examples]
```

# 🍰 A Short Taste
Here is a short taste on what neuracore can do.
For a detailed walk-through, please refer to the [tutorial](./docs/tutorial.md) and [documentation](#-documentation), or [try it yourself on Google Colab](https://www.neuracore.com/try-on-colab).
```python
import neuracore as nc # pip install neuracore
import time

# ensure you have an account at neuracore.com
nc.login()

# Connect to a robot with URDF
nc.connect_robot(
    robot_name="MyRobot", 
    urdf_path="/path/to/robot.urdf",
)

# Create a dataset for recording
nc.create_dataset(
    name="My Robot Dataset",
    description="Example dataset with multiple data types"
)

# Recording and streaming data
nc.start_recording()
t = time.time()
nc.log_joint_positions(positions={'joint1': 0.5, 'joint2': -0.3}, timestamp=t)
nc.log_rgb(name="top_camera", rgb=image_array, timestamp=t)
# Stop recording, the dataset is automatically uploaded to the cloud
nc.stop_recording()

# Kick off cloud training
job_data = nc.start_training_run(
    name="MyTrainingJob",
    dataset_name="My Robot Dataset",
    algorithm_name="diffusion_policy",
    num_gpus=5,
    frequency=50,
    ...
)

# Load a trained model locally
policy = nc.policy(
    train_run_name="MyTrainingJob",
    ...
)

# Get model inputs
nc.log_joint_positions(positions={'joint1': 0.5, 'joint2': -0.3})
nc.log_rgb(name="top_camera", rgb=image_array)
# Model Inference
predictions = policy.predict(timeout=5)
```

# 📚 Documentation
- [Examples](./examples/README.md)
- [Tutorial](./docs/tutorial.md)
- [Training](./docs/training.md)
- [Command Line Tools](./docs/commandline.md)
- [Dataset Importer](./docs/dataset_importer.md)
- [Environment Variables](./docs/environment_variable.md)
- [Contribution Guide](./docs/contribution_guide.md)
- [Data Daemon](./docs/data_daemon.md)

# 💬 Community

We are building Neuracore to help everyone accelerate their robot learning workflows, and we'd love to hear from you! Join our community to get help, share ideas, and stay updated:

- [Discord](https://discord.gg/DF5m8V6nbD) - Chat with the community and get support
- [GitHub Issues](https://github.com/NeuracoreAI/neuracore/issues) - Report bugs and request features

# 🧾 Citation

If you use Neuracore in your research, please consider citing:

```bibtex
@software{Neuracore,
  author = {Neuracore Team},
  title = {Neuracore},
  month = {January},
  year = {2026},
  url = {https://github.com/NeuracoreAI/neuracore}
}
