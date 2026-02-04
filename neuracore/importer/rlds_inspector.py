"""Inspector script for RLDS datasets.

This module provides utilities for inspecting and debugging RLDS (Robotics Learning
Dataset Standard) datasets loaded via TensorFlow Datasets.
"""

import tensorflow_datasets as tfds

TARGET_EPISODE = 0
CONTROL_RATE_HZ = 5  # Depends on the dataset!
dataset_path = "PATH_TO_DATASET"


b = tfds.builder_from_directory(builder_dir=dataset_path)
ds = b.as_dataset(split=f"train[{TARGET_EPISODE}:{TARGET_EPISODE + 1}]")
episode = next(iter(ds))
breakpoint()
