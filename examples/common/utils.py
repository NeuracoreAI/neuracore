import numpy as np


def generate_random_point_cloud(num_points=2500, bounds=(-1, 1)):
    """Generate random 3D points within a cube (default [-1,1] for x,y,z)."""
    return np.random.uniform(bounds[0], bounds[1], size=(num_points, 3)).astype(
        np.float32
    )


def generate_random_rgb(num_points=2500):
    """Generate random RGB colours for each point."""
    return np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)


def get_camera_matrices():
    """Return dummy identity intrinsics and extrinsics for logging compatibility."""
    intrinsics = np.eye(3, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)
    return intrinsics, extrinsics
