import numpy as np
import mujoco


def get_camera_matrices(model, data, cam_id, height, width):
    """
    Compute camera intrinsics and extrinsics using projection-matrix style math.
    """
    # --- Intrinsics ---
    fovy = model.cam_fovy[cam_id] * np.pi / 180.0
    fy = (height / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- Extrinsics ---
    position = data.cam_xpos[cam_id]
    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = position

    return intrinsics, extrinsics


def _create_uniform_pixel_coords(resolution):
    H, W = resolution
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)


def depth_to_point_cloud(depth, intrinsics, extrinsics):
    """
    Convert depth image to a 3D point cloud in world coordinates.
    """
    H, W = depth.shape
    pixel_coords = _create_uniform_pixel_coords((H, W))

    # Camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    Z = depth
    X = (pixel_coords[..., 0] - cx) * Z / fx
    Y = (pixel_coords[..., 1] - cy) * Z / fy
    points_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # Transform to world coordinates
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (R @ points_cam.T).T + t

    return points_world.astype(np.float32)
