"""Base MuJoCo environment classes with improved organization."""

import mujoco
import numpy as np
from pydantic import BaseModel


class CameraData(BaseModel):
    """Camera observation data structure."""

    rgb: np.ndarray
    depth: np.ndarray | None = None
    point_cloud: np.ndarray | None = None

    class Config:
        arbitrary_types_allowed = True


class Observation(BaseModel):
    """Complete observation structure for bimanual robot tasks."""

    qpos: dict[str, float]
    qvel: dict[str, float]
    env_state: np.ndarray
    cameras: dict[str, CameraData]
    mocap_pose_left: np.ndarray | None = None
    mocap_pose_right: np.ndarray | None = None
    gripper_ctrl: np.ndarray | None = None
    reward: float | None = None
    end_effector_poses: dict[str, list[float]] | None = None
    gripper_open_amounts: dict[str, float] | None = None

    class Config:
        arbitrary_types_allowed = True


class MuJoCoEnvironment:
    """Base MuJoCo environment class with common functionality."""

    # Camera constants
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480

    # Depth constants
    MIN_DEPTH_M: float = 0.0
    MAX_DEPTH_M: float = 2.0

    def __init__(
        self, model_path: str, time_limit: float = 20.0, dt: float = 0.02
    ) -> None:
        """Initialize the MuJoCo environment.

        Args:
            model_path: Path to the MuJoCo XML model file.
            time_limit: Maximum episode duration in seconds.
            dt: Physics timestep in seconds.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.time_limit = time_limit
        self.dt = dt
        self.step_count = 0

        # Create renderer for visualization
        self.renderer = mujoco.Renderer(
            self.model, width=self.CAMERA_WIDTH, height=self.CAMERA_HEIGHT
        )

    def reset(self) -> Observation:
        """Reset the environment to initial state.

        Returns:
            Initial observation.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.initialize_episode()
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool]:
        """Execute one environment step.

        Args:
            action: Control action to execute.

        Returns:
            Tuple of (observation, reward, done).
        """
        self.before_step(action)
        for _ in range(int(self.dt / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self.get_observation()
        reward = self.get_reward()
        done = self.step_count * self.dt >= self.time_limit

        return obs, reward, done

    def before_step(self, action: np.ndarray) -> None:
        """Process action before stepping physics.

        Args:
            action: Action to process.
        """
        raise NotImplementedError

    def initialize_episode(self) -> None:
        """Initialize episode-specific state."""
        raise NotImplementedError

    def get_observation(self) -> Observation:
        """Get current observation.

        Returns:
            Current environment observation.
        """
        raise NotImplementedError

    def get_reward(self) -> float:
        """Calculate reward for current state.

        Returns:
            Reward value.
        """
        raise NotImplementedError

    def get_camera_extrinsics(self, camera_name: str) -> np.ndarray:
        """Get camera extrinsics transformation matrix.

        Args:
            camera_name: Name of the camera.

        Returns:
            4x4 extrinsics matrix.

        Raises:
            ValueError: If camera name not found.
        """
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found")

        cam_pos = self.data.cam_xpos[camera_id]
        cam_rot = self.data.cam_xmat[camera_id].reshape(3, 3)
        extr = np.eye(4)
        extr[:3, :3] = cam_rot.T
        extr[:3, 3] = cam_pos
        return extr

    def get_camera_intrinsics(self, camera_name: str) -> np.ndarray:
        """Get camera intrinsics matrix.

        Args:
            camera_name: Name of the camera.

        Returns:
            3x3 intrinsics matrix.

        Raises:
            ValueError: If camera name not found.
        """
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found")

        fov = self.model.cam_fovy[camera_id]
        theta = np.deg2rad(fov)
        fx = self.CAMERA_WIDTH / 2 / np.tan(theta / 2)
        fy = self.CAMERA_HEIGHT / 2 / np.tan(theta / 2)
        cx = (self.CAMERA_WIDTH - 1) / 2.0
        cy = (self.CAMERA_HEIGHT - 1) / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def rgbd_to_pointcloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intr: np.ndarray,
        extr: np.ndarray,
        depth_trunc: float = 2.0,
    ) -> np.ndarray:
        """Convert RGB-D images to point cloud.

        Args:
            rgb: RGB image array.
            depth: Depth image array.
            intr: Camera intrinsics matrix.
            extr: Camera extrinsics matrix.
            depth_trunc: Maximum depth threshold.

        Returns:
            Point cloud as Nx6 array (xyz + rgb).
        """
        cc, rr = np.meshgrid(
            np.arange(self.CAMERA_WIDTH), np.arange(self.CAMERA_HEIGHT), sparse=True
        )
        valid = (depth > 0) & (depth < depth_trunc)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
        y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
        xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
        color = rgb.transpose([2, 0, 1]).reshape((3, -1)).T / 255.0
        mask = np.isnan(xyz[:, 2])
        xyz = xyz[~mask]
        color = color[~mask]
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_t = (extr @ xyz_h.T).T
        xyzrgb = np.hstack([xyz_t[:, :3], color])
        return xyzrgb

    def render(self, camera_name: str, depth: bool = False) -> np.ndarray:
        """Render from specified camera.

        Args:
            camera_name: Name of camera to render from.
            depth: Whether to render depth image.

        Returns:
            Rendered image array.

        Raises:
            ValueError: If camera name not found.
        """
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found")

        self.renderer.update_scene(self.data, camera_id)

        if depth:
            self.renderer.enable_depth_rendering()
        img = self.renderer.render().copy()
        if depth:
            self.renderer.disable_depth_rendering()
        return img


class BimanualViperXTask(MuJoCoEnvironment):
    """Base class for bimanual ViperX robot tasks."""

    # Robot constants
    EPISODE_LENGTH: int = 400
    START_ARM_POSE = np.array([
        0,
        -1.73,
        1.17,
        0,
        0.56,
        0,
        0.02,
        -0.02,
        0,
        -1.73,
        1.17,
        0,
        0.56,
        0.0,
        0.02,
        -0.02,
    ])

    # Gripper constants
    GRIPPER_POSITION_OPEN: float = 0.05800
    GRIPPER_POSITION_CLOSE: float = 0.01844

    # Joint names
    LEFT_ARM_JOINT_NAMES = [
        "vx300s_left/waist",
        "vx300s_left/shoulder",
        "vx300s_left/elbow",
        "vx300s_left/forearm_roll",
        "vx300s_left/wrist_angle",
        "vx300s_left/wrist_rotate",
    ]
    LEFT_GRIPPER_JOINT_NAMES = ["vx300s_left/left_finger", "vx300s_left/right_finger"]
    RIGHT_ARM_JOINT_NAMES = [
        "vx300s_right/waist",
        "vx300s_right/shoulder",
        "vx300s_right/elbow",
        "vx300s_right/forearm_roll",
        "vx300s_right/wrist_angle",
        "vx300s_right/wrist_rotate",
    ]
    RIGHT_GRIPPER_JOINT_NAMES = [
        "vx300s_right/left_finger",
        "vx300s_right/right_finger",
    ]
    LEFT_GRIPPER_OPEN = "vx300s_left/gripper_open"
    RIGHT_GRIPPER_OPEN = "vx300s_right/gripper_open"
    ACTION_KEYS = (
        LEFT_ARM_JOINT_NAMES
        + [LEFT_GRIPPER_OPEN]
        + RIGHT_ARM_JOINT_NAMES
        + [RIGHT_GRIPPER_OPEN]
    )

    def __init__(
        self, model_path: str, random: np.random.Generator | None = None
    ) -> None:
        """Initialize bimanual ViperX task.

        Args:
            model_path: Path to MuJoCo XML model.
            random: Optional random number generator.
        """
        super().__init__(model_path)
        self.random = random or np.random.default_rng()

    @classmethod
    def normalize_gripper_position(cls, x: float) -> float:
        """Normalize gripper position to [0, 1] range.

        Args:
            x: Raw gripper position.

        Returns:
            Normalized gripper position.
        """
        return np.clip(
            (x - cls.GRIPPER_POSITION_CLOSE)
            / (cls.GRIPPER_POSITION_OPEN - cls.GRIPPER_POSITION_CLOSE),
            0.0,
            1.0,
        )

    @classmethod
    def unnormalize_gripper_position(cls, x: float) -> float:
        """Unnormalize gripper position from [0, 1] range.

        Args:
            x: Normalized gripper position.

        Returns:
            Raw gripper position.
        """
        return (
            x * (cls.GRIPPER_POSITION_OPEN - cls.GRIPPER_POSITION_CLOSE)
            + cls.GRIPPER_POSITION_CLOSE
        )

    def get_qpos(self) -> dict[str, float]:
        """Get joint positions as dictionary.

        Returns:
            Dictionary mapping joint names to positions.
        """
        qpos_raw = self.data.qpos.copy()
        joint_dict = {}
        for i in range(16):  # First 16 joints are robot joints
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_dict[joint_name] = qpos_raw[i]
        return joint_dict

    def get_qvel(self) -> dict[str, float]:
        """Get joint velocities as dictionary.

        Returns:
            Dictionary mapping joint names to velocities.
        """
        qvel_raw = self.data.qvel.copy()
        joint_dict = {}
        for i in range(16):  # First 16 joints are robot joints
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_dict[joint_name] = qvel_raw[i]
        return joint_dict

    def get_env_state(self) -> np.ndarray:
        """Get environment-specific state.

        Returns:
            Environment state array.
        """
        raise NotImplementedError

    def _get_end_effector_pose(self, effector_name: str) -> list[float] | None:
        """Get end effector pose from MuJoCo body positions and orientations.

        Args:
            effector_name: Name of the end effector.

        Returns:
            End effector pose as 7-element list [x, y, z, qx, qy, qz, qw].
        """
        effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, effector_name
        )
        if effector_id == -1:
            return None
        x, y, z = self.data.xpos[effector_id].copy()
        qw, qx, qy, qz = self.data.xquat[effector_id].copy()
        return [x, y, z, qx, qy, qz, qw]

    def get_end_effector_poses(self) -> dict[str, list[float]]:
        """Get end effector poses from MuJoCo body positions and orientations.

        Returns:
            Dictionary mapping end effector names to
            7-element pose lists [x, y, z, qx, qy, qz, qw].
        """
        poses = {}

        left_ee_pose = self._get_end_effector_pose("vx300s_left/gripper_link")
        if left_ee_pose is not None:
            poses["left_ee"] = left_ee_pose

        right_ee_pose = self._get_end_effector_pose("vx300s_right/gripper_link")
        if right_ee_pose is not None:
            poses["right_ee"] = right_ee_pose

        return poses

    def _get_gripper_open_amount(self, gripper_name: str) -> float:
        """Get gripper open amount from gripper finger joint position.

        Args:
            gripper_name: Name of the gripper.

        Returns:
            Normalized gripper open amount [0, 1].
        """
        qpos_dict = self.get_qpos()
        gripper_pos = qpos_dict.get(gripper_name, 0.0)
        return self.normalize_gripper_position(gripper_pos)

    def get_gripper_open_amounts(self) -> dict[str, float]:
        """Get gripper open amounts from gripper finger joint positions.

        Returns:
            Dictionary mapping gripper names to normalized open amounts [0, 1].
        """
        open_amounts = {}
        open_amounts["left_gripper"] = self._get_gripper_open_amount(
            "vx300s_left/left_finger"
        )
        open_amounts["right_gripper"] = self._get_gripper_open_amount(
            "vx300s_right/left_finger"
        )
        return open_amounts

    def get_observation(self) -> Observation:
        """Get complete observation with camera data.

        Returns:
            Structured observation object.
        """
        # Get angle camera data
        angle_rgb = self.render("angle")
        angle_depth = self.render("angle", depth=True)
        angle_pcd = self.rgbd_to_pointcloud(
            angle_rgb,
            angle_depth,
            self.get_camera_intrinsics("angle"),
            self.get_camera_extrinsics("angle"),
        )

        cameras = {
            "angle": CameraData(rgb=angle_rgb, depth=angle_depth, point_cloud=angle_pcd)
        }

        return Observation(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            env_state=self.get_env_state(),
            cameras=cameras,
            end_effector_poses=self.get_end_effector_poses(),
            gripper_open_amounts=self.get_gripper_open_amounts(),
        )

    def sample_box_pose(self) -> np.ndarray:
        """Sample random box pose within workspace.

        Returns:
            Box pose array (position + quaternion).
        """
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.array([x_range, y_range, z_range])
        cube_position = self.random.uniform(ranges[:, 0], ranges[:, 1])
        cube_quat = np.array([1, 0, 0, 0])
        return np.concatenate([cube_position, cube_quat])
