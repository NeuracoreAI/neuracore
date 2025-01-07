import numpy as np
from pyquaternion import Quaternion


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]["t"] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(
            self.curr_left_waypoint, next_left_waypoint, self.step_count
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]
        box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 100,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
            },  # approach meet position
            {
                "t": 260,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            },  # move left
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            },  # stay
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 90,
                "xyz": box_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 130,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 170,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 200,
                "xyz": meet_xyz + np.array([0.05, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 220,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # open gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # move to right
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # stay
        ]
