"""Import NYU ROT dataset directly into Neuracore from original pkl files.

Data format (from transform_pickle.py + demo_class.py):
  [0] images:  (n_eps, n_steps, 3, 84, 84) uint8
  [1] states:  (n_eps, n_steps, 3) float32  — EE pos (x,y,z) in decimeters
               (n_eps, n_steps, 4) for Pour — EE pos + joint7 servo angle (degrees)
  [2] actions: (n_eps, n_steps, 3) float32  — delta EE pos / 0.25, range [-1,1]
               (n_eps, n_steps, 4) for Pour — delta EE pos / 0.25 + delta joint7 / 40
  [3] rewards: (n_eps, n_steps) float32     — 0 during task, 1 at success/end

EE orientation is FIXED per task class (pitch/roll set at robot init, never changes).
xArm units: positions in decimeters (0.1 m). Angles in degrees.
Pinocchio expects SI units (meters, radians).
"""

import json
import os
import pickle
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import neuracore as nc
from neuracore.importer.core.robot_utils import RobotUtils

EXPERT_DEMOS_ZIP = Path.home() / "Downloads" / "osfstorage-archive.zip"
EXTRACT_DIR = Path("/tmp/nyu_rot_extract")
URDF_PATH = Path.home() / "xarm_ros2" / "xarm7.urdf"
FPS = 5
ACTION_SCALE_POS = 0.25  # stored_action * this = delta EE pos in decimeters
ACTION_SCALE_ROT = 40.0  # stored_action[3] * this = delta joint7 in degrees
EE_FRAME = "link_eef"
GRIPPER_NAME = "gripper"
GRIPPER_OPEN_RAD = 0.0  # drive_joint lower limit = open
GRIPPER_CLOSED_RAD = 0.85  # drive_joint upper limit = closed
GRIPPER_JOINT_NAMES = {
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
}


@dataclass
class TaskMeta:
    instruction: str
    roll: float  # degrees, xArm extrinsic RPY
    pitch: float
    yaw: float
    gripper_closed: bool  # constant throughout episode
    has_wrist: bool = False  # Pour: joint7 explicitly recorded


# Orientation from demo_class.py defaults per class:
#   Reach/BaseClass default: pitch=0, roll=180
#   HorizontalReach:         pitch=-90, roll=180
#   SideReach:               pitch=0, roll=90
#   Pour:                    pitch=90, roll=-90
TASK_META: dict[str, TaskMeta] = {
    "RobotEraseBoard-v1/expert_demos": TaskMeta(
        "erase the board", 180, 0, 0, gripper_closed=True
    ),
    "RobotHangHanger-v1/expert_demos": TaskMeta(
        "hang the hanger on the rod", 180, -90, 0, gripper_closed=True
    ),
    "RobotReach-v1/expert_demos": TaskMeta(
        "reach the blue mark on the table", 180, 0, 0, gripper_closed=False
    ),
    "RobotDoorClose-v1/expert_demos": TaskMeta(
        "close the door", 180, 0, 0, gripper_closed=True
    ),
    "RobotCupStacking-v1/expert_demos": TaskMeta(
        "stack the cups", 180, 0, 0, gripper_closed=True
    ),
    "RobotTurnKnob-v1/expert_demos": TaskMeta(
        "turn the knob", 180, 0, 0, gripper_closed=True
    ),
    "RobotInsertPeg-v1/expert_demos_easy": TaskMeta(
        "insert the peg in the cup (easy)", 180, 0, 0, gripper_closed=True
    ),
    "RobotInsertPeg-v1/expert_demos_medium": TaskMeta(
        "insert the peg in the cup (medium)", 180, 0, 0, gripper_closed=True
    ),
    "RobotInsertPeg-v1/expert_demos_hard": TaskMeta(
        "insert the peg in the cup (hard)", 180, 0, 0, gripper_closed=True
    ),
    "RobotButtonPress-v1/expert_demos": TaskMeta(
        "press the button", 180, 0, 0, gripper_closed=True
    ),
    "RobotHangBag-v1/expert_demos": TaskMeta(
        "hang the bag on the hook", 180, -90, 0, gripper_closed=True
    ),
    "RobotBoxOpen-v1/expert_demos": TaskMeta(
        "open the box", 90, 0, 0, gripper_closed=False
    ),
    "RobotPour-v1/expert_demos": TaskMeta(
        "pour the almonds into the cup", -90, 90, 0, gripper_closed=True, has_wrist=True
    ),
    "RobotHangMug-v1/expert_demos": TaskMeta(
        "hang the mug on the hook", 180, -90, 0, gripper_closed=True
    ),
}


def rpy_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """xArm extrinsic RPY (degrees) → quaternion [qx, qy, qz, qw]."""
    return R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()


def build_ee_pose(pos_dm: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Combine EE position (decimeters) + quaternion into 7-element pose array."""
    return np.array([*pos_dm, *quat_xyzw], dtype=np.float64)


def extract_data() -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    demos_zip = EXTRACT_DIR / "expert_demos.zip"
    if not demos_zip.exists():
        print("Extracting expert_demos.zip from archive...")
        with zipfile.ZipFile(EXPERT_DEMOS_ZIP) as zf:
            zf.extract("expert_demos.zip", EXTRACT_DIR)
    robotgym_dir = EXTRACT_DIR / "expert_demos" / "robotgym"
    if not robotgym_dir.exists():
        print("Extracting robotgym pkl files...")
        with zipfile.ZipFile(demos_zip) as zf:
            for name in zf.namelist():
                if name.startswith("expert_demos/robotgym/"):
                    zf.extract(name, EXTRACT_DIR)
    return robotgym_dir


def collect_episodes(robotgym_dir: Path):
    """Yield (key, meta, images, states, actions, rewards) per episode."""
    for task_dir in sorted(robotgym_dir.iterdir()):
        for pkl_path in sorted(task_dir.iterdir()):
            key = f"{task_dir.name}/{pkl_path.stem}"
            meta = TASK_META.get(key)
            if meta is None:
                print(f"  Skipping unknown key: {key}")
                continue
            with open(pkl_path, "rb") as f:
                images, states, actions, rewards = pickle.load(f)
            for ep_idx in range(images.shape[0]):
                yield key, meta, images[ep_idx], states[ep_idx], actions[
                    ep_idx
                ], rewards[ep_idx]


def _restart_daemon() -> None:
    daemon_cmd = [sys.executable, "-m", "neuracore.data_daemon"]
    subprocess.run([*daemon_cmd, "stop"], check=False)
    time.sleep(2)
    subprocess.run([*daemon_cmd, "launch", "--background"], check=True)
    time.sleep(2)
    print("Daemon restarted.")


def main():
    robotgym_dir = extract_data()
    robot_utils = RobotUtils(str(URDF_PATH), os.path.dirname(str(URDF_PATH)))

    _restart_daemon()
    nc.login()
    nc.connect_robot(
        robot_name="Ufactory xArm 7",
        urdf_path=str(URDF_PATH),
        overwrite=True,
        shared=True,
    )

    from neuracore.core.data.dataset import Dataset

    existing = Dataset.get_by_name("NYU ROT", non_exist_ok=True)
    if existing is not None:
        print(f"Deleting existing dataset '{existing.id}' before recreating...")
        existing.delete()

    nc.create_dataset(
        name="NYU ROT",
        tags=[
            "xarm",
            "manipulation",
            "pick-and-place",
            "rot",
            "nyu",
            "imitation-learning",
        ],
        description=(
            "NYU ROT (Regularized Optimal Transport) dataset collected on a "
            "Ufactory xArm 7 robot. 14 episodes across 12 manipulation tasks. "
            "From 'Watch and Match: Supercharging Imitation with Regularized "
            "Optimal Transport' (Haldar et al., 2023)."
        ),
        shared=True,
    )

    episodes = list(collect_episodes(robotgym_dir))
    print(f"Found {len(episodes)} episodes")

    gripper_annotations_path = Path(__file__).parent / "gripper_states.json"
    gripper_annotations: dict[str, bool] = {}
    if gripper_annotations_path.exists():
        with open(gripper_annotations_path) as f:
            gripper_annotations = json.load(f)
        print(f"Using gripper annotations from {gripper_annotations_path}")

    gripper_open = 1.0
    gripper_closed = 0.0

    for ep_idx, (key, meta, images, states, actions, rewards) in enumerate(episodes):
        n_steps = images.shape[0]
        print(
            f"Episode {ep_idx + 1}/{len(episodes)}: {meta.instruction} "
            f"({n_steps} steps)"
        )

        quat = rpy_to_quat(meta.roll, meta.pitch, meta.yaw)
        is_closed = gripper_annotations.get(key, meta.gripper_closed)
        gripper_val = gripper_closed if is_closed else gripper_open
        gripper_rad = GRIPPER_CLOSED_RAD if is_closed else GRIPPER_OPEN_RAD
        gripper_joints = {
            "drive_joint": gripper_rad,
            "left_finger_joint": gripper_rad,
            "left_inner_knuckle_joint": gripper_rad,
            "right_outer_knuckle_joint": gripper_rad,
            "right_finger_joint": gripper_rad,
            "right_inner_knuckle_joint": gripper_rad,
        }

        nc.start_recording()
        t = time.time()
        dt = 1.0 / FPS
        prev_ik = None
        prev_target_ik = None

        for step in range(n_steps):
            pos_m = states[step, :3] / 10.0  # decimeters → meters

            # IK for current EE pose
            try:
                joint_pos = robot_utils.end_effector_to_joint_positions(
                    build_ee_pose(pos_m, quat), EE_FRAME, prev_ik
                )
                prev_ik = list(joint_pos.values())
                arm_joints = {
                    k: v for k, v in joint_pos.items() if k not in GRIPPER_JOINT_NAMES
                }
                nc.log_joint_positions(arm_joints, timestamp=t)
                nc.log_visual_joint_positions(arm_joints, timestamp=t)
            except ValueError:
                print(f"    IK failed at step {step}, skipping joint positions")

            # IK for target EE pose
            target_pos_m = pos_m + actions[step, :3] * ACTION_SCALE_POS / 10.0
            try:
                target_joint_pos = robot_utils.end_effector_to_joint_positions(
                    build_ee_pose(target_pos_m, quat), EE_FRAME, prev_target_ik
                )
                prev_target_ik = list(target_joint_pos.values())
                arm_target_joints = {
                    k: v
                    for k, v in target_joint_pos.items()
                    if k not in GRIPPER_JOINT_NAMES
                }
                nc.log_joint_target_positions(arm_target_joints, timestamp=t)
            except ValueError:
                print(f"    Target IK failed at step {step}, skipping")

            # EE pose (meters)
            nc.log_end_effector_pose(EE_FRAME, build_ee_pose(pos_m, quat), timestamp=t)

            # Target EE pose
            nc.log_pose("ee_target", build_ee_pose(target_pos_m, quat), timestamp=t)

            # Gripper (binary constant per episode)
            nc.log_parallel_gripper_open_amount(GRIPPER_NAME, gripper_val, timestamp=t)
            nc.log_parallel_gripper_target_open_amount(
                GRIPPER_NAME, gripper_val, timestamp=t
            )
            nc.log_joint_positions(gripper_joints, timestamp=t)
            nc.log_visual_joint_positions(gripper_joints, timestamp=t)
            nc.log_joint_target_positions(gripper_joints, timestamp=t)

            # Pour: wrist servo angle (joint7) in radians — overrides IK joint7
            if meta.has_wrist:
                j7_rad = float(np.deg2rad(states[step, 3]))
                target_j7 = j7_rad + float(
                    np.deg2rad(actions[step, 3] * ACTION_SCALE_ROT)
                )
                nc.log_joint_positions({"joint7": j7_rad}, timestamp=t)
                nc.log_visual_joint_positions({"joint7": j7_rad}, timestamp=t)
                nc.log_joint_target_positions({"joint7": target_j7}, timestamp=t)

            nc.log_rgb("image", np.transpose(images[step], (1, 2, 0)), timestamp=t)
            nc.log_language("instruction", meta.instruction, timestamp=t)

            t += dt

        nc.stop_recording()
        print("  Done")

    print("All episodes imported.")


if __name__ == "__main__":
    main()
