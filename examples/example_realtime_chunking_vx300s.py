"""Async overlapping-chunk rollout (RTC or temporal ensemble) on VX300s.

Unlike :mod:`example_local_endpoint_vx300s`, this drives the robot with
:func:`neuracore.policy_realtime`: inference runs in a background thread while
the control loop consumes one action per tick via ``get_action(obs)``.

Requires a trained **DiffusionPolicy** (RTC) or any chunking policy that
exposes ``_predict_action`` (temporal ensemble). Install ML extras first::

    pip install "neuracore[ml,examples]"

Examples::

    python example_realtime_chunking_vx300s.py \\
        --train-run-name MyTrainingJob --mode rtc

    python example_realtime_chunking_vx300s.py \\
        --model-file /path/to/model.nc.zip --mode temporal_ensemble
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from common.base_env import BimanualViperXTask
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, BOX_POSE, make_sim_env
from neuracore_types import DataType, JointData, RGBCameraData, SynchronizedPoint

import neuracore as nc
from neuracore.ml.utils.real_time_chunking import RTCConfig
from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

TRAINING_JOB_NAME = "MyTrainingJob"
CAMERA_NAMES = ["angle"]
NC_CAM_NAME = "angle"
CONTROL_HZ = 50.0
EPISODE_LENGTH = 400
NUM_ROLLOUTS = 5

JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES
    + BimanualViperXTask.LEFT_GRIPPER_JOINT_NAMES
    + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    + BimanualViperXTask.RIGHT_GRIPPER_JOINT_NAMES
)


def _make_sync_point(obs) -> SynchronizedPoint:
    """Build a model observation from a MuJoCo env observation."""
    return SynchronizedPoint(
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=obs.qpos[name]) for name in JOINT_NAMES
            },
            DataType.RGB_IMAGES: {
                NC_CAM_NAME: RGBCameraData(frame=obs.cameras[NC_CAM_NAME].rgb),
            },
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--train-run-name",
        default=TRAINING_JOB_NAME,
        help="Training run name to download the model from (default: %(default)s)",
    )
    source.add_argument(
        "--model-file",
        default=None,
        help="Path to a local model.nc.zip (overrides --train-run-name)",
    )
    parser.add_argument(
        "--mode",
        choices=("rtc", "temporal_ensemble"),
        default="rtc",
        help="Async chunking strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=CONTROL_HZ,
        help="Control-loop rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=16,
        help="Actions executed per chunk before replan (s)",
    )
    parser.add_argument(
        "--inference-delay",
        type=int,
        default=4,
        help="RTC freeze prefix in ticks (d); ignored for temporal_ensemble",
    )
    parser.add_argument(
        "--ensemble-m",
        type=float,
        default=0.01,
        help="Temporal-ensemble decay rate m",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=NUM_ROLLOUTS,
        help="Number of MuJoCo episodes to run",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable on-screen rendering",
    )
    return parser.parse_args()


def main() -> None:
    """Run async chunked rollouts in the Transfer Cube sim."""
    args = _parse_args()
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )

    control_hz = float(args.control_hz)
    if args.mode == "rtc":
        config: RTCConfig | TemporalEnsembleConfig = RTCConfig(
            inference_delay=int(args.inference_delay),
            execution_horizon=int(args.execution_horizon),
        )
        adapt = True
    else:
        config = TemporalEnsembleConfig(
            execution_horizon=int(args.execution_horizon),
            m=float(args.ensemble_m),
        )
        adapt = False

    load_kwargs: dict = {
        "mode": args.mode,
        "config": config,
        "control_hz": control_hz,
        "adapt_inference_delay": adapt,
    }
    if args.model_file is not None:
        print(f"Loading model from {args.model_file} (mode={args.mode})")
        policy = nc.policy_realtime(model_file=args.model_file, **load_kwargs)
    else:
        print(f"Loading train run {args.train_run_name!r} (mode={args.mode})")
        policy = nc.policy_realtime(train_run_name=args.train_run_name, **load_kwargs)

    print(
        f"Realtime policy ready: H={policy.prediction_horizon}, "
        f"mode={policy.mode}, control_hz={control_hz}"
    )

    tick_period = 1.0 / control_hz
    onscreen_render = not args.no_render
    render_cam_name = CAMERA_NAMES[0]

    for episode_idx in range(args.num_rollouts):
        print(f"{episode_idx=}")
        env = make_sim_env(seed=42 + episode_idx)
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()
        sync = _make_sync_point(obs)
        episode_max = 0.0

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
            plt.ion()

        policy.start()
        try:
            # Seed the controller so the background thread can produce chunk 0.
            policy.get_action(sync)
            if not policy.wait_for_first_chunk(timeout=120.0):
                raise TimeoutError("Timed out waiting for the first action chunk")

            for _ in range(EPISODE_LENGTH):
                tick_started = time.monotonic()
                action = policy.get_action(sync)
                if action is None:
                    raise RuntimeError("get_action returned None after first chunk")

                obs, reward, done = env.step(np.asarray(action, dtype=np.float64))
                sync = _make_sync_point(obs)
                episode_max = max(episode_max, reward)

                if onscreen_render:
                    plt_img.set_data(obs.cameras[render_cam_name].rgb)
                    plt.pause(0.001)

                remaining = tick_period - (time.monotonic() - tick_started)
                if remaining > 0:
                    time.sleep(remaining)

                if done:
                    break
        finally:
            policy.stop(timeout=60.0)

        stats = policy.stats()
        print(
            f"  stats: chunks={stats.chunks} d={stats.inference_delay} "
            f"s={stats.execution_horizon} deadline_misses={stats.deadline_misses} "
            f"stalled={stats.stalled_ticks} "
            f"median_latency_ms={stats.median_latency_s * 1e3:.1f}"
        )
        if episode_max >= 4.0:
            print(f"Episode {episode_idx} successful.")
        else:
            print(f"Episode {episode_idx} failed (max reward={episode_max}).")

        if onscreen_render:
            plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
