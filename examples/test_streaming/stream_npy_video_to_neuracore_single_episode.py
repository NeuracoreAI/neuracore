#!/usr/bin/env python3
"""Stream NPY/video teleop data to Neuracore.

Reads episode data and streams it to Neuracore,
recreating a dataset as if it had been recorded live.
"""

import argparse
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any
import numpy as np
import neuracore as nc



from common.configs import (
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    URDF_PATH,
)
from common.episode_loader import load_rgb_frames


def load_episode(episode_dir: Path) -> list[tuple[float, str, Any]]:
    """Load episode data and return events sorted by timestamp.

    Returns list of (timestamp, event_type, payload) where event_type is one of:
    'joint_positions', 'joint_target_positions', 'gripper_open', 'gripper_target',
    'rgb'
    """
    events: list[tuple[float, str, Any]] = []

    # Joint positions (radians)
    jp_path = episode_dir / "joint_positions.npz"
    if jp_path.exists():
        print("  Loading joint_positions.npz...", end=" ", flush=True)
        data = np.load(jp_path)
        ts = data["timestamps"]
        vals = data["values"]
        for t, row in zip(ts, vals):
            d = {name: float(v) for name, v in zip(JOINT_NAMES, row)}
            events.append((float(t), "joint_positions", d))
        print(f"{len(ts)} samples")

    # Joint target positions (radians)
    jtp_path = episode_dir / "joint_target_positions.npz"
    if jtp_path.exists():
        print("  Loading joint_target_positions.npz...", end=" ", flush=True)
        data = np.load(jtp_path)
        ts = data["timestamps"]
        vals = data["values"]
        for t, row in zip(ts, vals):
            d = {name: float(v) for name, v in zip(JOINT_NAMES, row)}
            events.append((float(t), "joint_target_positions", d))
        print(f"{len(ts)} samples")

    # Gripper open amounts
    go_path = episode_dir / "gripper_open_amounts.npz"
    if go_path.exists():
        print("  Loading gripper_open_amounts.npz...", end=" ", flush=True)
        data = np.load(go_path)
        for t, v in zip(data["timestamps"], data["values"]):
            events.append((float(t), "parallel_gripper_open_amounts", float(v)))
        print(f"{len(data['timestamps'])} samples")

    # Gripper target open amounts
    gt_path = episode_dir / "gripper_target_open_amounts.npz"
    if gt_path.exists():
        print("  Loading gripper_target_open_amounts.npz...", end=" ", flush=True)
        data = np.load(gt_path)
        for t, v in zip(data["timestamps"], data["values"]):
            events.append((float(t), "parallel_gripper_target_open_amounts", float(v)))
        print(f"{len(data['timestamps'])} samples")

    rgb_result = load_rgb_frames(episode_dir)
    if rgb_result is not None:
        ts_rgb, frames = rgb_result
        print(f"  Loaded {len(frames)} RGB frames")
        for t, frame in zip(ts_rgb, frames):
            if frame.ndim == 3:
                events.append((float(t), "rgb", frame))
            else:
                raise ValueError(f"RGB frame is not a numpy array: {frame.shape}")
    else:
        raise ValueError(f"No RGB frames found in {episode_dir}!")

    print("  Sorting events...", end=" ", flush=True)
    events.sort(key=lambda e: e[0])
    print(f"{len(events)} total")
    return events


LOGGING_FREQUENCY = 1000.0  # Hz
LOGGING_TIME_TOLERANCE = 0.001  # seconds


class ReplayClock:
    """Shared replay start (wall time).

    Main sets `start_wall` right before releasing workers.
    """

    start_wall: float | None = None


def _logger_worker(
    event_type: str,
    worker_queue: Queue[tuple[float, Any]],
    log_fn: Callable[[Any, float], None],
    start_event: threading.Event,
    t_start: float,
    replay_clock: ReplayClock,
) -> None:
    """Worker thread for a single Neuracore data type.

    In real-time mode: waits until target wall time
    (`replay_clock.start_wall + (event_ts - t_start)`), then logs to
    Neuracore. All streams share the same replay clock so they stay aligned.
    """

    def _log_event(evt_type: str, payload: Any, timestamp: float) -> None:
        """Log an event to Neuracore with the given timestamp (wall time)."""
        try:
            log_fn(payload, timestamp)
        except Exception as e:
            print(f"⚠️ [{evt_type}] failed to log event: {e}")

    start_event.wait()
    replay_start_wall = (
        replay_clock.start_wall
        if replay_clock.start_wall is not None
        else time.monotonic()
    )

    while True:
        start_time = time.monotonic()

        item = worker_queue.get()
        if item is None:
            break  # to stop the thread

        event_timestamp, payload = item

        target_timestamp_wall = replay_start_wall + (event_timestamp - t_start)
        # keep checking the remaining time until it is less than the tolerance
        while True:
            wall_time_now = time.monotonic()
            remaining = target_timestamp_wall - wall_time_now
            if remaining < LOGGING_TIME_TOLERANCE:
                break
            time.sleep(1 / LOGGING_FREQUENCY)
        _log_event(event_type, payload, wall_time_now)

        # sleep the remaining time to keep the logging frequency, if there is any
        elapsed = time.monotonic() - start_time
        sleep_time = 1.0 / LOGGING_FREQUENCY - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[{event_type}] logger thread exiting.")


def stream_to_neuracore(
    events: list[tuple[float, str, Any]],
) -> None:
    """Stream events to Neuracore using one thread + queue per data type.

    In real-time mode, all streams use a shared replay clock so events are
    logged at wall time = replay_start_wall + (event_timestamp - t_start),
    keeping streams aligned.
    """
    if not events:
        print("No events to stream.")
        return

    t_start = events[0][0]

    # Map event_type -> logging function using Neuracore APIs.
    # Extend this mapping to support more Neuracore data types.
    def log_joint_positions_fn(payload: Any, ts: float) -> None:
        nc.log_joint_positions(positions=payload, timestamp=ts)

    def log_joint_target_positions_fn(payload: Any, ts: float) -> None:
        nc.log_joint_target_positions(target_positions=payload, timestamp=ts)

    def log_gripper_open_fn(payload: Any, ts: float) -> None:
        nc.log_parallel_gripper_open_amounts(
            values={GRIPPER_LOGGING_NAME: float(payload)}, timestamp=ts
        )

    def log_gripper_target_fn(payload: Any, ts: float) -> None:
        nc.log_parallel_gripper_target_open_amounts(
            values={GRIPPER_LOGGING_NAME: float(payload)}, timestamp=ts
        )

    def log_rgb_fn(payload: Any, ts: float) -> None:
        # payload is already a numpy frame from load_rgb_frames
        nc.log_rgb(name="rgb", rgb=payload, timestamp=ts)

    LOGGERS: dict[str, Callable[[Any, float], None]] = {
        "joint_positions": log_joint_positions_fn,
        "joint_target_positions": log_joint_target_positions_fn,
        "parallel_gripper_open_amounts": log_gripper_open_fn,
        "parallel_gripper_target_open_amounts": log_gripper_target_fn,
        "rgb": log_rgb_fn,
    }

    queues: dict[str, "Queue[tuple[float, Any] | None]"] = {
        event_type: Queue() for event_type in LOGGERS.keys()
    }

    last_event_timestamp: dict[str, float] = {}
    # Fill the queues with the events.
    for event_timestamp, event_type, payload in events:
        if event_timestamp < last_event_timestamp.get(event_type, 0):
            print(
                "⚠️ Event timestamp "
                f"{event_timestamp} is smaller than the last event timestamp "
                f"{last_event_timestamp.get(event_type, 0)} "
                f"for event type '{event_type}', skipping."
            )
            continue
        last_event_timestamp[event_type] = event_timestamp
        q = queues.get(event_type)
        if q is not None:
            q.put((event_timestamp, payload))
        else:
            print(f"⚠️ No queue for event type '{event_type}', skipping.")
    for q in queues.values():
        q.put(None)  # sentinel to stop each worker

    start_event = threading.Event()
    replay_clock = ReplayClock()

    # Threads for logging different event types.
    threads: list[threading.Thread] = []
    for event_type, worker_queue in queues.items():
        thread = threading.Thread(
            target=_logger_worker,
            args=(
                event_type,
                worker_queue,
                LOGGERS[event_type],
                start_event,
                t_start,
                replay_clock,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    nc.start_recording()

    # Replay starts now: workers use this as the wall-time origin.
    replay_clock.start_wall = time.monotonic()
    start_event.set()

    for t in threads:
        t.join()

    nc.stop_recording()
    print("✓ Streaming complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream NPY/video teleop data to Neuracore"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing episode_XXXX subdirs (output of example 7)",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to stream (0-based)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for the Neuracore dataset (default: imported-<input_dir>-ep<index>)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    episode_dir = input_dir / f"episode_{args.episode_index:04d}"
    if not episode_dir.is_dir():
        print(f"❌ Episode directory not found: {episode_dir}")
        sys.exit(1)

    dataset_name = args.dataset_name or (
        f"imported-{input_dir.name}-ep{args.episode_index}"
    )

    print("=" * 60)
    print("STREAM NPY/VIDEO TO NEURACORE")
    print("=" * 60)
    print(f"Input:    {episode_dir}")
    print(f"Dataset:  {dataset_name}")
    print("Mode:     real-time replay")
    print()

    # Load episode
    print("📂 Loading episode data...")
    events = load_episode(episode_dir)

    # Connect to Neuracore
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name="AgileX PiPER",
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    print(f"\n🔧 Creating dataset '{dataset_name}'...")
    nc.create_dataset(
        name=dataset_name,
        description=f"Imported from {input_dir.name} episode {args.episode_index}",
    )

    print("\n📤 Streaming to Neuracore...")
    stream_to_neuracore(events)

    # nc.logout()
    print("\n👋 Done.")


if __name__ == "__main__":
    main()
