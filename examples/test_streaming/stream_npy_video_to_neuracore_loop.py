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
from queue import Empty, Queue
from typing import Any

import numpy as np
from common.configs import GRIPPER_LOGGING_NAME, JOINT_NAMES, URDF_PATH
from common.episode_loader import load_rgb_frames

import neuracore as nc


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
    stop_event: threading.Event,
    t_start: float,
    replay_clock: ReplayClock,
) -> None:
    """Worker thread for a single Neuracore data type.

    In real-time mode: waits until target wall time
    (`replay_clock.start_wall + (event_ts - t_start)`), then logs to
    Neuracore. Exits when `stop_event` is set; no `None` sentinels.
    """

    def _log_event(evt_type: str, payload: Any, timestamp: float) -> None:
        try:
            log_fn(payload, timestamp)
        except Exception as e:
            print(f"⚠️ [{evt_type}] failed to log event: {e}")

    start_event.wait()
    get_timeout = 0.5

    while not stop_event.is_set():
        replay_start_wall = replay_clock.start_wall
        if replay_start_wall is None:
            time.sleep(1 / LOGGING_FREQUENCY)
            continue

        start_time = time.monotonic()
        try:
            item = worker_queue.get(timeout=get_timeout)
        except Empty:
            continue

        if stop_event.is_set():
            break

        event_timestamp, payload = item

        target_timestamp_wall = replay_start_wall + (event_timestamp - t_start)
        while not stop_event.is_set():
            wall_time_now = time.monotonic()
            remaining = target_timestamp_wall - wall_time_now
            if remaining < LOGGING_TIME_TOLERANCE:
                break
            time.sleep(min(1 / LOGGING_FREQUENCY, max(0, remaining)))
        wall_time_now = time.monotonic()
        _log_event(event_type, payload, wall_time_now)

        elapsed = time.monotonic() - start_time
        sleep_time = 1.0 / LOGGING_FREQUENCY - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[{event_type}] logger thread exiting.")


def _filler_loop(
    events: list[tuple[float, str, Any]],
    queues: dict[str, Queue[tuple[float, Any]]],
    replay_clock: ReplayClock,
    start_event: threading.Event,
    stop_event: threading.Event,
) -> None:
    """Repeatedly push the same episode into queues.

    Reset replay clock at each cycle. Exits when `stop_event` is set.
    """
    last_event_timestamp: dict[str, float] = {}
    cycle = 0
    while not stop_event.is_set():
        replay_clock.start_wall = time.monotonic()
        if cycle == 0:
            start_event.set()
        last_event_timestamp.clear()
        for event_timestamp, event_type, payload in events:
            if stop_event.is_set():
                break
            if event_timestamp < last_event_timestamp.get(event_type, 0):
                continue
            last_event_timestamp[event_type] = event_timestamp
            q = queues.get(event_type)
            if q is not None:
                q.put((event_timestamp, payload))
        cycle += 1
        if stop_event.is_set():
            break
        while not stop_event.is_set():
            if all(q.empty() for q in queues.values()):
                break
            time.sleep(0.05)
    print("[filler] thread exiting.")


def stream_to_neuracore(
    events: list[tuple[float, str, Any]],
    stop_event: threading.Event,
    record: bool = False,
) -> list[threading.Thread]:
    """Stream events to Neuracore using one thread + queue per data type.

    Filler loop pushes the same episode indefinitely until stop_event is set.
    Returns the worker threads + filler so caller can join after setting stop_event.
    """
    if not events:
        print("No events to stream.")
        return []

    t_start = events[0][0]

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

    queues: dict[str, Queue[tuple[float, Any]]] = {
        event_type: Queue() for event_type in LOGGERS.keys()
    }

    start_event = threading.Event()
    replay_clock = ReplayClock()

    threads: list[threading.Thread] = []
    for event_type, worker_queue in queues.items():
        thread = threading.Thread(
            target=_logger_worker,
            args=(
                event_type,
                worker_queue,
                LOGGERS[event_type],
                start_event,
                stop_event,
                t_start,
                replay_clock,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    filler = threading.Thread(
        target=_filler_loop,
        args=(events, queues, replay_clock, start_event, stop_event),
        daemon=True,
    )
    filler.start()
    threads.append(filler)

    if record:
        nc.start_recording()

    return threads


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
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "Start Neuracore recording when process starts; "
            "stop when process exits (e.g. Ctrl+C)"
        ),
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
    print(f"Record:   {args.record}")
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

    if args.record:
        print(f"\n🔧 Creating dataset '{dataset_name}'...")
        nc.create_dataset(
            name=dataset_name,
            description=f"Imported from {input_dir.name} episode {args.episode_index}",
        )

    stop_event = threading.Event()
    print("\n📤 Streaming to Neuracore (same episode in a loop until Ctrl+C)...")
    threads = stream_to_neuracore(events, stop_event=stop_event, record=args.record)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        if args.record:
            nc.stop_recording()
        print("✓ Streaming complete.")

    nc.logout()
    print("\n👋 Done.")


if __name__ == "__main__":
    main()
