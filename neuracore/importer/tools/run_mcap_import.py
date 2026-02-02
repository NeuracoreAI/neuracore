#!/usr/bin/env python3
"""Run end-to-end MCAP import with optional inspection and post-run reporting."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _types_repo(repo_root: Path) -> Path | None:
    candidate = repo_root.parent / "neuracore_types"
    return candidate if candidate.exists() else None


def _build_pythonpath(repo_root: Path) -> str:
    parts: list[str] = [str(repo_root)]
    types_repo = _types_repo(repo_root)
    if types_repo is not None:
        parts.append(str(types_repo))

    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _resolve_output_dataset_name(config_path: Path) -> str | None:
    try:
        from neuracore_types.nc_data import DatasetImportConfig

        config = DatasetImportConfig.from_file(config_path)
        return config.output_dataset.name
    except Exception:
        pass

    # Fallback parser to avoid requiring a PyYAML dependency in this utility script.
    try:
        text = config_path.read_text(encoding="utf-8")
        block = re.search(
            r"(?ms)^\s*output_dataset\s*:\s*\n(?P<body>(?:^[ \t]+.*\n?)*)", text
        )
        if block:
            name_match = re.search(
                r"(?m)^[ \t]+name\s*:\s*(?P<name>.+?)\s*$", block.group("body")
            )
            if name_match:
                candidate = name_match.group("name").strip().strip("'\"")
                if candidate:
                    return candidate
    except Exception:
        pass
    return None


def _iter_mcap_files(dataset_path: Path, limit: int | None = None) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    files = sorted(dataset_path.rglob("*.mcap"))
    if limit is not None and limit > 0:
        return files[:limit]
    return files


def _run_command(cmd: list[str], env: dict[str, str], log_f: TextIO) -> int:
    command_str = shlex.join(cmd)
    print(f"\n$ {command_str}")
    log_f.write(f"\n$ {command_str}\n")
    log_f.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert process.stdout is not None
    start_time = time.monotonic()
    last_heartbeat = start_time
    while True:
        ready, _, _ = select.select([process.stdout], [], [], 1.0)
        if ready:
            line = process.stdout.readline()
            if line:
                print(line, end="")
                log_f.write(line)
                log_f.flush()
                continue
            if process.poll() is not None:
                break
        if process.poll() is not None:
            break
        now = time.monotonic()
        if now - last_heartbeat >= 30.0:
            elapsed = int(now - start_time)
            heartbeat = f"[run_mcap_import] command still running ({elapsed}s elapsed)"
            print(heartbeat)
            log_f.write(heartbeat + "\n")
            log_f.flush()
            last_heartbeat = now

    process.wait()
    log_f.flush()
    return int(process.returncode)


def _inspect_mcap_files(
    *,
    mcap_cli: Path,
    files: list[Path],
    env: dict[str, str],
    log_f: TextIO,
) -> int:
    if not files:
        print("No .mcap files found to inspect.")
        return 0

    for mcap_file in files:
        for cmd in (
            [str(mcap_cli), "info", str(mcap_file)],
            [str(mcap_cli), "list", "channels", str(mcap_file)],
        ):
            rc = _run_command(cmd, env=env, log_f=log_f)
            if rc != 0:
                return rc
    return 0


def _report_dataset(dataset_name: str, max_rows: int) -> int:
    try:
        import neuracore as nc
        from neuracore.core.data.dataset import Dataset
    except Exception as exc:
        print(f"Failed loading Neuracore client for summary: {exc}")
        return 1

    try:
        nc.login()
        dataset = Dataset.get_by_name(dataset_name)
        if dataset is None:
            print(f"Dataset not found: '{dataset_name}'")
            return 1
        recordings = list(dataset)
    except Exception as exc:
        print(f"Failed reading dataset summary for '{dataset_name}': {exc}")
        return 1

    print(f"\nDataset: {dataset_name}")
    print(f"Recording count: {len(recordings)}")
    if not recordings:
        return 0

    durations = [max(0.0, float(r.end_time) - float(r.start_time)) for r in recordings]
    total = sum(durations)
    print(
        "Duration seconds: "
        f"total={total:.3f} min={min(durations):.3f} "
        f"max={max(durations):.3f} avg={total/len(durations):.3f}"
    )

    rows_to_print = min(max_rows, len(recordings))
    print(f"Showing first {rows_to_print} recording(s):")
    for idx, recording in enumerate(recordings[:rows_to_print]):
        start_iso = dt.datetime.fromtimestamp(
            float(recording.start_time), tz=dt.timezone.utc
        ).isoformat()
        end_iso = dt.datetime.fromtimestamp(
            float(recording.end_time), tz=dt.timezone.utc
        ).isoformat()
        duration = float(recording.end_time) - float(recording.start_time)
        print(
            f"{idx:03d} id={recording.id} "
            f"duration_s={duration:.3f} start={start_iso} end={end_iso}"
        )
    if len(recordings) > rows_to_print:
        print(f"... {len(recordings) - rows_to_print} more recording(s) not shown")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MCAP import runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run MCAP import end-to-end: optional mcap-cli inspect, importer run, "
            "and recording duration summary."
        )
    )
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--mcap-cli",
        type=Path,
        default=Path("/usr/local/bin/mcap"),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run `mcap info` and `mcap list channels` before import.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=1,
        help="How many MCAP files to inspect when --dataset-dir is a directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-validation-warnings", action="store_true")
    parser.add_argument(
        "--max-recording-seconds",
        type=float,
        default=None,
        help=(
            "Sets NEURACORE_IMPORT_MAX_RECORDING_SECONDS for this run " "(e.g. 240)."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Dataset name for summary lookup. Defaults to output_dataset.name "
            "from --dataset-config."
        ),
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip post-import recording summary lookup.",
    )
    parser.add_argument(
        "--summary-max-rows",
        type=int,
        default=20,
        help="Max number of recording rows to print in summary.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--log-prefix", default="mcap_import")
    return parser.parse_args()


def main() -> int:
    """Run inspect/import/summary workflow and return an exit code."""
    args = parse_args()

    if not args.dataset_config.exists():
        print(f"Config not found: {args.dataset_config}")
        return 2
    if not args.dataset_dir.exists():
        print(f"Dataset path not found: {args.dataset_dir}")
        return 2
    if not args.robot_dir.exists():
        print(f"Robot dir not found: {args.robot_dir}")
        return 2
    if args.inspect and not args.mcap_cli.exists():
        print(f"mcap CLI not found: {args.mcap_cli}")
        return 2

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"{args.log_prefix}_{stamp}.log"

    repo_root = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(repo_root)
    if args.max_recording_seconds is not None:
        env["NEURACORE_IMPORT_MAX_RECORDING_SECONDS"] = str(args.max_recording_seconds)

    dataset_name = args.dataset_name or _resolve_output_dataset_name(
        args.dataset_config
    )

    print(f"Repo root: {repo_root}")
    print(f"Log file: {log_path}")
    if args.max_recording_seconds is not None:
        print(
            "Using NEURACORE_IMPORT_MAX_RECORDING_SECONDS="
            f"{env['NEURACORE_IMPORT_MAX_RECORDING_SECONDS']}"
        )
    if dataset_name:
        print(f"Dataset for summary: {dataset_name}")

    with log_path.open("w", encoding="utf-8") as log_f:
        if args.inspect:
            inspect_files = _iter_mcap_files(args.dataset_dir, limit=args.inspect_limit)
            rc = _inspect_mcap_files(
                mcap_cli=args.mcap_cli,
                files=inspect_files,
                env=env,
                log_f=log_f,
            )
            if rc != 0:
                print(f"Inspection failed (exit={rc}). See {log_path}")
                return rc

        cmd = [
            args.python_bin,
            "-m",
            "neuracore.importer.importer",
            "--dataset-config",
            str(args.dataset_config),
            "--dataset-dir",
            str(args.dataset_dir),
            "--robot-dir",
            str(args.robot_dir),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.no_validation_warnings:
            cmd.append("--no-validation-warnings")

        import_rc = _run_command(cmd, env=env, log_f=log_f)
        if import_rc != 0:
            print(f"Import failed (exit={import_rc}). See {log_path}")
            return import_rc

    print(f"\nImport completed successfully. Full log: {log_path}")

    if args.skip_summary:
        return 0
    if not dataset_name:
        print("Skipping summary because dataset name could not be resolved.")
        return 0
    return _report_dataset(dataset_name, max_rows=max(1, args.summary_max_rows))


if __name__ == "__main__":
    raise SystemExit(main())
