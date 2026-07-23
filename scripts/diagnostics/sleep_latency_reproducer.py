#!/usr/bin/env python3
"""Minimal absolute-deadline sleep cadence for macOS scheduler captures."""

from __future__ import annotations

import argparse
import time

from tests.integration.platform.data_daemon.shared.diagnostic_sleep import (
    diagnostic_sleep,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--period-ms", type=float, default=1000 / 15)
    parser.add_argument("--anomaly-threshold-ms", type=float, default=5.0)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.period_ms <= 0:
        parser.error("duration and period must be positive")

    period_ns = round(args.period_ms * 1_000_000)
    stop_ns = time.monotonic_ns() + round(args.duration_s * 1_000_000_000)
    next_deadline_ns = time.monotonic_ns()
    iteration = 0
    while next_deadline_ns < stop_ns:
        next_deadline_ns += period_ns
        remaining_ns = next_deadline_ns - time.monotonic_ns()
        if remaining_ns > 0:
            diagnostic_sleep(
                remaining_ns / 1_000_000_000,
                "minimal-15hz-cadence",
                anomaly_threshold_ms=args.anomaly_threshold_ms,
                correlation_id=f"minimal/iteration={iteration}",
            )
        iteration += 1


if __name__ == "__main__":
    main()
