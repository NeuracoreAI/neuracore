#!/usr/bin/env python3
"""Plot daemon event timeline from a CSV event log.

Usage:
    python -m neuracore.data_daemon.tools.plot_daemon_events
    python -m neuracore.data_daemon.tools.plot_daemon_events /path/to/csv

Defaults to ~/.neuracore/data_daemon/daemon_events_timeline.csv when no path is given.
Produces an interactive HTML scatter plot alongside the CSV file.

Generate the CSV by running the daemon in debug mode:
    NDD_DEBUG=true nc-data-daemon launch
    nc-data-daemon launch --debug
"""

import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go

from neuracore.data_daemon.helpers import get_daemon_db_path

DEFAULT_EVENTS_CSV = get_daemon_db_path().parent / "daemon_events_timeline.csv"


def plot_events(csv_path: Path) -> None:
    """Read the event log CSV and write an interactive HTML timeline.

    Args:
        csv_path: Path to the CSV file produced by EventLogger.
    """
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No events found in CSV.")
        return

    rows.sort(key=lambda row: float(row["timestamp"]))

    event_order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row["event_name"] not in seen:
            event_order.append(row["event_name"])
            seen.add(row["event_name"])

    fig = go.Figure()

    for event in event_order:
        xs = []
        ys = []

        for row in rows:
            if row["event_name"] == event:
                xs.append(float(row["timestamp"]))
                ys.append(event)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=event,
                marker=dict(size=8, opacity=0.85),
            )
        )

    fig.update_layout(
        title="Events over Time",
        xaxis_title="Timestamp (s)",
        yaxis_title="Event",
        yaxis=dict(
            categoryorder="array",
            categoryarray=event_order[::-1],
        ),
        height=max(400, len(event_order) * 35 + 150),
    )

    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(
        xaxis=dict(title="Timestamp (s)"),
        yaxis=dict(title="Event", autorange="reversed"),
        legend_title="Event Type",
        height=max(400, len(event_order) * 35 + 150),
    )

    output_path = csv_path.with_suffix(".html")
    fig.write_html(output_path)
    print(f"Chart written to {output_path}")


def main() -> None:
    """Parse arguments and generate the event timeline chart."""
    parser = argparse.ArgumentParser(
        description="Plot daemon event timeline from a CSV event log"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_EVENTS_CSV,
        help=f"Path to event log CSV (default: {DEFAULT_EVENTS_CSV})",
    )
    args = parser.parse_args()

    plot_events(args.csv_path)


if __name__ == "__main__":
    main()
