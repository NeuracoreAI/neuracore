"""Shared display helpers for training CLI commands."""

from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Map for pretty status coloring on cloud runs
STATUS_STYLES = {
    "COMPLETED": "green",
    "RUNNING": "blue",
    "PENDING": "yellow",
    "PREPARING_DATA": "yellow",
    "FAILED": "red",
    "CANCELLED": "red",
}


@dataclass
class RunDisplayRow:
    """Display-ready representation of a training run for tabular output."""

    name: str
    date: str
    success: str
    algorithm: str
    dataset: str
    status: str | None = None


def style_success(success: str) -> Text:
    """Colorize success text."""
    normalized = success.lower()
    if normalized == "yes":
        return Text(success, style="green bold")
    if normalized == "no":
        return Text(success, style="red bold")
    return Text(success, style="yellow bold")


def style_status(status: str) -> Text:
    """Colorize status text for cloud runs."""
    style = STATUS_STYLES.get(status.upper(), "white")
    return Text(status, style=f"{style} bold")


def print_run_table(
    console: Console,
    title: str,
    rows: list[RunDisplayRow],
    include_status: bool,
) -> None:
    """Render tabular run data with consistent formatting."""
    table = Table(
        title=title,
        box=box.MINIMAL,
        show_header=True,
        header_style="bold",
        expand=False,
    )

    table.add_column("Name", min_width=8, max_width=28, overflow="fold")
    table.add_column(
        "Date",
        min_width=14,
        max_width=19 if include_status else 17,
        overflow="fold",
        no_wrap=True,
    )
    table.add_column("Success", min_width=7, max_width=8, no_wrap=True)
    if include_status:
        table.add_column("Status", min_width=7, max_width=12, no_wrap=True)
    table.add_column("Algorithm", min_width=8, max_width=18, overflow="fold")
    table.add_column("Dataset", min_width=8, max_width=18, overflow="fold")

    for row in rows:
        success_cell = style_success(row.success)
        values: list[Text | str] = [
            row.name,
            row.date,
            success_cell,
        ]
        if include_status:
            status_value = row.status or "UNKNOWN"
            values.append(style_status(status_value))
        values.extend([row.algorithm, row.dataset])
        table.add_row(*values)

    console.print(table)
