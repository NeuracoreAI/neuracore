"""Progress reporting abstractions for MCAP preprocessing and logging phases.

The importer has two long-running phases with different execution contexts
(worker subprocess, inline execution, tests). These reporters provide a single
interface so phase logic is independent of UI/transport details.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rich.progress import Progress, TaskID


class ProgressReporter(Protocol):
    """Common progress reporter protocol used by MCAP pipelines."""

    def start_phase(self, phase_name: str, total_items: int) -> None:
        """Start a named phase with an optional total (0 means unknown)."""

    def update(self, completed: int) -> None:
        """Report a new completed count."""

    def finish_phase(self) -> None:
        """Finalize reporting for the active phase."""


class NullProgressReporter:
    """No-op reporter for tests or silent operation."""

    def start_phase(self, phase_name: str, total_items: int) -> None:
        """Ignore phase start events."""
        return

    def update(self, completed: int) -> None:
        """Ignore progress updates."""
        return

    def finish_phase(self) -> None:
        """Ignore phase completion events."""
        return


class LoggingProgressReporter:
    """Emit periodic progress updates through the logger."""

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        *,
        report_every: int = 1000,
    ) -> None:
        """Create a logger-backed reporter with throttled update frequency."""
        self._logger = logger
        self._label = label
        self._report_every = max(1, report_every)
        self._phase_name = ""
        self._total_items = 0
        self._last_reported = 0

    def start_phase(self, phase_name: str, total_items: int) -> None:
        """Initialize counters and log the phase start."""
        self._phase_name = phase_name
        self._total_items = max(0, total_items)
        self._last_reported = 0
        total_label = str(self._total_items) if self._total_items > 0 else "unknown"
        self._logger.info("%s | %s: 0/%s", self._label, self._phase_name, total_label)

    def update(self, completed: int) -> None:
        """Log periodic progress updates."""
        if completed < 0:
            return
        should_report = (completed - self._last_reported) >= self._report_every
        if self._total_items > 0 and completed >= self._total_items:
            should_report = True
        if not should_report:
            return

        self._last_reported = completed
        if self._total_items > 0:
            pct = 100.0 * completed / max(1, self._total_items)
            self._logger.info(
                "%s | %s: %s/%s (%.1f%%)",
                self._label,
                self._phase_name,
                completed,
                self._total_items,
                pct,
            )
            return

        self._logger.info("%s | %s: %s", self._label, self._phase_name, completed)

    def finish_phase(self) -> None:
        """Log phase completion."""
        self._logger.info("%s | %s: done", self._label, self._phase_name)


@dataclass(slots=True)
class EmitProgressReporter:
    """Emit throttled progress updates through a callback.

    This lets MCAP reuse the base importer's progress transport (via callback)
    instead of constructing queue payloads independently.
    """

    emit_progress: Callable[[int, int | None, str | None], None]
    label: str
    report_every: int = 100

    _phase_name: str = ""
    _total_items: int = 0
    _last_reported: int = 0
    _last_completed: int = 0

    def start_phase(self, phase_name: str, total_items: int) -> None:
        """Start a phase and emit an initial zero-progress event."""
        self._phase_name = phase_name
        self._total_items = max(0, total_items)
        self._last_reported = 0
        self._last_completed = 0
        self._emit(0)

    def update(self, completed: int) -> None:
        """Emit throttled callback updates based on completion count."""
        if completed < 0:
            return
        self._last_completed = completed
        interval = max(1, int(self.report_every))
        should_report = (completed - self._last_reported) >= interval
        if self._total_items > 0 and completed >= self._total_items:
            should_report = True
        if not should_report:
            return
        self._emit(completed)

    def _emit(self, completed: int) -> None:
        label = self.label
        if self._phase_name:
            suffix = f" [{self._phase_name}]"
            label = f"{label}{suffix}" if label else self._phase_name
        try:
            self.emit_progress(completed, self._total_items or None, label)
            self._last_reported = completed
        except Exception:
            return

    def finish_phase(self) -> None:
        """Emit a final update if progress changed since the last emit."""
        if self._last_completed != self._last_reported:
            self._emit(self._last_completed)
        return


class RichProgressReporter:
    """Render progress in a provided rich Progress instance."""

    def __init__(self, progress: Progress, label: str) -> None:
        """Bind a rich progress instance and label for future tasks."""
        self._progress = progress
        self._label = label
        self._task_id: TaskID | None = None

    def start_phase(self, phase_name: str, total_items: int) -> None:
        """Create a new rich task for the active phase."""
        desc = f"{self._label} [{phase_name}]"
        self._task_id = self._progress.add_task(
            desc,
            total=total_items or None,
            completed=0,
        )

    def update(self, completed: int) -> None:
        """Update the rich task completion count."""
        if self._task_id is None:
            return
        self._progress.update(self._task_id, completed=completed, refresh=True)

    def finish_phase(self) -> None:
        """Leave the task rendered; lifecycle is managed by caller."""
        return
