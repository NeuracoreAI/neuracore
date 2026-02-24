"""JSONL formatter utilities for training and validation log files."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from neuracore.ml.logging.log_levels import LOG_SEVERITY_LEVELS


class JsonLineLogFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a single JSON object string.

        Args:
            record: The log record to format.

        Returns:
            A JSON string representing the log record.
        """
        message = record.getMessage()
        severity = record.levelname.upper()
        if severity not in LOG_SEVERITY_LEVELS:
            severity = "INFO"

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        event: dict[str, object] = {
            "timestamp": timestamp.replace("+00:00", "Z"),
            "severity": severity,
            "message": message,
        }
        try:
            return json.dumps(event, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps({"severity": "ERROR", "message": message})
