"""Sampled logger for high-frequency log messages.

Provides utilities to reduce log spam by only logging at configurable intervals.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def make_sampled_logger(
    log_format: str,
    log_interval: int = 1000,
    target_logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> Callable[..., None]:
    """Create a sampled logger that logs first/last items at intervals.

    Returns a function that logs messages for:
    - First trace: first and last item
    - Every Nth trace (based on log_interval): first and last item

    Args:
        log_format: Format string for the log message. First placeholder receives
                    the message number, remaining placeholders receive extra_args.
        log_interval: Log every Nth trace (default 1000)
        target_logger: Logger instance to use (default: module logger)
        level: Log level to use (default: DEBUG)

    Returns:
        A function: (trace_id, item_idx, total_items, *format_args) -> None
    """
    seen_traces: dict[str, int] = {}
    message_counter = 0
    _logger = target_logger or logger

    def log_sampled(
        trace_id: str,
        item_idx: int,
        total_items: int,
        *format_args: object,
    ) -> None:
        nonlocal message_counter

        if trace_id not in seen_traces:
            message_counter += 1
            seen_traces[trace_id] = message_counter

        msg_num = seen_traces[trace_id]
        is_first = item_idx == 0
        is_last = item_idx == total_items - 1
        should_log = msg_num == 1 or msg_num % log_interval == 0

        if should_log and (is_first or is_last):
            _logger.log(level, log_format, msg_num, *format_args)

    return log_sampled
