"""Shared terminal log formatting for the Python SDK and the Rust components."""

from __future__ import annotations

import logging
from collections.abc import Sequence

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-30s %(message)s"

RUST_TRACE_LEVEL = 5
RUST_TRACE_LEVEL_NAME = "TRACE"


def register_rust_trace_level() -> None:
    """Name the sub-DEBUG severity that Rust trace events arrive on."""
    logging.addLevelName(RUST_TRACE_LEVEL, RUST_TRACE_LEVEL_NAME)


def build_stream_handler() -> logging.StreamHandler:
    """Return a stderr handler using the shared Neuracore line format."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def install_stream_logging(
    level: int = logging.INFO,
    extra_handlers: Sequence[logging.Handler] = (),
) -> None:
    """Configure root logging with the shared line format.

    Args:
        level: Root logger severity threshold.
        extra_handlers: Handlers to attach alongside the stream handler. Each
            keeps whichever formatter it was given by its caller.
    """
    register_rust_trace_level()
    handlers: list[logging.Handler] = [build_stream_handler(), *extra_handlers]
    logging.basicConfig(level=level, handlers=handlers, force=True)
