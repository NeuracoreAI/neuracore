"""Logging handlers that split records between stdout and stderr."""

import logging
import sys


def build_stream_handlers() -> list[logging.Handler]:
    """Build handlers sending records below WARNING to stdout, the rest to stderr.

    Returns:
        A stdout handler and a stderr handler.
    """
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    return [stdout_handler, stderr_handler]
