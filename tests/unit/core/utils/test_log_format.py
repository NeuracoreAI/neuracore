import logging

import pytest

from neuracore.core.utils.log_format import (
    LOG_FORMAT,
    RUST_TRACE_LEVEL,
    install_stream_logging,
)

TIMESTAMP_WIDTH = len("2026-09-14 19:16:39,392")


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def _render(name, level, message):
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    return logging.Formatter(LOG_FORMAT).format(record)


def test_renders_the_same_columns_as_the_rust_daemon():
    line = _render(
        "data_daemon.upload.worker", logging.WARNING, "retry scheduled attempt=3"
    )

    timestamp, rest = line[:TIMESTAMP_WIDTH], line[TIMESTAMP_WIDTH:]
    assert timestamp[4] == "-"
    assert timestamp[10] == " "
    assert timestamp[19] == ","
    assert rest == " WARNING  data_daemon.upload.worker      retry scheduled attempt=3"


def test_installs_trace_level_and_keeps_extra_handler_formatters(restore_root_logging):
    json_formatter = logging.Formatter("%(message)s")
    file_like_handler = logging.StreamHandler()
    file_like_handler.setFormatter(json_formatter)

    install_stream_logging(logging.DEBUG, extra_handlers=[file_like_handler])

    assert logging.getLevelName(RUST_TRACE_LEVEL) == "TRACE"

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert file_like_handler in root.handlers
    assert file_like_handler.formatter is json_formatter

    shared = [h for h in root.handlers if h is not file_like_handler]
    assert len(shared) == 1
    assert shared[0].formatter._fmt == LOG_FORMAT
