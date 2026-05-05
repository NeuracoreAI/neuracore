import json
import logging
import sys

from neuracore.ml.logging.json_line_formatter import JsonLineLogFormatter


def test_json_line_formatter_includes_traceback_for_exception_logs() -> None:
    logger = logging.getLogger("test_json_line_formatter")
    record = None
    try:
        raise ValueError("bad sample")
    except ValueError:
        record = logger.makeRecord(
            name=logger.name,
            level=logging.ERROR,
            fn=__file__,
            lno=10,
            msg="Error loading item %s.",
            args=(3,),
            exc_info=sys.exc_info(),
            func=None,
            extra=None,
        )

    formatted = JsonLineLogFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["severity"] == "ERROR"
    assert payload["message"] == "Error loading item 3."
    assert "Traceback (most recent call last)" in payload["traceback"]
    assert "ValueError: bad sample" in payload["traceback"]


def test_json_line_formatter_omits_traceback_without_exception() -> None:
    logger = logging.getLogger("test_json_line_formatter")
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn=__file__,
        lno=10,
        msg="plain message",
        args=(),
        exc_info=None,
        func=None,
        extra=None,
    )

    payload = json.loads(JsonLineLogFormatter().format(record))

    assert payload["message"] == "plain message"
    assert "traceback" not in payload
