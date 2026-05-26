"""Utility functions."""

import functools
import logging
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_NOISY_THIRD_PARTY = [
    "datasets",
    "huggingface_hub",
    "transformers",
    "urllib3",
    "filelock",
    "asyncio",
    "tensorflow",
    "tensorflow_datasets",
    "absl",
]


def deprecated(reason: str = "") -> Callable[[F], F]:
    """Decorator to mark functions as deprecated using warnings."""

    def decorator(func: F) -> Any:
        @functools.wraps(func)
        def wrapper(*args: tuple, **kwargs: dict[str, Any]) -> Any:
            warnings.warn(
                f"Function {func.__name__} is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def setup_logging(
    level: int | str | None = None,
    suppress_third_party: bool = True,
    json_format: bool | None = None,
    log_file: Path | str | None = None,
    force: bool = False,
    console: object = None,
    handler: logging.Handler | None = None,
) -> None:
    """Configure the root logger for the current process.

    Call once at each process entry point (CLI callback, worker main).
    Not for library code. Explicit arguments take precedence over env vars.

    Args:
        level: Log level (e.g. logging.DEBUG or "DEBUG"). Defaults to
            NEURACORE_LOG_LEVEL env var, or INFO.
        suppress_third_party: Silence known noisy third-party loggers.
        json_format: Emit JSON lines. Defaults to NEURACORE_LOG_JSON env var,
            or False.
        log_file: Path to write logs to in addition to stderr. Defaults to
            NEURACORE_LOG_FILE env var, or None.
        force: Re-apply configuration even if already called once.
        console: Optional Rich Console instance passed to RichHandler.
        handler: Custom handler to install (e.g. QueueHandler for workers).
            When provided, json_format/log_file/console are ignored.

    Environment variables (all optional, override defaults):
        NEURACORE_LOG_LEVEL — log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
        NEURACORE_LOG_FILE  — path to write logs to in addition to stderr
        NEURACORE_LOG_JSON  — set to "1" to emit JSON lines (for training/server)
    """
    root = logging.getLogger()
    already_configured = any(
        getattr(existing, "_neuracore_managed", False) for existing in root.handlers
    )

    if already_configured and not force:
        return

    if isinstance(level, int):
        resolved_level = level
    else:
        level_name = level or os.environ.get("NEURACORE_LOG_LEVEL", "INFO")
        numeric = logging.getLevelName(level_name.upper())
        if not isinstance(numeric, int):
            warnings.warn(
                f"Unknown log level {level!r}, defaulting to INFO.", stacklevel=2
            )
            numeric = logging.INFO
        resolved_level = numeric

    if not already_configured:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")

    for existing_handler in list(root.handlers):
        if getattr(existing_handler, "_neuracore_managed", False):
            root.removeHandler(existing_handler)

    handlers = (
        [handler]
        if handler is not None
        else _create_handlers(json_format, log_file, console)
    )
    root.setLevel(resolved_level)
    for new_handler in handlers:
        new_handler._neuracore_managed = True  # type: ignore[attr-defined]
        root.addHandler(new_handler)

    if suppress_third_party and handler is None:
        suppress_level = max(resolved_level, logging.WARNING)
        for logger_name in _NOISY_THIRD_PARTY:
            logger = logging.getLogger(logger_name)
            logger.setLevel(suppress_level)
            logger.handlers.clear()  # remove own StreamHandlers to avoid double-logging
            logger.propagate = True  # let WARNING+ reach root's RichHandler

    logging.captureWarnings(True)


def _create_handlers(
    json_format: bool | None, log_file: Path | str | None, console: object
) -> list[logging.Handler]:
    """Build console and optional file handlers."""
    json_format = (
        json_format
        if json_format is not None
        else os.environ.get("NEURACORE_LOG_JSON", "0") == "1"
    )
    log_file = log_file or os.environ.get("NEURACORE_LOG_FILE")
    log_file = Path(log_file) if log_file else None

    formatter: logging.Formatter | None = None
    if json_format:
        from neuracore.ml.logging.json_line_formatter import JsonLineLogFormatter

        formatter = JsonLineLogFormatter()
        console_handler: logging.Handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
    else:
        from rich.highlighter import NullHighlighter
        from rich.logging import RichHandler
        from rich.text import Text

        console_handler = RichHandler(
            console=console,  # type: ignore[arg-type]
            rich_tracebacks=False,
            show_path=True,
            highlighter=NullHighlighter(),
            omit_repeated_times=False,
            log_time_format=lambda dt: Text(
                dt.strftime("[%d %b %y %H:%M:%S.") + f"{dt.microsecond // 1000:03d}]"
            ),
        )

    handlers: list[logging.Handler] = [console_handler]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        if formatter:
            file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    return handlers
