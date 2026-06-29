"""Hugging Face Hub download helpers with rate-limit retries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BASE_DELAY_S = 2.0
_DEFAULT_MAX_DELAY_S = 120.0


def _exception_indicates_rate_limit(exc: BaseException) -> bool:
    """Return True if a single exception indicates an HF Hub rate limit."""
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = ()  # type: ignore[misc, assignment]

    if isinstance(exc, HfHubHTTPError):
        return exc.response is not None and exc.response.status_code == 429

    response = getattr(exc, "response", None)
    return response is not None and getattr(response, "status_code", None) == 429


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception or its cause chain indicates HTTP 429.

    Transformers often wraps ``HfHubHTTPError`` in ``OSError`` without
    including ``429`` in the outer message, so the cause chain must be walked.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _exception_indicates_rate_limit(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def call_with_hf_hub_retry(
    func: Callable[[], T],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay_s: float = _DEFAULT_BASE_DELAY_S,
    max_delay_s: float = _DEFAULT_MAX_DELAY_S,
) -> T:
    """Call ``func`` with exponential backoff on Hugging Face Hub rate limits.

    Bind call arguments with :func:`functools.partial` before passing ``func``.

    Args:
        func: Zero-argument callable that may raise on Hub download failure.
        max_retries: Maximum number of attempts after the first failure.
        base_delay_s: Initial backoff delay in seconds.
        max_delay_s: Maximum backoff delay in seconds.

    Returns:
        Return value of ``func``.

    Raises:
        The last exception if all retries are exhausted.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = min(base_delay_s * (2**attempt), max_delay_s)
            logger.warning(
                "Hugging Face Hub rate limit (attempt %d/%d); retrying in %.1fs: %s",
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise ValueError(f"max_retries must be >= 0, got {max_retries}")
