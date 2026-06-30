"""Tests for Hugging Face Hub retry helpers."""

from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import HfHubHTTPError

from neuracore.ml.utils.hf_hub_retry import _is_rate_limit_error, call_with_hf_hub_retry


def test_is_rate_limit_error_from_hf_hub_http_error() -> None:
    response_429 = MagicMock()
    response_429.status_code = 429
    assert _is_rate_limit_error(
        HfHubHTTPError("429 Client Error: Too Many Requests", response=response_429)
    )

    response_500 = MagicMock()
    response_500.status_code = 500
    assert not _is_rate_limit_error(
        HfHubHTTPError("500 Server Error", response=response_500)
    )
    assert not _is_rate_limit_error(Exception("connection refused"))


def test_is_rate_limit_error_from_wrapped_oserror() -> None:
    response = MagicMock()
    response.status_code = 429
    hub_error = HfHubHTTPError(
        "429 Client Error: Too Many Requests",
        response=response,
    )
    wrapped = OSError(
        "We couldn't connect to 'https://huggingface.co' to load the files, "
        "and couldn't find them in the cached files."
    )
    wrapped.__cause__ = hub_error
    assert _is_rate_limit_error(wrapped)


def test_call_with_hf_hub_retry_succeeds_on_second_attempt() -> None:
    response = MagicMock()
    response.status_code = 429
    rate_limit_error = HfHubHTTPError(
        "429 Client Error: Too Many Requests",
        response=response,
    )
    func = MagicMock(side_effect=[rate_limit_error, "ok"])
    assert call_with_hf_hub_retry(func, max_retries=2, base_delay_s=0) == "ok"
    assert func.call_count == 2


def test_call_with_hf_hub_retry_succeeds_on_wrapped_oserror() -> None:
    response = MagicMock()
    response.status_code = 429
    hub_error = HfHubHTTPError(
        "429 Client Error: Too Many Requests",
        response=response,
    )
    wrapped = OSError("couldn't connect to huggingface.co")
    wrapped.__cause__ = hub_error

    func = MagicMock(side_effect=[wrapped, "ok"])
    assert call_with_hf_hub_retry(func, max_retries=2, base_delay_s=0) == "ok"
    assert func.call_count == 2


def test_call_with_hf_hub_retry_raises_non_rate_limit() -> None:
    func = MagicMock(side_effect=ValueError("bad config"))
    with pytest.raises(ValueError, match="bad config"):
        call_with_hf_hub_retry(func, max_retries=3, base_delay_s=0)
