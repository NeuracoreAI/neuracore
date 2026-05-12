from collections import Counter
from unittest.mock import Mock

import pytest
import requests
import requests_mock
from requests.adapters import HTTPAdapter

from neuracore.core.utils.http_session import Session

# cspell:ignore poolmanager

_URL = "https://api.neuracore.com"


def test_session_is_requests_session():
    with Session() as s:
        assert isinstance(s, requests.Session)


def test_session_disables_keep_alive():
    with Session() as s:
        assert s.keep_alive is False


def test_session_accepts_keep_alive_kwarg():
    with Session(keep_alive=True) as s:
        assert s.keep_alive is True


def test_session_closes_on_exit():
    with Session() as s:
        adapter_call_counts = Counter(map(id, s.adapters.values()))
        adapters = {id(adapter): adapter for adapter in s.adapters.values()}

        for adapter in adapters.values():
            adapter.close = Mock()

        for adapter in adapters.values():
            adapter.close.assert_not_called()

    for adapter_id, adapter in adapters.items():
        assert adapter.close.call_count == adapter_call_counts[adapter_id]


def test_session_mounts_http_and_https():
    with Session() as s:
        assert isinstance(s.get_adapter("https://"), HTTPAdapter)
        assert isinstance(s.get_adapter("http://"), HTTPAdapter)


def test_retry_config():
    with Session() as s:
        retry = s.get_adapter(_URL).max_retries
    assert retry.connect == 3
    assert retry.read == 0
    assert retry.status == 0
    assert retry.total == 3


def test_retry_allows_all_methods():
    with Session() as s:
        retry = s.get_adapter(_URL).max_retries
    assert retry.allowed_methods is False


def test_no_retry_on_5xx():
    with requests_mock.Mocker() as m:
        m.get(f"{_URL}/health", status_code=500)
        with Session() as s:
            r = s.get(f"{_URL}/health")
    assert r.status_code == 500
    assert m.call_count == 1


def test_each_session_is_independent():
    with Session() as s1:
        pass
    with Session() as s2:
        pass
    assert s1 is not s2


@pytest.mark.parametrize("scheme", ["http://", "https://"])
def test_adapters_share_retry_config(scheme):
    with Session() as s:
        adapter = s.get_adapter(scheme)
    assert adapter.max_retries.connect == 3
