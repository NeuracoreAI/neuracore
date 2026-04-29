import pytest
import requests
import requests_mock
from requests.adapters import HTTPAdapter

import neuracore.core.utils.http_session as _mod
from neuracore.core.utils.http_session import _build_session, get_session

_URL = "https://api.neuracore.com"


@pytest.fixture(autouse=True)
def reset_session():
    _mod._session = None
    yield
    _mod._session = None


def test_get_session_returns_session():
    assert isinstance(get_session(), requests.Session)


def test_get_session_is_singleton():
    assert get_session() is get_session()


def test_get_session_rebuilds_when_adapters_cleared():
    s1 = get_session()
    s1.adapters.clear()
    s2 = get_session()
    assert s2 is not s1
    assert s2.adapters


def test_session_mounts_http_and_https():
    s = _build_session()
    assert isinstance(s.get_adapter("https://"), HTTPAdapter)
    assert isinstance(s.get_adapter("http://"), HTTPAdapter)


def test_retry_config():
    retry = _build_session().get_adapter(_URL).max_retries
    assert retry.connect == 3
    assert retry.read == 0
    assert retry.status == 0
    assert retry.total == 3


def test_retry_allows_all_methods():
    retry = _build_session().get_adapter(_URL).max_retries
    assert retry.allowed_methods is False


def test_no_retry_on_5xx():
    with requests_mock.Mocker() as m:
        m.get(f"{_URL}/health", status_code=500)
        r = _build_session().get(f"{_URL}/health")
    assert r.status_code == 500
    assert m.call_count == 1
