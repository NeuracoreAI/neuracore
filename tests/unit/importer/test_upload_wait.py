import logging

import pytest
import requests

from neuracore.importer.core import upload_wait
from neuracore.importer.core.exceptions import ImporterError
from neuracore.importer.core.upload_wait import wait_for_uploads

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(upload_wait.time, "sleep", calls.append)
    return calls


def _backend(monkeypatch, answers: dict[str, list]) -> None:
    def is_complete(recording_id: str) -> bool:
        answer = answers[recording_id].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(
        upload_wait.backend_utils, "is_recording_upload_complete", is_complete
    )


def test_polls_every_pending_recording(monkeypatch, sleeps):
    answers = {"a": [False, True], "b": [True]}
    _backend(monkeypatch, answers)

    wait_for_uploads(["a", "b"], logger=LOGGER, poll_interval_s=1.0)

    assert answers == {"a": [], "b": []}
    assert sleeps == [1.0]


def test_slow_first_recording_does_not_block_the_others(monkeypatch, sleeps):
    answers = {"slow": [False, False, True], "fast": [True]}
    _backend(monkeypatch, answers)

    wait_for_uploads(["slow", "fast"], logger=LOGGER)

    assert answers == {"slow": [], "fast": []}


def test_transient_errors_are_retried(monkeypatch, sleeps):
    answers = {"a": [requests.ConnectionError("down"), True]}
    _backend(monkeypatch, answers)

    wait_for_uploads(["a"], logger=LOGGER)

    assert answers == {"a": []}


def test_fails_when_no_upload_progresses(monkeypatch, sleeps):
    _backend(monkeypatch, {"a": [False]})

    with pytest.raises(ImporterError, match="1 recording"):
        wait_for_uploads(["a"], logger=LOGGER, timeout_per_recording_s=0.0)


def test_no_recordings_returns_at_once(monkeypatch, sleeps):
    _backend(monkeypatch, {})

    wait_for_uploads([], logger=LOGGER)

    assert sleeps == []


def test_budget_scales_with_pending_recordings(monkeypatch, sleeps):
    clock = iter([0.0, 0.0, 15.0, 15.0])
    monkeypatch.setattr(upload_wait.time, "monotonic", lambda: next(clock))
    _backend(monkeypatch, {"a": [False, True], "b": [False, True]})

    # 10 s per recording: two pending allow 20 s, so 15 s is not a timeout.
    wait_for_uploads(["a", "b"], logger=LOGGER, timeout_per_recording_s=10.0)
