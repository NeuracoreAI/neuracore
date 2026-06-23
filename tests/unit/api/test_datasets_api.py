"""Unit tests for dataset API."""

import json

import pytest

from neuracore.api import datasets as api_datasets
from neuracore.core.exceptions import DatasetError


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict | None = None,
        reason: str = "",
        raw_text: str | None = None,
    ):
        self.status_code = status_code
        self.reason = reason
        self._payload = payload
        self._raw_text = raw_text
        self.ok = 200 <= status_code < 300

    def json(self) -> dict:
        if self._raw_text is not None:
            raise ValueError("not json")
        return self._payload

    @property
    def text(self) -> str:
        if self._raw_text is not None:
            return self._raw_text
        return json.dumps(self._payload)


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_kwargs: dict | None = None

    def post(self, *args, **kwargs) -> _FakeResponse:
        self.last_kwargs = kwargs
        return self._response


class _DummyAuth:
    def get_headers(self) -> dict:
        return {}


class _DummyDataset:
    def __init__(self, name: str):
        self.name = name
        self.id = f"id-{name}"


def _wire(monkeypatch, response: _FakeResponse) -> _FakeSession:
    session = _FakeSession(response)
    monkeypatch.setattr(api_datasets, "get_auth", lambda: _DummyAuth())
    monkeypatch.setattr(api_datasets, "get_current_org", lambda: "org-1")
    monkeypatch.setattr(
        api_datasets.Dataset,
        "get_by_name",
        classmethod(lambda cls, name, non_exist_ok=False: _DummyDataset(name)),
    )
    monkeypatch.setattr(api_datasets, "thread_local_session", lambda: session)
    return session


def test_merge_datasets_surfaces_backend_error(monkeypatch):
    """A failed merge surfaces the backend message without doubling the prefix."""
    # Arrange
    detail = "Failed to merge datasets: Dataset not found."
    _wire(monkeypatch, _FakeResponse(404, {"detail": {"error": detail}}))

    # Act
    with pytest.raises(DatasetError) as exc_info:
        api_datasets.merge_datasets("merged", ["a", "b"])

    # Assert
    assert str(exc_info.value) == detail


def test_merge_datasets_falls_back_to_status_when_no_detail(monkeypatch):
    """A failed merge without a parseable detail falls back to status and reason."""
    # Arrange
    _wire(monkeypatch, _FakeResponse(500, {"detail": {}}, reason="Server Error"))

    # Act
    with pytest.raises(DatasetError) as exc_info:
        api_datasets.merge_datasets("merged", ["a", "b"])

    # Assert
    assert str(exc_info.value) == "500 Server Error"


def test_merge_datasets_success_returns_dataset(monkeypatch):
    """A successful merge posts resolved source ids and returns the new dataset."""
    # Arrange
    session = _wire(monkeypatch, _FakeResponse(200, {"id": "new-id"}))

    class _FakeModel:
        id = "new-id"
        name = "merged"
        size_bytes = 10
        tags: list[str] = []
        is_shared = False
        all_data_types: dict = {}

    class _FakeDataset:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def get_by_name(cls, name, non_exist_ok=False):
            return _DummyDataset(name)

    monkeypatch.setattr(
        api_datasets.DatasetModel,
        "model_validate",
        staticmethod(lambda payload: _FakeModel()),
    )
    monkeypatch.setattr(api_datasets, "Dataset", _FakeDataset)

    # Act
    result = api_datasets.merge_datasets("merged", ["a", "b"])

    # Assert
    assert result.id == "new-id"
    assert session.last_kwargs["json"] == {
        "name": "merged",
        "sourceDatasetIds": ["id-a", "id-b"],
    }


def test_clone_dataset_surfaces_backend_error(monkeypatch):
    """A failed clone surfaces the backend message without doubling the prefix."""
    # Arrange
    detail = "Failed to clone dataset: Dataset not found."
    _wire(monkeypatch, _FakeResponse(404, {"detail": {"error": detail}}))

    # Act
    with pytest.raises(DatasetError) as exc_info:
        api_datasets.clone_dataset(
            "clone", source_dataset=_DummyDataset("src"), wait=False
        )

    # Assert
    assert str(exc_info.value) == detail


def test_clone_dataset_falls_back_to_status_when_no_detail(monkeypatch):
    """A failed clone without a parseable detail falls back to status and reason."""
    # Arrange
    _wire(monkeypatch, _FakeResponse(500, {"detail": {}}, reason="Server Error"))

    # Act
    with pytest.raises(DatasetError) as exc_info:
        api_datasets.clone_dataset(
            "clone", source_dataset=_DummyDataset("src"), wait=False
        )

    # Assert
    assert str(exc_info.value) == "500 Server Error"


def test_clone_dataset_success_returns_dataset(monkeypatch):
    """A successful clone posts the source id and returns the new dataset."""
    # Arrange
    session = _wire(monkeypatch, _FakeResponse(200, {"id": "clone-id"}))

    class _FakeModel:
        id = "clone-id"
        name = "clone"
        size_bytes = 10
        tags: list[str] = []
        is_shared = False
        description = ""
        all_data_types: dict = {}

    class _FakeDataset:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        api_datasets.DatasetModel,
        "model_validate",
        staticmethod(lambda payload: _FakeModel()),
    )
    monkeypatch.setattr(api_datasets, "Dataset", _FakeDataset)

    # Act
    result = api_datasets.clone_dataset(
        "clone", source_dataset=_DummyDataset("src"), wait=False
    )

    # Assert
    assert result.id == "clone-id"
    assert session.last_kwargs["json"] == {
        "name": "clone",
        "sourceDatasetId": "id-src",
    }
