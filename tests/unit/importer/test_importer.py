import pytest

from neuracore.core.exceptions import DatasetError
from neuracore.importer.core.exceptions import DatasetOperationError
from neuracore.importer.importer import _create_dataset_with_overwrite_guard


def test_create_dataset_with_overwrite_guard_retries_if_old_id_returned(monkeypatch):
    class _Dataset:
        def __init__(self, dataset_id: str):
            self.id = dataset_id

    responses = [_Dataset("old-id"), _Dataset("old-id"), _Dataset("new-id")]

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "neuracore.importer.importer.nc.create_dataset",
        lambda **_kwargs: responses.pop(0),
    )
    monkeypatch.setattr(
        "neuracore.importer.importer.time.sleep",
        lambda interval: sleep_calls.append(interval),
    )

    result = _create_dataset_with_overwrite_guard(
        dataset_name="my-dataset",
        description="desc",
        tags=["tag"],
        shared=False,
        deleted_dataset_id="old-id",
        timeout_s=10.0,
        poll_interval_s=0.5,
    )

    assert result.id == "new-id"
    assert sleep_calls == [0.5, 0.5]


def test_create_dataset_with_overwrite_guard_raises_on_timeout(monkeypatch):
    class _Dataset:
        def __init__(self, dataset_id: str):
            self.id = dataset_id

    monkeypatch.setattr(
        "neuracore.importer.importer.nc.create_dataset",
        lambda **_kwargs: _Dataset("old-id"),
    )
    monkeypatch.setattr("neuracore.importer.importer.time.sleep", lambda *_args: None)

    monotonic_values = iter([0.0, 0.5, 1.1])
    monkeypatch.setattr(
        "neuracore.importer.importer.time.monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(
        DatasetOperationError,
        match="Timed out waiting to create replacement dataset 'my-dataset'.",
    ):
        _create_dataset_with_overwrite_guard(
            dataset_name="my-dataset",
            description=None,
            tags=None,
            shared=False,
            deleted_dataset_id="old-id",
            timeout_s=1.0,
            poll_interval_s=0.1,
        )


def test_create_dataset_with_overwrite_guard_retries_on_transient_create_error(
    monkeypatch,
):
    class _Dataset:
        def __init__(self, dataset_id: str):
            self.id = dataset_id

    responses = iter([
        DatasetError("Failed to create dataset: Dataset already exists"),
        _Dataset("new-id"),
    ])

    def fake_create_dataset(**_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "neuracore.importer.importer.nc.create_dataset",
        fake_create_dataset,
    )
    monkeypatch.setattr(
        "neuracore.importer.importer.time.sleep",
        lambda interval: sleep_calls.append(interval),
    )

    result = _create_dataset_with_overwrite_guard(
        dataset_name="my-dataset",
        description=None,
        tags=None,
        shared=False,
        deleted_dataset_id="old-id",
        timeout_s=10.0,
        poll_interval_s=0.5,
    )

    assert result.id == "new-id"
    assert sleep_calls == [0.5]
