"""Tests for the Dataset class."""

import pytest

from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset


@pytest.mark.usefixtures("mock_auth_requests")
@pytest.mark.usefixtures("mock_login")
class TestDataset:

    def test_create_dataset(self, dataset_dict):
        dataset = Dataset.create(name=dataset_dict["name"])
        assert isinstance(dataset, Dataset)
        assert dataset.name == dataset_dict["name"]
        length = len(dataset)
        assert dataset._num_recordings == length

    def test_get_by_id(self, dataset_dict):

        ds = Dataset.get_by_id(dataset_dict["id"])
        assert ds.id == dataset_dict["id"]
        assert ds.name == dataset_dict["name"]

    def test_get_by_name(self, dataset_dict):
        ds = Dataset.get_by_name(dataset_dict["name"])
        assert ds.name == dataset_dict["name"]
        assert ds.id == dataset_dict["id"]

    def test_iteration(self, dataset_dict, recordings_list):
        ds = Dataset(**dataset_dict, recordings=recordings_list)
        recs = list(ds)
        assert all(isinstance(r, Recording) for r in recs)
        assert recs[0].id == recordings_list[0]["id"]

        iterator = iter(ds)
        rec_next = next(iterator)
        assert rec_next.id == recordings_list[0]["id"]
        rec_next = next(iterator)
        assert rec_next.id == recordings_list[1]["id"]
        with pytest.raises(StopIteration):
            next(iterator)

    def test_len(self, dataset_dict, recordings_list):
        ds = Dataset(**dataset_dict, recordings=recordings_list)
        assert len(ds) == len(recordings_list)

    def test_synchronize(self, dataset_dict):
        ds = Dataset(**dataset_dict)
        synced_ds = ds.synchronize()
        assert isinstance(synced_ds, SynchronizedDataset)
        assert synced_ds.dataset.id == ds.id

    def test_lazy_loading_of_recordings(self, dataset_dict):
        dataset = Dataset(**dataset_dict)
        assert len(dataset) == dataset._num_recordings
