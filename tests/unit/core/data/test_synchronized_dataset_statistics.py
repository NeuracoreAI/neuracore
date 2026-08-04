"""Tests for SynchronizedDataset.calculate_statistics against an async backend."""

import re
from unittest.mock import patch

import pytest
import requests
from neuracore_types import DatasetStatisticsJobStatus, DataType

from neuracore.core.const import API_URL
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.exceptions import DatasetError

SYNCED_DATASET_ID = "synced_dataset_id"
JOB_ID = "synced_dataset_id-abc123def456"
CROSS_EMBODIMENT = {"robot": {DataType.JOINT_POSITIONS: {0: "j"}}}


def job_json(
    status: DatasetStatisticsJobStatus,
    num_completed: int = 0,
    total: int = 2,
    error: str | None = None,
) -> dict:
    """Build a statistics job payload."""
    return {
        "job_id": JOB_ID,
        "synchronized_dataset_id": SYNCED_DATASET_ID,
        "status": status.value,
        "num_recordings": total,
        "num_completed_recordings": num_completed,
        "error": error,
    }


RESULT_JSON = {
    "synchronized_dataset_id": SYNCED_DATASET_ID,
    "input_cross_embodiment_description": {"robot": {"JOINT_POSITIONS": {"0": "j"}}},
    "output_cross_embodiment_description": {"robot": {"JOINT_POSITIONS": {"0": "j"}}},
    "dataset_statistics": {"input": {}, "output": {}},
}


@pytest.mark.usefixtures("mock_login")
class TestCalculateStatistics:
    """Tests for the start, poll and fetch flow."""

    @pytest.fixture(autouse=True)
    def no_poll_delay(self, monkeypatch):
        """Remove the poll delay so the loop runs at full speed."""
        monkeypatch.setattr(
            "neuracore.core.data.synced_dataset.DATASET_STATISTICS_POLL_INTERVAL_S", 0
        )

    @pytest.fixture
    def synced_dataset(self, dataset_dict, recordings_list, tmp_path):
        """Create a SynchronizedDataset instance for testing."""
        from neuracore.core.data.dataset import Dataset

        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        dataset.cache_dir = tmp_path / "cache"
        dataset.cache_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(SynchronizedDataset, "_perform_synced_data_prefetch"):
            return SynchronizedDataset(
                id=SYNCED_DATASET_ID,
                dataset=dataset,
                frequency=30,
                cross_embodiment_union=None,
                prefetch_videos=False,
            )

    @pytest.fixture
    def endpoints(self, mocked_org_id):
        """The three statistics endpoints."""
        base = f"{API_URL}/org/{mocked_org_id}/synchronized-dataset"
        return {
            "start": f"{base}/calculate-dataset-statistics",
            "progress": f"{base}/dataset-statistics-progress/{JOB_ID}",
            "result": f"{base}/dataset-statistics/{JOB_ID}",
        }

    def calculate(self, synced_dataset: SynchronizedDataset):
        """Run calculate_statistics with both cross-embodiment descriptions."""
        return synced_dataset.calculate_statistics(
            input_cross_embodiment_description=CROSS_EMBODIMENT,
            output_cross_embodiment_description=CROSS_EMBODIMENT,
        )

    def test_warm_job_skips_polling_and_shows_no_progress_bar(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """An already-complete job returns its result without any noise."""
        mock_data_requests.post(
            endpoints["start"],
            json=job_json(DatasetStatisticsJobStatus.COMPLETE, num_completed=2),
        )
        progress = mock_data_requests.get(
            endpoints["progress"], json=job_json(DatasetStatisticsJobStatus.COMPLETE, 2)
        )
        result = mock_data_requests.get(endpoints["result"], json=RESULT_JSON)

        statistics = self.calculate(synced_dataset)

        assert statistics.synchronized_dataset_id == SYNCED_DATASET_ID
        assert progress.call_count == 0
        assert result.call_count == 1

    def test_polls_until_complete_then_fetches_the_result(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """The client polls through the aggregate stage before fetching."""
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 0)
        )
        progress = mock_data_requests.get(
            endpoints["progress"],
            [
                {"json": job_json(DatasetStatisticsJobStatus.RUNNING, 1)},
                {"json": job_json(DatasetStatisticsJobStatus.AGGREGATING, 2)},
                {"json": job_json(DatasetStatisticsJobStatus.COMPLETE, 2)},
            ],
        )
        result = mock_data_requests.get(endpoints["result"], json=RESULT_JSON)

        statistics = self.calculate(synced_dataset)

        assert statistics.synchronized_dataset_id == SYNCED_DATASET_ID
        assert progress.call_count == 3
        assert result.call_count == 1

    def test_completion_keys_off_status_not_the_counter(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """The aggregate stage runs after the counter saturates."""
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 0)
        )
        progress = mock_data_requests.get(
            endpoints["progress"],
            [
                {"json": job_json(DatasetStatisticsJobStatus.RUNNING, 2)},
                {"json": job_json(DatasetStatisticsJobStatus.AGGREGATING, 2)},
                {"json": job_json(DatasetStatisticsJobStatus.COMPLETE, 2)},
            ],
        )
        result = mock_data_requests.get(endpoints["result"], json=RESULT_JSON)

        self.calculate(synced_dataset)

        assert progress.call_count == 3
        assert result.call_count == 1

    def test_job_failure_surfaces_the_backend_message_verbatim(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """A 422 breaks the poll loop with the server's explanation."""
        message = (
            "The dataset changed while statistics were being calculated. "
            "Calculating them again will use the dataset's current recordings."
        )
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 0)
        )
        mock_data_requests.get(
            endpoints["progress"],
            status_code=422,
            json={"detail": {"error": message, "status": 422}},
        )

        with pytest.raises(DatasetError, match=re.escape(message)):
            self.calculate(synced_dataset)

    def test_failed_status_in_the_body_raises(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """A FAILED job is fatal even when the response is a 200."""
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 0)
        )
        mock_data_requests.get(
            endpoints["progress"],
            json=job_json(DatasetStatisticsJobStatus.FAILED, 1, error="boom"),
        )

        with pytest.raises(DatasetError, match="boom"):
            self.calculate(synced_dataset)

    def test_transient_progress_failures_are_retried(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """A saturated backend must not discard minutes of waiting."""
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 0)
        )
        mock_data_requests.get(
            endpoints["progress"],
            [
                {"status_code": 503, "json": {}},
                {"exc": requests.exceptions.ConnectionError},
                {"json": job_json(DatasetStatisticsJobStatus.COMPLETE, 2)},
            ],
        )
        result = mock_data_requests.get(endpoints["result"], json=RESULT_JSON)

        statistics = self.calculate(synced_dataset)

        assert statistics.synchronized_dataset_id == SYNCED_DATASET_ID
        assert result.call_count == 1

    def test_a_missing_job_is_not_retried(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """Polling cannot resolve a job the server does not have."""
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 0)
        )
        progress = mock_data_requests.get(
            endpoints["progress"],
            status_code=404,
            json={"detail": {"error": "Statistics job not found", "status": 404}},
        )

        with pytest.raises(DatasetError, match="Statistics job not found"):
            self.calculate(synced_dataset)

        assert progress.call_count == 1

    def test_deadline_exceeded_reports_the_job(
        self, mock_data_requests, synced_dataset, endpoints, monkeypatch
    ):
        """A wedged job fails with enough context to resume it."""
        monkeypatch.setattr(
            "neuracore.core.data.synced_dataset.DATASET_STATISTICS_TIMEOUT_S", 0
        )
        mock_data_requests.post(
            endpoints["start"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 1)
        )
        mock_data_requests.get(
            endpoints["progress"], json=job_json(DatasetStatisticsJobStatus.RUNNING, 1)
        )

        with pytest.raises(DatasetError) as error:
            self.calculate(synced_dataset)

        assert JOB_ID in str(error.value)
        assert "1/2" in str(error.value)
        assert "resumes" in str(error.value)

    def test_start_failure_surfaces_the_backend_message(
        self, mock_data_requests, synced_dataset, endpoints
    ):
        """A rejected start is reported rather than polled for."""
        mock_data_requests.post(
            endpoints["start"],
            status_code=404,
            json={"detail": {"error": "Dataset not found", "status": 404}},
        )

        with pytest.raises(DatasetError, match="Dataset not found"):
            self.calculate(synced_dataset)
