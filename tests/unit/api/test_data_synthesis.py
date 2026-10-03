"""Tests for the data synthesis client API."""

import pytest
from neuracore_types import Dataset, DataType

import neuracore as nc
from neuracore.core.const import API_URL

TEST_DATASET_ID = "dataset123"


@pytest.fixture
def job_response() -> dict:
    return {
        "id": "synth_job_123",
        "name": "expand tabletop",
        "dataset_id": TEST_DATASET_ID,
        "output_dataset_name": "test_dataset (synthetic)",
        "algorithm": {
            "name": "procedural-in-painter",
            "parameters": {"segmentation_prompts": ["Table"]},
        },
        "scaling_factor": 2,
        "include_original": False,
        "status": "QUEUED",
        "launch_time": 1.0,
        "recordings_total": 0,
        "recordings_completed": 0,
        "deleted": False,
        "deleted_at": None,
    }


@pytest.fixture
def algorithm_specs() -> list[dict]:
    """What the service says is on offer.

    The client no longer holds a list of algorithms, so this is a response
    shape rather than anything the client knows.
    """
    return [{
        "name": "procedural-in-painter",
        "config_name": "procedural_in_painter",
        "title": "Procedural in-painting",
        "description": "Fills the objects a prompt names with a pattern.",
        "json_schema": {
            "properties": {"segmentation_prompts": {"type": "array"}},
            "required": ["segmentation_prompts"],
        },
    }]


@pytest.fixture
def mock_endpoints(mock_auth_requests, mocked_org_id, job_response, algorithm_specs):
    """Register the endpoints the data synthesis API walks through."""

    def register(
        job_status_code: int = 200,
        existing_jobs: list[dict] | None = None,
    ) -> str:
        dataset = Dataset(
            id=TEST_DATASET_ID,
            name="test_dataset",
            created_at=0.0,
            modified_at=0.0,
            size_bytes=2048,
            tags=[],
            is_shared=False,
            num_demonstrations=20,
            all_data_types={DataType.RGB_IMAGES: 1},
            common_data_types={DataType.RGB_IMAGES: 1},
        )
        org_url = f"{API_URL}/synthesis/org/{mocked_org_id}"
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json=dataset.model_dump(mode="json"),
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/jobs",
            json=existing_jobs if existing_jobs is not None else [],
            status_code=200,
        )
        mock_auth_requests.post(
            f"{org_url}/jobs",
            json=job_response,
            status_code=job_status_code,
        )
        mock_auth_requests.get(
            f"{org_url}/algorithms",
            json=algorithm_specs,
            status_code=200,
        )
        nc.login("test_api_key")
        return org_url

    return register


class TestAlgorithmListing:
    def test_lists_what_the_service_offers(self, mock_endpoints) -> None:
        # The client holds no list of its own: adding an algorithm to the
        # worker's package is meant to reach a notebook without a release of
        # this package.
        mock_endpoints()

        algorithms = nc.get_data_synthesis_algorithms()

        assert [algorithm["name"] for algorithm in algorithms] == [
            "procedural-in-painter"
        ]
        assert algorithms[0]["json_schema"]["required"] == ["segmentation_prompts"]


class TestStartRun:
    def test_starts_a_run_and_returns_the_job(self, mock_endpoints) -> None:
        mock_endpoints()

        job = nc.start_data_synthesis_run(
            name="expand tabletop",
            dataset_name="test_dataset",
            algorithm_name="procedural-in-painter",
            algorithm_config={"segmentation_prompts": ["Table"]},
        )

        assert job["id"] == "synth_job_123"

    def test_sends_the_resolved_dataset_id(
        self, mock_endpoints, mock_auth_requests
    ) -> None:
        mock_endpoints()

        nc.start_data_synthesis_run(
            name="expand tabletop",
            dataset_name="test_dataset",
            algorithm_name="procedural-in-painter",
            algorithm_config={"segmentation_prompts": ["Table"]},
            scaling_factor=3,
        )

        sent = mock_auth_requests.request_history[-1].json()
        assert sent["dataset_id"] == TEST_DATASET_ID
        assert sent["scaling_factor"] == 3
        assert sent["algorithm"]["name"] == "procedural-in-painter"
        assert sent["algorithm"]["parameters"] == {"segmentation_prompts": ["Table"]}
        # Synthesis never synchronizes, so no frequency is sent.
        assert "synchronization_details" not in sent

    def test_defaults_the_output_name_from_the_source(
        self, mock_endpoints, mock_auth_requests
    ) -> None:
        mock_endpoints()

        nc.start_data_synthesis_run(
            name="expand tabletop",
            dataset_name="test_dataset",
            algorithm_name="procedural-in-painter",
            algorithm_config={"segmentation_prompts": ["Table"]},
        )

        assert (
            mock_auth_requests.request_history[-1].json()["output_dataset_name"]
            == "test_dataset (synthetic)"
        )

    def test_auto_increments_a_name_already_in_use(
        self, mock_endpoints, mock_auth_requests, job_response
    ) -> None:
        mock_endpoints(existing_jobs=[{**job_response, "name": "expand"}])

        nc.start_data_synthesis_run(
            name="expand",
            dataset_name="test_dataset",
            algorithm_name="procedural-in-painter",
            algorithm_config={"segmentation_prompts": ["Table"]},
            name_auto_increment=True,
        )

        assert mock_auth_requests.request_history[-1].json()["name"] == "expand_1"

    def test_rejects_a_scaling_factor_that_is_not_a_whole_number_of_copies(
        self, mock_endpoints
    ) -> None:
        mock_endpoints()

        for bad in (0, -1, 11, 1.5):
            with pytest.raises(Exception):
                nc.start_data_synthesis_run(
                    name="expand",
                    dataset_name="test_dataset",
                    algorithm_name="procedural-in-painter",
                    algorithm_config={"segmentation_prompts": ["Table"]},
                    scaling_factor=bad,
                )

    def test_passes_an_algorithm_the_client_has_never_heard_of(
        self, mock_endpoints, mock_auth_requests
    ) -> None:
        # Whether an algorithm exists, and whether its parameters are valid, is
        # the service's to say. The client only has to carry them.
        mock_endpoints()

        nc.start_data_synthesis_run(
            name="expand",
            dataset_name="test_dataset",
            algorithm_name="greenaug",
            algorithm_config={"key_color": "auto"},
        )

        sent = mock_auth_requests.request_history[-1].json()
        assert sent["algorithm"] == {
            "name": "greenaug",
            "parameters": {"key_color": "auto"},
        }

    def test_surfaces_a_server_error_message(self, mock_endpoints) -> None:
        mock_endpoints(job_status_code=400)

        with pytest.raises(ValueError, match="Error starting data synthesis run"):
            nc.start_data_synthesis_run(
                name="expand",
                dataset_name="test_dataset",
                algorithm_name="procedural-in-painter",
                algorithm_config={"segmentation_prompts": ["Table"]},
            )


class TestJobQueries:
    def test_lists_jobs(self, mock_endpoints, job_response) -> None:
        mock_endpoints(existing_jobs=[job_response])

        assert [job["id"] for job in nc.get_data_synthesis_jobs()] == ["synth_job_123"]

    def test_reads_a_job_status(
        self, mock_endpoints, mock_auth_requests, mocked_org_id, job_response
    ) -> None:
        org_url = mock_endpoints()
        mock_auth_requests.get(
            f"{org_url}/jobs/synth_job_123",
            json={**job_response, "status": "RUNNING"},
            status_code=200,
        )

        assert nc.get_data_synthesis_job_status("synth_job_123") == "RUNNING"

    def test_reports_a_missing_job(
        self, mock_endpoints, mock_auth_requests, mocked_org_id
    ) -> None:
        org_url = mock_endpoints()
        mock_auth_requests.get(f"{org_url}/jobs/absent", json={}, status_code=404)

        with pytest.raises(ValueError, match="Job not found"):
            nc.get_data_synthesis_job_data("absent")


class TestLifecycleActions:
    def test_cancels_a_job(
        self, mock_endpoints, mock_auth_requests, mocked_org_id
    ) -> None:
        org_url = mock_endpoints()
        mock_auth_requests.post(
            f"{org_url}/jobs/synth_job_123/cancel",
            json={"status": "cancelled"},
            status_code=200,
        )

        nc.cancel_data_synthesis_job("synth_job_123")

    def test_deletes_a_job(
        self, mock_endpoints, mock_auth_requests, mocked_org_id
    ) -> None:
        org_url = mock_endpoints()
        mock_auth_requests.delete(
            f"{org_url}/jobs/synth_job_123",
            json={"status": "deleted"},
            status_code=200,
        )

        nc.delete_data_synthesis_job("synth_job_123")

    def test_surfaces_a_failed_cancel(
        self, mock_endpoints, mock_auth_requests, mocked_org_id
    ) -> None:
        org_url = mock_endpoints()
        mock_auth_requests.post(
            f"{org_url}/jobs/synth_job_123/cancel",
            json={"detail": "already finished"},
            status_code=409,
        )

        with pytest.raises(ValueError, match="Error cancelling"):
            nc.cancel_data_synthesis_job("synth_job_123")
