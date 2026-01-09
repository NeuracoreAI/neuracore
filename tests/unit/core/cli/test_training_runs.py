"""Tests for training runs CLI commands."""

import json

import pytest
from typer.testing import CliRunner

from neuracore.core.cli.app import app
from neuracore.core.cli.training_runs import (
    _format_duration,
    _format_robot_data_spec,
    _format_timestamp,
    _get_model_artifact_path,
    get_training_run,
    list_training_runs,
)
from neuracore.core.const import API_URL
from neuracore.core.exceptions import TrainingRunError

runner = CliRunner()

MOCKED_ORG_ID = "test-org-id"


@pytest.fixture
def training_jobs_response():
    """Create a mock training jobs list response."""
    return [
        {
            "id": "job_123",
            "name": "training_run_1",
            "dataset_id": "dataset_123",
            "synced_dataset_id": "synced_123",
            "algorithm": "cnnmlp",
            "algorithm_id": "algo_123",
            "status": "COMPLETED",
            "cloud_compute_job_id": "compute_123",
            "zone": "us-central1-a",
            "launch_time": 1704067200.0,  # 2024-01-01 00:00:00
            "start_time": 1704067260.0,  # 2024-01-01 00:01:00
            "end_time": 1704070800.0,  # 2024-01-01 01:00:00
            "epoch": 100,
            "step": 5000,
            "algorithm_config": {
                "hidden_dim": 512,
                "num_layers": 3,
            },
            "gpu_type": "NVIDIA_TESLA_T4",
            "num_gpus": 1,
            "resumed_at": None,
            "previous_training_time": None,
            "error": None,
            "resume_points": [1704070000.0, 1704070500.0],
            "input_robot_data_spec": {
                "robot_1": {
                    "RGB_IMAGES": ["front_camera", "side_camera"],
                    "JOINT_POSITIONS": ["joint_1", "joint_2"],
                }
            },
            "output_robot_data_spec": {
                "robot_1": {
                    "JOINT_TARGET_POSITIONS": ["joint_1", "joint_2"],
                }
            },
            "synchronization_details": {
                "frequency": 10,
                "max_delay_s": 0.5,
                "allow_duplicates": True,
                "robot_data_spec": {},
            },
        },
        {
            "id": "job_456",
            "name": "training_run_2",
            "dataset_id": "dataset_456",
            "synced_dataset_id": None,
            "algorithm": "act",
            "algorithm_id": "algo_456",
            "status": "RUNNING",
            "cloud_compute_job_id": "compute_456",
            "zone": "us-west1-b",
            "launch_time": 1704153600.0,  # 2024-01-02 00:00:00
            "start_time": 1704153660.0,
            "end_time": None,
            "epoch": 50,
            "step": 2500,
            "algorithm_config": {},
            "gpu_type": "NVIDIA_TESLA_A100",
            "num_gpus": 2,
            "resumed_at": None,
            "previous_training_time": None,
            "error": None,
            "resume_points": [],
            "input_robot_data_spec": {},
            "output_robot_data_spec": {},
            "synchronization_details": {
                "frequency": 20,
                "max_delay_s": 1e20,
                "allow_duplicates": False,
                "robot_data_spec": {},
            },
        },
        {
            "id": "job_789",
            "name": "failed_run",
            "dataset_id": "dataset_789",
            "synced_dataset_id": None,
            "algorithm": "diffusion_policy",
            "algorithm_id": "algo_789",
            "status": "FAILED",
            "cloud_compute_job_id": None,
            "zone": None,
            "launch_time": 1704240000.0,  # 2024-01-03 00:00:00
            "start_time": None,
            "end_time": None,
            "epoch": -1,
            "step": -1,
            "algorithm_config": {},
            "gpu_type": "NVIDIA_TESLA_T4",
            "num_gpus": 1,
            "resumed_at": None,
            "previous_training_time": None,
            "error": "Out of memory",
            "resume_points": [],
            "input_robot_data_spec": {},
            "output_robot_data_spec": {},
            "synchronization_details": {
                "frequency": 10,
                "max_delay_s": 1.0,
                "allow_duplicates": True,
                "robot_data_spec": {},
            },
        },
    ]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_timestamp_with_value(self):
        """Test formatting a valid timestamp."""
        timestamp = 1704067200.0  # 2024-01-01 00:00:00 UTC
        result = _format_timestamp(timestamp)
        # Should return a formatted date string
        assert "2024" in result or "2023" in result  # Depends on timezone

    def test_format_timestamp_none(self):
        """Test formatting None timestamp."""
        result = _format_timestamp(None)
        assert result == "N/A"

    def test_format_duration_both_values(self):
        """Test formatting duration with both timestamps."""
        start = 1704067200.0
        end = 1704070800.0  # 1 hour later
        result = _format_duration(start, end)
        assert "1h" in result

    def test_format_duration_short(self):
        """Test formatting short duration."""
        start = 1704067200.0
        end = 1704067230.0  # 30 seconds later
        result = _format_duration(start, end)
        assert "30s" in result

    def test_format_duration_none_start(self):
        """Test formatting duration with None start time."""
        result = _format_duration(None, 1704067200.0)
        assert result == "N/A"

    def test_format_duration_none_end(self):
        """Test formatting duration with None end time."""
        result = _format_duration(1704067200.0, None)
        assert result == "N/A"

    def test_format_robot_data_spec_empty(self):
        """Test formatting empty robot data spec."""
        result = _format_robot_data_spec({})
        assert "(none)" in result

    def test_format_robot_data_spec_with_data(self):
        """Test formatting robot data spec with data."""
        spec = {
            "robot_1": {
                "RGB_IMAGES": ["cam1", "cam2"],
                "JOINT_POSITIONS": ["j1"],
            }
        }
        result = _format_robot_data_spec(spec)
        assert "robot_1" in result
        assert "RGB_IMAGES" in result
        assert "cam1" in result
        assert "cam2" in result

    def test_get_model_artifact_path(self):
        """Test model artifact path generation."""
        path = _get_model_artifact_path("org_123", "job_456")
        assert path == "organizations/org_123/training/job_456/model.nc.zip"


class TestListTrainingRunsCLI:
    """Tests for the list-training-runs CLI command."""

    def test_cli_help_shows_options(self):
        """Test that help shows all options."""
        result = runner.invoke(
            app,
            ["list-training-runs", "--help"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "--status" in result.output
        assert "--limit" in result.output
        assert "--verbose" in result.output

    def test_list_training_runs_no_auth(self, temp_config_dir, reset_neuracore):
        """Test list-training-runs without authentication."""
        result = runner.invoke(
            app,
            ["list-training-runs"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        # Should fail due to no auth
        assert result.exit_code == 1 or "Authentication" in result.output

    def test_list_training_runs_success(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test successful list of training runs."""
        import neuracore as nc

        nc.login("test_api_key")

        # Mock training jobs endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["list-training-runs"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "training_run_1" in result.output or "job_123" in result.output

    def test_list_training_runs_with_status_filter(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test list with status filter."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["list-training-runs", "--status", "COMPLETED"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        # Should show completed jobs
        assert "COMPLETED" in result.output or "training_run_1" in result.output

    def test_list_training_runs_with_limit(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test list with limit option."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["list-training-runs", "-n", "1"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "1 training run" in result.output

    def test_list_training_runs_empty(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mocked_org_id,
    ):
        """Test list when no training runs exist."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=[],
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["list-training-runs"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "No training runs found" in result.output


class TestInspectTrainingRunCLI:
    """Tests for the inspect-training-run CLI command."""

    def test_cli_help_shows_options(self):
        """Test that help shows all options."""
        result = runner.invoke(
            app,
            ["inspect-training-run", "--help"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--json" in result.output

    def test_inspect_training_run_success(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test successful inspection of a training run."""
        import neuracore as nc

        nc.login("test_api_key")

        job = training_jobs_response[0]
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123",
            json=job,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["inspect-training-run", "job_123"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "training_run_1" in result.output
        assert "COMPLETED" in result.output
        assert "cnnmlp" in result.output
        # Check for input/output data spec sections
        assert "Input Data Spec" in result.output
        assert "Output Data Spec" in result.output
        # Check for artifact path
        assert "model.nc.zip" in result.output

    def test_inspect_training_run_json_output(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test JSON output option."""
        import neuracore as nc

        nc.login("test_api_key")

        job = training_jobs_response[0]
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123",
            json=job,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["inspect-training-run", "job_123", "--json"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        # Should be valid JSON
        parsed = json.loads(result.output)
        assert parsed["id"] == "job_123"
        assert parsed["name"] == "training_run_1"

    def test_inspect_training_run_with_config(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test showing algorithm configuration."""
        import neuracore as nc

        nc.login("test_api_key")

        job = training_jobs_response[0]
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123",
            json=job,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["inspect-training-run", "job_123", "--config"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "Algorithm Configuration" in result.output
        assert "hidden_dim" in result.output

    def test_inspect_training_run_not_found(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mocked_org_id,
    ):
        """Test inspection of non-existent training run."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/nonexistent",
            status_code=404,
            json={"detail": "Training job not found"},
        )

        result = runner.invoke(
            app,
            ["inspect-training-run", "nonexistent"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_inspect_training_run_with_error(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test inspection of a failed training run shows error."""
        import neuracore as nc

        nc.login("test_api_key")

        job = training_jobs_response[2]  # Failed job
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/job_789",
            json=job,
            status_code=200,
        )

        result = runner.invoke(
            app,
            ["inspect-training-run", "job_789"],
            color=False,
            env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
        )
        assert result.exit_code == 0
        assert "FAILED" in result.output
        assert "Out of memory" in result.output


class TestLibraryFunctions:
    """Tests for the library functions (non-CLI)."""

    def test_list_training_runs_function(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test the list_training_runs library function."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        jobs = list_training_runs()
        assert len(jobs) == 3
        # Should be sorted by launch_time descending
        assert jobs[0]["id"] == "job_789"  # Most recent

    def test_list_training_runs_with_filter(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test filtering by status."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        jobs = list_training_runs(status_filter="RUNNING")
        assert len(jobs) == 1
        assert jobs[0]["status"] == "RUNNING"

    def test_list_training_runs_with_limit(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test limiting results."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        jobs = list_training_runs(limit=2)
        assert len(jobs) == 2

    def test_get_training_run_function(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test the get_training_run library function."""
        import neuracore as nc

        nc.login("test_api_key")

        job = training_jobs_response[0]
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123",
            json=job,
            status_code=200,
        )

        result = get_training_run("job_123")
        assert result["id"] == "job_123"
        assert result["name"] == "training_run_1"

    def test_get_training_run_not_found(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mocked_org_id,
    ):
        """Test get_training_run with non-existent job."""
        import neuracore as nc

        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/nonexistent",
            status_code=404,
            json={"detail": "Training job not found"},
        )

        with pytest.raises(TrainingRunError, match="not found"):
            get_training_run("nonexistent")

    def test_connection_error(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mocked_org_id,
    ):
        """Test handling of connection errors."""
        import requests
        from unittest.mock import patch

        import neuracore as nc

        nc.login("test_api_key")

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()

            with pytest.raises(TrainingRunError, match="connect"):
                list_training_runs()
