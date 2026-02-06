"""Tests for training run utilities."""

import builtins
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import requests
from neuracore_types import TrainingJobStatus

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.exceptions import TrainingRunError
from neuracore.ml.cli.training_runs_cloud import (
    _format_duration,
    _format_robot_data_spec,
    _format_timestamp,
    _get_model_artifact_path,
    get_training_run,
    list_training_runs,
)

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
        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        jobs = list_training_runs()
        assert len(jobs) == 3
        # Should be sorted by launch_time descending
        assert jobs[0].id == "job_789"  # Most recent

    def test_list_training_runs_with_filter(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test filtering by status."""
        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs",
            json=training_jobs_response,
            status_code=200,
        )

        jobs = list_training_runs(status_filter="RUNNING")
        assert len(jobs) == 1
        assert jobs[0].status == TrainingJobStatus.RUNNING

    def test_list_training_runs_with_limit(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        training_jobs_response,
        mocked_org_id,
    ):
        """Test limiting results."""
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
        nc.login("test_api_key")

        job = training_jobs_response[0]
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123",
            json=job,
            status_code=200,
        )

        result = get_training_run("job_123")
        assert result.id == "job_123"
        assert result.name == "training_run_1"

    def test_get_training_run_not_found(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mocked_org_id,
    ):
        """Test get_training_run with non-existent job."""
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
        nc.login("test_api_key")

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()

            with pytest.raises(TrainingRunError, match="connect"):
                list_training_runs()


def drop_cached(modules: list[str]) -> dict[str, ModuleType]:
    """Drop cached modules and return the ones that were removed."""
    cached: dict[str, ModuleType] = {}

    for key in modules:
        mod = sys.modules.pop(key, None)
        if mod is not None:
            cached[key] = mod

    return cached


class TestDependencyBoundaries:
    """Ensure optional ML deps don't break core CLI or pull heavy deps."""

    def test_cli_app_handles_missing_ml_dependencies(self, monkeypatch):
        """Import core CLI even when ML/omegaconf is absent."""

        # Force import failure for omegaconf to simulate missing ml extras.
        real_import = builtins.__import__

        # Drop cached modules so import path runs with the patched importer.
        cached = drop_cached([
            "neuracore.core.cli.training_commands",
            "neuracore.core.cli.app",
        ])

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "omegaconf":
                raise ModuleNotFoundError("No module named 'omegaconf'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        try:
            import neuracore.core.cli.app as app

            assert app._training_app is None
            assert app._training_import_error is not None
            assert any(cmd.name == "training" for cmd in app.app.registered_commands)

            with pytest.raises(SystemExit):
                app.training_placeholder()
        finally:
            # Remove modules created during the test if they weren't present before.
            if "neuracore.core.cli.app" not in cached:
                sys.modules.pop("neuracore.core.cli.app", None)
            if "neuracore.core.cli.training_commands" not in cached:
                sys.modules.pop("neuracore.core.cli.training_commands", None)
            # Restore any cached modules we removed to avoid affecting other tests.
            sys.modules.update(cached)

    def test_ml_training_runs_cli_does_not_pull_torch(self, monkeypatch):
        """Guard against torch becoming an import-time dependency of the CLI."""

        real_import = builtins.__import__

        drop_cached(["neuracore.ml.cli.training_runs_cloud"])

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise AssertionError("torch should not be imported by CLI utilities")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Reload module under patched importer to catch accidental torch imports.
        import neuracore.ml.cli.training_runs_cloud as ml_mod

        # Basic sanity check to ensure module still loaded.
        assert hasattr(ml_mod, "list_training_runs")
