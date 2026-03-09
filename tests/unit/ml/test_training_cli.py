"""Tests for the training CLI group (local and cloud)."""

import pytest
from typer.testing import CliRunner

import neuracore as nc
from neuracore.core.cli.app import app
from neuracore.core.const import API_URL

runner = CliRunner()


@pytest.fixture
def sample_training_jobs_response():
    """Sample cloud training jobs for list command."""
    return [
        {
            "id": "job_123",
            "name": "training_run_1",
            "status": "COMPLETED",
            "launch_time": 1704067200.0,
            "algorithm": "cnnmlp",
            "dataset_id": "dataset_123",
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
            "status": "RUNNING",
            "launch_time": 1704153600.0,
            "algorithm": "act",
            "dataset_id": "dataset_456",
            "synchronization_details": {
                "frequency": 20,
                "max_delay_s": 1.0,
                "allow_duplicates": False,
                "robot_data_spec": {},
            },
        },
    ]


def test_training_local_list(tmp_path):
    """Local list shows runs with success flag derived from train.log."""
    run_ok = tmp_path / "run_ok"
    run_ok.mkdir()
    (run_ok / "train.log").write_text("... Training completed successfully!")
    hydra_dir_ok = run_ok / ".hydra"
    hydra_dir_ok.mkdir()
    (hydra_dir_ok / "config.yaml").write_text(
        "algorithm_id: algo-123\ndataset_name: dataset-alpha\n"
    )

    run_bad = tmp_path / "run_bad"
    run_bad.mkdir()
    (run_bad / "train.log").write_text("something went wrong")
    hydra_dir_bad = run_bad / ".hydra"
    hydra_dir_bad.mkdir()
    (hydra_dir_bad / "config.yaml").write_text(
        "algorithm_id: algo-456\ndataset_id: dataset-beta\n"
    )

    result = runner.invoke(
        app,
        ["training", "list", "--local", "--root", str(tmp_path)],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "run_ok" in result.output
    assert "Yes" in result.output
    assert "run_bad" in result.output
    assert "No" in result.output
    assert "algo-123" in result.output
    assert "dataset-alpha" in result.output
    assert "algo-456" in result.output
    assert "dataset-beta" in result.output


def test_training_local_list_when_empty(tmp_path):
    """Gracefully handle empty root for local runs."""
    result = runner.invoke(
        app,
        ["training", "list", "--local", "--root", str(tmp_path)],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "No local training runs found" in result.output


def test_training_cloud_list(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
    sample_training_jobs_response,
):
    """Cloud list fetches jobs from API and shows success flag."""
    nc.login("test_api_key")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=sample_training_jobs_response,
        status_code=200,
    )

    result = runner.invoke(
        app,
        ["training", "list", "--cloud"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "training_run_1" in result.output
    assert "Yes" in result.output
    assert "training_run_2" in result.output
    assert "No" in result.output
    assert "cnnmlp" in result.output
    assert "dataset_123" in result.output


def test_training_local_inspect(tmp_path):
    """Inspect a local run using saved metadata."""
    run_dir = tmp_path / "run_local"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_run.json").write_text(
        """
{
  "id": "run_local",
  "name": "run_local",
  "status": "RUNNING",
  "algorithm": "cnnmlp",
  "dataset_id": "dataset_123",
  "launch_time": 1704067200.0,
  "local_output_dir": "%s",
  "epoch": -1,
  "step": -1,
  "gpu_type": "NVIDIA_TESLA_T4",
  "num_gpus": 1,
  "synchronization_details": {
    "frequency": 10,
    "max_delay_s": null,
    "allow_duplicates": null
  },
  "input_robot_data_spec": {"robot_1": {"RGB_IMAGES": ["cam1"]}},
  "output_robot_data_spec": {"robot_1": {"JOINT_TARGET_POSITIONS": ["joint1"]}},
  "frequency": 10,
  "output_prediction_horizon": 5
}
"""
        % str(run_dir)
    )

    result = runner.invoke(
        app,
        [
            "training",
            "inspect",
            "--local",
            "--root",
            str(tmp_path),
            "--training-name",
            "run_local",
        ],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Local Training Run" in result.output
    assert "cnnmlp" in result.output
    assert "dataset_123" in result.output
    assert "RGB_IMAGES" in result.output
    assert "JOINT_TARGET_POSITIONS" in result.output
    assert "Hardware" in result.output
    assert "Synchronization Details" in result.output


def test_training_list_defaults_to_all(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
    sample_training_jobs_response,
    tmp_path,
):
    """Default list with no flags shows both sections when available."""
    # local run
    run_ok = tmp_path / "run_ok"
    run_ok.mkdir()
    (run_ok / "train.log").write_text("... Training completed successfully!")
    (run_ok / ".hydra").mkdir()
    (run_ok / ".hydra" / "config.yaml").write_text(
        "algorithm_id: algo-123\ndataset_name: dataset-alpha\n"
    )

    nc.login("test_api_key")
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=sample_training_jobs_response,
        status_code=200,
    )

    result = runner.invoke(
        app,
        ["training", "list", "--root", str(tmp_path)],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Local Training Runs" in result.output
    assert "Cloud Training Runs" in result.output


def test_training_local_delete(tmp_path):
    """Delete a local training run directory."""
    run_dir = tmp_path / "run_local"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_run.json").write_text('{"name": "run_local"}')

    result = runner.invoke(
        app,
        [
            "training",
            "delete",
            "--local",
            "--root",
            str(tmp_path),
            "--training-name",
            "run_local",
            "--yes",
        ],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Deleted local training run 'run_local'" in result.output
    assert not run_dir.exists()


def test_training_cloud_delete(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
    sample_training_jobs_response,
):
    """Delete a cloud training run by name."""
    nc.login("test_api_key")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=sample_training_jobs_response,
        status_code=200,
    )
    mock_auth_requests.delete(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123", status_code=204
    )

    result = runner.invoke(
        app,
        ["training", "delete", "--training-name", "training_run_1", "--yes"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Deleted cloud training run 'training_run_1' (job_123)." in result.output
    assert any(
        req.method == "DELETE" and req.url.endswith("/job_123")
        for req in mock_auth_requests.request_history
    )
