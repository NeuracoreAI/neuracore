"""Tests for TrainingStorageHandler."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from neuracore.core.const import API_URL
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler

ORG_ID = "test-org-id"
JOB_ID = "test-job-id"
BASE_JOB_URL = f"{API_URL}/org/{ORG_ID}/training/jobs/{JOB_ID}"
SIGNED_URL = "https://storage.example.com/signed-url"


@pytest.fixture
def mock_auth(monkeypatch):
    """Patch get_auth to return headers without real credentials."""
    mock = MagicMock()
    mock.get_headers.return_value = {"Authorization": "Bearer test-token"}
    monkeypatch.setattr(
        "neuracore.ml.utils.training_storage_handler.get_auth", lambda: mock
    )


@pytest.fixture
def mock_org(monkeypatch):
    """Patch get_current_org to return a fixed org ID."""
    monkeypatch.setattr(
        "neuracore.ml.utils.training_storage_handler.get_current_org",
        lambda: ORG_ID,
    )


@pytest.fixture
def handler(tmp_path, requests_mock, mock_auth, mock_org):
    """Create a cloud-enabled TrainingStorageHandler with mocked HTTP."""
    requests_mock.get(BASE_JOB_URL, json={}, status_code=200)
    return TrainingStorageHandler(
        local_dir=str(tmp_path),
        training_job_id=JOB_ID,
    )


@pytest.fixture
def local_handler(tmp_path, mock_auth, mock_org):
    """Create a local-only TrainingStorageHandler (no cloud)."""
    return TrainingStorageHandler(local_dir=str(tmp_path))


class TestInit:
    def test_init_with_job_id_verifies_job_exists_in_cloud(
        self, tmp_path, requests_mock, mock_auth, mock_org
    ):
        requests_mock.get(BASE_JOB_URL, json={}, status_code=200)
        cloud_handler = TrainingStorageHandler(
            local_dir=str(tmp_path), training_job_id=JOB_ID
        )
        assert cloud_handler.log_to_cloud is True
        assert cloud_handler.training_job_id == JOB_ID
        assert cloud_handler.org_id == ORG_ID

    def test_init_raises_value_error_when_job_not_found(
        self, tmp_path, requests_mock, mock_auth, mock_org
    ):
        requests_mock.get(BASE_JOB_URL, json={"detail": "Not found"}, status_code=404)
        with pytest.raises(ValueError, match=JOB_ID):
            TrainingStorageHandler(local_dir=str(tmp_path), training_job_id=JOB_ID)

    def test_init_without_job_id_disables_cloud_logging(
        self, tmp_path, mock_auth, mock_org
    ):
        local_only_handler = TrainingStorageHandler(local_dir=str(tmp_path))
        assert local_only_handler.log_to_cloud is False
        assert local_only_handler.training_job_id is None

    def test_init_sets_local_dir_to_output_when_none_given(self, mock_auth, mock_org):
        handler_with_default_dir = TrainingStorageHandler(local_dir=None)
        assert handler_with_default_dir.local_dir == Path("./output")


class TestGetUploadUrl:
    def test_get_upload_url_returns_signed_url_from_response(
        self, handler, requests_mock
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url",
            json={"url": SIGNED_URL},
            status_code=200,
        )
        url = handler._get_upload_url("checkpoints/ckpt.pt", "application/octet-stream")
        assert url == SIGNED_URL

    def test_get_upload_url_raises_value_error_when_response_is_not_200(
        self, handler, requests_mock
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url",
            json={"detail": "Forbidden"},
            status_code=403,
        )
        with pytest.raises(ValueError, match="checkpoints/ckpt.pt"):
            handler._get_upload_url("checkpoints/ckpt.pt", "application/octet-stream")

    def test_get_upload_url_sends_filepath_and_content_type_as_query_params(
        self, handler, requests_mock
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url",
            json={"url": SIGNED_URL},
            status_code=200,
        )
        handler._get_upload_url("logs/train.log", "text/plain")
        qs = requests_mock.last_request.qs
        assert qs["filepath"] == ["logs/train.log"]
        assert qs["content_type"] == ["text/plain"]


class TestGetCheckpointDownloadUrl:
    def test_get_checkpoint_download_url_calls_correct_endpoint_and_returns_signed_url(
        self, handler, requests_mock
    ):
        checkpoint_name = "checkpoint_latest.pt"
        requests_mock.get(
            f"{BASE_JOB_URL}/checkpoint_url/{checkpoint_name}",
            json={"url": SIGNED_URL},
            status_code=200,
        )
        url = handler._get_checkpoint_download_url(checkpoint_name)
        assert url == SIGNED_URL
        assert checkpoint_name in requests_mock.last_request.url

    def test_get_checkpoint_download_url_raises_value_error_when_checkpoint_not_found(
        self, handler, requests_mock
    ):
        checkpoint_name = "checkpoint_missing.pt"
        requests_mock.get(
            f"{BASE_JOB_URL}/checkpoint_url/{checkpoint_name}",
            json={"detail": "Not found"},
            status_code=404,
        )
        with pytest.raises(ValueError, match=checkpoint_name):
            handler._get_checkpoint_download_url(checkpoint_name)


class TestSaveCheckpoint:
    def test_save_checkpoint_writes_checkpoint_dict_to_local_dir(
        self, local_handler, tmp_path
    ):
        checkpoint = {"epoch": 1, "loss": 0.5}
        local_handler.save_checkpoint(checkpoint, Path("checkpoint_1.pt"))
        loaded = torch.load(tmp_path / "checkpoint_1.pt", weights_only=True)
        assert loaded["epoch"] == 1
        assert loaded["loss"] == 0.5

    def test_save_checkpoint_removes_local_file_after_successful_cloud_upload(
        self, handler, requests_mock, tmp_path
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        requests_mock.put(SIGNED_URL, status_code=200)

        handler.save_checkpoint({"epoch": 2}, Path("checkpoint_2.pt"))

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 1
        assert not (tmp_path / "checkpoint_2.pt").exists()

    def test_save_checkpoint_keeps_local_file_when_cloud_upload_fails(
        self, handler, requests_mock, tmp_path
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        requests_mock.put(SIGNED_URL, status_code=500, text="Server Error")

        handler.save_checkpoint({"epoch": 3}, Path("checkpoint_3.pt"))

        assert (tmp_path / "checkpoint_3.pt").exists()

    def test_save_checkpoint_creates_missing_parent_directories(
        self, local_handler, tmp_path
    ):
        local_handler.save_checkpoint({"epoch": 1}, Path("subdir/checkpoint_1.pt"))
        assert (tmp_path / "subdir" / "checkpoint_1.pt").exists()

    def test_save_checkpoint_converts_omegaconf_config_so_it_can_be_loaded_back(
        self, local_handler, tmp_path
    ):
        omegaconf = pytest.importorskip("omegaconf")
        cfg = omegaconf.OmegaConf.create(
            {"optimizer": {"lr": 0.001, "betas": [0.9, 0.999]}}
        )
        local_handler.save_checkpoint(
            {"epoch": 1, "config": cfg}, Path("checkpoint_1.pt")
        )
        loaded = torch.load(tmp_path / "checkpoint_1.pt", weights_only=True)
        assert loaded["config"] == {"optimizer": {"lr": 0.001, "betas": [0.9, 0.999]}}
        assert isinstance(loaded["config"]["optimizer"], dict)


class TestLoadCheckpoint:
    def test_load_checkpoint_downloads_from_signed_url_then_loads_from_disk(
        self, handler, requests_mock
    ):
        checkpoint_name = "checkpoint_latest.pt"
        checkpoint_data = {"epoch": 5, "model_state": {}}
        requests_mock.get(
            f"{BASE_JOB_URL}/checkpoint_url/{checkpoint_name}",
            json={"url": SIGNED_URL},
            status_code=200,
        )
        requests_mock.get(
            SIGNED_URL,
            content=_serialize_checkpoint(checkpoint_data),
            status_code=200,
        )

        result = handler.load_checkpoint(checkpoint_name)

        assert result["epoch"] == 5

    def test_load_checkpoint_raises_value_error_when_signed_url_download_fails(
        self, handler, requests_mock, tmp_path
    ):
        checkpoint_name = "checkpoint_latest.pt"
        requests_mock.get(
            f"{BASE_JOB_URL}/checkpoint_url/{checkpoint_name}",
            json={"url": SIGNED_URL},
            status_code=200,
        )
        requests_mock.get(SIGNED_URL, status_code=403, text="Forbidden")

        with pytest.raises(ValueError, match=checkpoint_name):
            handler.load_checkpoint(checkpoint_name)

    def test_load_checkpoint_loads_directly_from_local_dir_without_cloud(
        self, local_handler, tmp_path
    ):
        checkpoint_name = "checkpoint_latest.pt"
        torch.save({"epoch": 3}, tmp_path / checkpoint_name)

        result = local_handler.load_checkpoint(checkpoint_name)

        assert result["epoch"] == 3


class TestDeleteCheckpoint:
    def test_delete_checkpoint_removes_local_file_from_disk(
        self, local_handler, tmp_path
    ):
        ckpt_path = tmp_path / "checkpoint_1.pt"
        ckpt_path.write_bytes(b"data")

        local_handler.delete_checkpoint(Path("checkpoint_1.pt"))

        assert not ckpt_path.exists()

    def test_delete_checkpoint_does_not_raise_when_local_file_does_not_exist(
        self, local_handler
    ):
        local_handler.delete_checkpoint(Path("nonexistent.pt"))

    def test_delete_checkpoint_sends_delete_request_to_cloud_endpoint(
        self, handler, requests_mock, tmp_path
    ):
        checkpoint_name = "checkpoint_1.pt"
        (tmp_path / checkpoint_name).write_bytes(b"data")
        requests_mock.delete(
            f"{BASE_JOB_URL}/checkpoints/{checkpoint_name}", status_code=200
        )

        handler.delete_checkpoint(Path(checkpoint_name))

        delete_requests = [
            r for r in requests_mock.request_history if r.method == "DELETE"
        ]
        assert len(delete_requests) == 1
        assert f"checkpoints/{checkpoint_name}" in delete_requests[0].url

    def test_delete_checkpoint_does_not_raise_when_cloud_delete_fails(
        self, handler, requests_mock, tmp_path
    ):
        checkpoint_name = "checkpoint_1.pt"
        (tmp_path / checkpoint_name).write_bytes(b"data")
        requests_mock.delete(
            f"{BASE_JOB_URL}/checkpoints/{checkpoint_name}",
            status_code=500,
            text="Server Error",
        )

        handler.delete_checkpoint(Path(checkpoint_name))


class TestSaveModelArtifacts:
    def test_save_model_artifacts_creates_artifacts_dir_and_calls_create_nc_archive(
        self, local_handler, tmp_path
    ):
        mock_model = MagicMock()
        output_dir = Path("run_1")

        with patch(
            "neuracore.ml.utils.training_storage_handler.create_nc_archive"
        ) as mock_archive:
            local_handler.save_model_artifacts(mock_model, output_dir)

        artifacts_dir = tmp_path / output_dir / "artifacts"
        assert artifacts_dir.exists()
        mock_archive.assert_called_once_with(
            model=mock_model,
            output_dir=artifacts_dir,
            algorithm_config={},
        )

    def test_save_model_artifacts_uploads_each_artifact_file_to_cloud(
        self, handler, requests_mock, tmp_path
    ):
        mock_model = MagicMock()
        output_dir = Path("run_1")

        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        requests_mock.put(SIGNED_URL, status_code=200)

        def fake_archive(model, output_dir, algorithm_config):
            (output_dir / "model.pt").write_bytes(b"model_data")
            (output_dir / "config.json").write_bytes(b"{}")

        with patch(
            "neuracore.ml.utils.training_storage_handler.create_nc_archive",
            side_effect=fake_archive,
        ):
            handler.save_model_artifacts(mock_model, output_dir)

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 2


class TestUpdateTrainingProgress:
    def test_sends_epoch_and_step_with_null_error_to_cloud(
        self, handler, requests_mock
    ):
        requests_mock.put(f"{BASE_JOB_URL}/update", status_code=200)

        handler.update_training_progress(epoch=3, step=150)

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 1
        assert put_requests[0].json() == {"epoch": 3, "step": 150, "error": None}

    def test_makes_no_http_request_without_job_id(self, local_handler, requests_mock):
        local_handler.update_training_progress(epoch=1, step=1)

        assert len(requests_mock.request_history) == 0


class TestReportTrainingError:
    def test_sends_error_with_null_epoch_and_step_to_cloud(
        self, handler, requests_mock
    ):
        requests_mock.put(f"{BASE_JOB_URL}/update", status_code=200)

        handler.report_training_error("OOM error")

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 1
        assert put_requests[0].json() == {
            "epoch": None,
            "step": None,
            "error": "OOM error",
        }

    def test_makes_no_http_request_without_job_id(self, local_handler, requests_mock):
        local_handler.report_training_error("some error")

        assert len(requests_mock.request_history) == 0


class TestConvertOmegaconfToPython:
    def test_convert_omegaconf_converts_nested_dict_and_list_config_to_builtin_types(
        self, local_handler
    ):
        omegaconf = pytest.importorskip("omegaconf")
        cfg = omegaconf.OmegaConf.create(
            {"optimizer": {"lr": 0.001, "betas": [0.9, 0.999]}}
        )
        result = local_handler._convert_omegaconf_to_python(cfg)
        assert result == {"optimizer": {"lr": 0.001, "betas": [0.9, 0.999]}}
        assert isinstance(result, dict)
        assert isinstance(result["optimizer"], dict)
        assert isinstance(result["optimizer"]["betas"], list)


def _serialize_checkpoint(data: dict) -> bytes:
    """Serialize a checkpoint dict to bytes via torch.save."""
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()
