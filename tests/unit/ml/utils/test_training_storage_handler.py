"""Tests for TrainingStorageHandler."""

import io
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from neuracore.core.const import API_URL
from neuracore.ml.utils import upload_storage_mixin
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler

ORG_ID = "test-org-id"
JOB_ID = "test-job-id"
BASE_JOB_URL = f"{API_URL}/org/{ORG_ID}/training/jobs/{JOB_ID}"
SIGNED_URL = "https://storage.example.com/signed-url"
INPUT_CROSS_EMBODIMENT_DESCRIPTION = {"robot": {"joints": {"0": {"name": "j0"}}}}
OUTPUT_CROSS_EMBODIMENT_DESCRIPTION = {"robot": {"actions": {"0": {"name": "a0"}}}}


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    """Skip real backoff sleeps so retry-triggering tests run fast."""
    monkeypatch.setattr(upload_storage_mixin.time, "sleep", lambda *_: None)


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
        input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
        output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
    )


@pytest.fixture
def local_handler(tmp_path, mock_auth, mock_org):
    """Create a local-only TrainingStorageHandler (no cloud)."""
    return TrainingStorageHandler(
        local_dir=str(tmp_path),
        input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
        output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
    )


class TestInit:
    def test_init_with_job_id_verifies_job_exists_in_cloud(
        self, tmp_path, requests_mock, mock_auth, mock_org
    ):
        requests_mock.get(BASE_JOB_URL, json={}, status_code=200)
        cloud_handler = TrainingStorageHandler(
            local_dir=str(tmp_path),
            training_job_id=JOB_ID,
            input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
            output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
        )
        assert cloud_handler.log_to_cloud is True
        assert cloud_handler.training_job_id == JOB_ID
        assert cloud_handler.org_id == ORG_ID

    def test_init_raises_value_error_when_job_not_found(
        self, tmp_path, requests_mock, mock_auth, mock_org
    ):
        requests_mock.get(BASE_JOB_URL, json={"detail": "Not found"}, status_code=404)
        with pytest.raises(ValueError, match=JOB_ID):
            TrainingStorageHandler(
                local_dir=str(tmp_path),
                training_job_id=JOB_ID,
                input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
                output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
            )

    def test_init_without_job_id_disables_cloud_logging(
        self, tmp_path, mock_auth, mock_org
    ):
        local_only_handler = TrainingStorageHandler(
            local_dir=str(tmp_path),
            input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
            output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
        )
        assert local_only_handler.log_to_cloud is False
        assert local_only_handler.training_job_id is None

    def test_init_sets_local_dir_to_output_when_none_given(self, mock_auth, mock_org):
        handler_with_default_dir = TrainingStorageHandler(
            local_dir=None,
            input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
            output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
        )
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
        handler.wait_for_pending_uploads()

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
        handler.wait_for_pending_uploads()

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


class TestSaveCheckpointConcurrencyGuard:
    """Covers the background-upload race guarded against in save_checkpoint.

    checkpoint_latest.pt (and similar reused paths) get torch.save()'d again
    on every call while a prior call's upload may still be reading the same
    file; save_checkpoint must wait for that prior upload before overwriting.
    """

    def test_second_save_to_same_path_waits_for_first_upload_to_finish(
        self, handler, requests_mock, tmp_path
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        first_upload_started = threading.Event()
        release_first_upload = threading.Event()
        call_count = {"n": 0}

        def slow_put(request, context):
            call_count["n"] += 1
            if call_count["n"] == 1:
                first_upload_started.set()
                release_first_upload.wait(timeout=5)
            context.status_code = 200
            return ""

        requests_mock.put(SIGNED_URL, text=slow_put)

        path = Path("checkpoint_latest.pt")
        handler.save_checkpoint({"epoch": 1}, path)
        assert first_upload_started.wait(timeout=2), "first upload should have started"

        second_save_returned = threading.Event()

        def do_second_save():
            handler.save_checkpoint({"epoch": 2}, path)
            second_save_returned.set()

        second_save_thread = threading.Thread(target=do_second_save)
        second_save_thread.start()

        # The second save must not proceed (and overwrite checkpoint_latest.pt
        # locally) while the first upload still has it open for reading.
        assert not second_save_returned.wait(timeout=0.3)

        release_first_upload.set()
        second_save_thread.join(timeout=5)
        assert second_save_returned.is_set()

        handler.wait_for_pending_uploads()
        assert call_count["n"] == 2
        # Both uploads succeeded, so the local file was deleted after the
        # second (successful) upload; confirm it was the epoch-2 checkpoint
        # that was actually sent, not stale/corrupted bytes from the first.
        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        last_uploaded = torch.load(io.BytesIO(put_requests[-1].body), weights_only=True)
        assert last_uploaded["epoch"] == 2
        assert not (tmp_path / "checkpoint_latest.pt").exists()

    def test_saves_to_different_paths_do_not_block_each_other(
        self, handler, requests_mock
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        release_upload = threading.Event()

        def slow_put(request, context):
            release_upload.wait(timeout=5)
            context.status_code = 200
            return ""

        requests_mock.put(SIGNED_URL, text=slow_put)

        # First save's upload blocks on the background worker until released.
        handler.save_checkpoint({"epoch": 1}, Path("checkpoint_1.pt"))

        second_returned = threading.Event()

        def do_second_save():
            handler.save_checkpoint({"epoch": 2}, Path("checkpoint_2.pt"))
            second_returned.set()

        second_save_thread = threading.Thread(target=do_second_save)
        second_save_thread.start()

        assert second_returned.wait(timeout=2), (
            "save_checkpoint for a distinct path must not wait on an "
            "unrelated in-flight upload"
        )

        release_upload.set()
        second_save_thread.join(timeout=5)
        handler.wait_for_pending_uploads()

    def test_delete_waits_for_pending_upload_of_same_path_before_deleting(
        self, handler, requests_mock, tmp_path
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        requests_mock.delete(
            f"{BASE_JOB_URL}/checkpoints/checkpoint_1.pt", status_code=200
        )
        upload_started = threading.Event()
        release_upload = threading.Event()

        def slow_put(request, context):
            upload_started.set()
            release_upload.wait(timeout=5)
            context.status_code = 200
            return ""

        requests_mock.put(SIGNED_URL, text=slow_put)

        path = Path("checkpoint_1.pt")
        handler.save_checkpoint({"epoch": 1}, path)
        assert upload_started.wait(timeout=2)

        delete_returned = threading.Event()

        def do_delete():
            handler.delete_checkpoint(path)
            delete_returned.set()

        delete_thread = threading.Thread(target=do_delete)
        delete_thread.start()

        assert not delete_returned.wait(timeout=0.3)

        release_upload.set()
        delete_thread.join(timeout=5)
        assert delete_returned.is_set()

    def test_wait_for_pending_uploads_drains_and_does_not_raise_on_failure(
        self, handler, requests_mock, tmp_path
    ):
        requests_mock.get(
            f"{BASE_JOB_URL}/upload-url", json={"url": SIGNED_URL}, status_code=200
        )
        requests_mock.put(SIGNED_URL, status_code=500, text="Server Error")

        handler.save_checkpoint({"epoch": 1}, Path("checkpoint_1.pt"))

        # Must not raise even though the upload ultimately fails after
        # exhausting retries.
        handler.wait_for_pending_uploads()

        assert (tmp_path / "checkpoint_1.pt").exists()

    def test_wait_for_pending_uploads_is_a_no_op_with_nothing_pending(
        self, local_handler
    ):
        local_handler.wait_for_pending_uploads()


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
            input_cross_embodiment_description=INPUT_CROSS_EMBODIMENT_DESCRIPTION,
            output_cross_embodiment_description=OUTPUT_CROSS_EMBODIMENT_DESCRIPTION,
            input_preprocessing_config={},
            output_preprocessing_config={},
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

        def fake_archive(
            model,
            output_dir,
            algorithm_config,
            input_cross_embodiment_description,
            output_cross_embodiment_description,
            input_preprocessing_config,
            output_preprocessing_config,
        ):
            (output_dir / "model.pt").write_bytes(b"model_data")
            (output_dir / "config.json").write_bytes(b"{}")

        with patch(
            "neuracore.ml.utils.training_storage_handler.create_nc_archive",
            side_effect=fake_archive,
        ):
            handler.save_model_artifacts(mock_model, output_dir)
        handler.wait_for_pending_uploads()

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 2


class TestUpdateTrainingProgress:
    def test_sends_epoch_and_step_with_null_error_to_cloud(
        self, handler, requests_mock
    ):
        requests_mock.put(f"{BASE_JOB_URL}/update", status_code=200)

        handler.update_training_progress(epoch=3, step=150)
        handler.wait_for_pending_progress_updates()

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 1
        assert put_requests[0].json() == {"epoch": 3, "step": 150, "error": None}

    def test_does_not_log_in_again_on_401(self, handler, requests_mock, monkeypatch):
        login = MagicMock()
        monkeypatch.setattr("neuracore.login", login)
        matcher = requests_mock.put(f"{BASE_JOB_URL}/update", status_code=401)

        handler.update_training_progress(epoch=3, step=150)
        handler.wait_for_pending_progress_updates()

        assert matcher.call_count == 1
        assert not login.called

    def test_makes_no_http_request_without_job_id(self, local_handler, requests_mock):
        local_handler.update_training_progress(epoch=1, step=1)
        local_handler.wait_for_pending_progress_updates()

        assert len(requests_mock.request_history) == 0

    def test_coalesces_updates_queued_while_one_is_in_flight(
        self, handler, requests_mock
    ):
        """A backlog of stale progress must not build up behind a slow endpoint.

        The first request is held until further updates have been queued, so
        the worker is guaranteed to still be busy when they arrive. Only the
        newest of those should be sent afterwards.
        """
        first_put_started = threading.Event()
        release_first_put = threading.Event()

        def _blocking_response(request, context):
            if not first_put_started.is_set():
                first_put_started.set()
                release_first_put.wait(timeout=5)
            context.status_code = 200
            return ""

        requests_mock.put(f"{BASE_JOB_URL}/update", text=_blocking_response)

        handler.update_training_progress(epoch=1, step=10)
        assert first_put_started.wait(timeout=5)
        handler.update_training_progress(epoch=1, step=20)
        handler.update_training_progress(epoch=1, step=30)
        release_first_put.set()
        handler.wait_for_pending_progress_updates()

        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert [r.json()["step"] for r in put_requests] == [10, 30]

    def test_a_failing_update_does_not_propagate(self, handler, requests_mock):
        """Progress reporting is best-effort; it must not fail a training run."""
        requests_mock.put(f"{BASE_JOB_URL}/update", exc=ConnectionError("down"))

        handler.update_training_progress(epoch=3, step=150)
        handler.wait_for_pending_progress_updates()


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
