"""TrainingStorageHandler for managing model training artifacts and checkpoints."""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
import torch
from torch import nn

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_session import thread_local_session
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.nc_archive import create_nc_archive
from neuracore.ml.utils.upload_storage_mixin import UploadStorageMixin

logger = logging.getLogger(__name__)


class TrainingStorageHandler(UploadStorageMixin):
    """Handles storage operations for both local and GCS."""

    def __init__(
        self,
        local_dir: str | None,
        training_job_id: str | None = None,
        algorithm_config: dict = {},
        input_cross_embodiment_description: dict[str, Any] = {},
        output_cross_embodiment_description: dict[str, Any] = {},
        input_preprocessing_config: PreprocessingConfiguration = (
            PreprocessingConfiguration()
        ),
        output_preprocessing_config: PreprocessingConfiguration = (
            PreprocessingConfiguration()
        ),
        verify_job_exists: bool = True,
    ) -> None:
        """Initialize the storage handler.

        Args:
            local_dir: Local directory to save artifacts and checkpoints.
            training_job_id: Optional ID of the training job for cloud logging.
            algorithm_config: Optional configuration for the algorithm.
            input_cross_embodiment_description: Input embodiment mapping
                to persist with model artifacts.
            output_cross_embodiment_description: Output embodiment mapping
                to persist with model artifacts.
            input_preprocessing_config: preprocessing configuration for the input
                data.
            output_preprocessing_config: preprocessing configuration for the output
                data.
            verify_job_exists: Whether to confirm the training job is reachable
                on construction. Costs one blocking request, so distributed
                ranks other than rank 0 skip it — rank 0 has already verified
                the same job, and only rank 0 writes to it.
        """
        self.local_dir = Path(local_dir or "./output")
        self.training_job_id = training_job_id
        self.algorithm_config = algorithm_config
        self.input_cross_embodiment_description = input_cross_embodiment_description
        self.output_cross_embodiment_description = output_cross_embodiment_description
        self.input_preprocessing_config = input_preprocessing_config
        self.output_preprocessing_config = output_preprocessing_config
        self.log_to_cloud = self.training_job_id is not None
        self.org_id = get_current_org()
        if self.log_to_cloud and verify_job_exists:
            response = self._get_request(
                f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}"
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Training job {self.training_job_id} not found or access denied."
                )

        # Checkpoint/artifact uploads run on this single background worker so
        # save_checkpoint()/save_model_artifacts() don't block the training
        # loop for the duration of a large upload. One worker means every
        # submitted job runs in strict submission order, which is what makes
        # delete_checkpoint() safe to call right after a save without extra
        # coordination. Created lazily since it's only needed when uploading.
        self._upload_executor: ThreadPoolExecutor | None = None
        self._pending_uploads_lock = threading.Lock()
        self._pending_uploads: dict[Path, Future] = {}

        # Progress updates are fire-and-forget from the training loop's point
        # of view. They get their own worker rather than sharing the upload
        # one, so a multi-gigabyte checkpoint upload can't leave reported
        # progress stalled behind it.
        self._progress_executor: ThreadPoolExecutor | None = None
        self._progress_lock = threading.Lock()
        self._pending_progress: tuple[int, int] | None = None
        self._progress_future: Future | None = None
        self._progress_worker_running = False

    def _get_upload_url(self, filepath: str, content_type: str) -> str:
        """Get a signed upload URL for a file in cloud storage.

        Args:
            filepath: Path of the file to upload.
            content_type: MIME type of the file.

        Returns:
            str: Signed URL for uploading the file.

        Raises:
            ValueError: If the request to get the upload URL fails.
        """
        params = {
            "filepath": filepath,
            "content_type": content_type,
        }

        response = self._get_request(
            f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}/upload-url",
            params=params,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get upload URL for {filepath}: {response.text}"
            )
        return response.json()["url"]

    def _get_checkpoint_download_url(self, checkpoint_name: str) -> str:
        """Get a signed download URL for a checkpoint file in cloud storage.

        Args:
            checkpoint_name: Name of the checkpoint file to download.

        Returns:
            str: Signed URL for downloading the checkpoint.

        Raises:
            ValueError: If the request to get the download URL fails.
        """
        response = self._get_request(
            f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}"
            f"/checkpoint_url/{checkpoint_name}",
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get download URL for {checkpoint_name}: {response.text}"
            )
        return response.json()["url"]

    def _get_upload_executor(self) -> ThreadPoolExecutor:
        """Lazily create the single-worker background upload executor."""
        if self._upload_executor is None:
            self._upload_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="nc-checkpoint-upload"
            )
        return self._upload_executor

    def _wait_for_pending_upload(self, local_path: Path) -> None:
        """Block until any in-flight upload of ``local_path`` has finished.

        Some destinations (e.g. model artifacts, which get regenerated in
        place) are reused across calls rather than written to a fresh path
        each time. Waiting here before that path is overwritten or deleted
        prevents a background upload thread from reading a file out from
        under a newer write.
        """
        with self._pending_uploads_lock:
            future = self._pending_uploads.get(local_path)
        if future is not None and not future.done():
            logger.debug(
                "Waiting for pending upload of %s to finish before reusing it",
                local_path,
            )
            future.result()

    def _submit_upload(
        self,
        local_path: Path,
        remote_filepath: str,
        content_type: str,
        delete_on_success: bool,
    ) -> None:
        """Upload ``local_path`` on the background worker.

        Args:
            local_path: Local file to upload.
            remote_filepath: Destination path within cloud storage.
            content_type: MIME type of the file being uploaded.
            delete_on_success: Whether to unlink the local file once the
                upload succeeds.
        """

        def _do_upload() -> None:
            try:
                uploaded = self.upload_file(local_path, remote_filepath, content_type)
            except Exception:
                logger.error(
                    "Unexpected error uploading %s to cloud path %s",
                    local_path,
                    remote_filepath,
                    exc_info=True,
                )
                return
            if uploaded and delete_on_success:
                try:
                    local_path.unlink()
                except Exception as e:
                    logger.warning(
                        "Could not delete local file %s after upload: %s",
                        local_path,
                        e,
                    )

        future = self._get_upload_executor().submit(_do_upload)
        with self._pending_uploads_lock:
            self._pending_uploads[local_path] = future

    def wait_for_pending_uploads(self) -> None:
        """Block until every submitted checkpoint/artifact upload has finished.

        Call this before the process exits (e.g. at the end of training) —
        checkpoint and artifact uploads run in the background, so without
        this the container/VM can be torn down while the final checkpoint is
        still mid-upload.
        """
        with self._pending_uploads_lock:
            futures = list(self._pending_uploads.values())
        for future in futures:
            future.result()

    def save_checkpoint(self, checkpoint: dict, relative_checkpoint_path: Path) -> None:
        """Save checkpoint to storage.

        Writes the checkpoint to disk synchronously, then — if cloud logging
        is enabled — uploads it on a background thread so this call doesn't
        block the training loop for the duration of the upload. If a caller
        reuses the same ``relative_checkpoint_path`` across calls, this first
        waits for any previous upload of that same path to finish, so it's
        never overwritten while still being read for upload.

        Args:
            checkpoint: Checkpoint dictionary to save.
            relative_checkpoint_path: Relative path for the checkpoint file.
        """
        save_path = self.local_dir / relative_checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.log_to_cloud:
            self._wait_for_pending_upload(save_path)

        # Convert OmegaConf objects to plain Python types
        # for compatibility with weights_only=True
        checkpoint = self._convert_omegaconf_to_python(checkpoint)
        torch.save(checkpoint, save_path)

        if self.log_to_cloud:
            self._submit_upload(
                save_path,
                remote_filepath=f"checkpoints/{relative_checkpoint_path.name}",
                content_type="application/octet-stream",
                delete_on_success=True,
            )

    def _convert_omegaconf_to_python(self, obj: Any) -> Any:
        """Recursively convert OmegaConf objects to plain Python types.

        This is needed when saving optimizers and schedulers in the checkpoint.

        Args:
            obj: Object that may contain OmegaConf objects.

        Returns:
            Object with OmegaConf objects converted to plain Python types.
        """
        try:
            from omegaconf import DictConfig, ListConfig
        except ImportError:
            # OmegaConf not available, return as-is
            return obj

        if isinstance(obj, DictConfig):
            return {k: self._convert_omegaconf_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, ListConfig):
            return [self._convert_omegaconf_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_omegaconf_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_omegaconf_to_python(item) for item in obj)
        else:
            return obj

    def load_checkpoint(self, checkpoint_name: str) -> dict:
        """Load checkpoint from storage.

        Args:
            checkpoint_name: Name of the checkpoint file to load.

        Returns:
            dict: Loaded checkpoint dictionary.

        Raises:
            ValueError: If the checkpoint cannot be downloaded or loaded.
        """
        load_path = self.local_dir / checkpoint_name
        if self.log_to_cloud:
            download_url = self._get_checkpoint_download_url(checkpoint_name)
            session = thread_local_session(retry_transient=True)
            response = session.get(download_url)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to download checkpoint {checkpoint_name}: {response.text}"
                )
            with open(load_path, "wb") as f:
                f.write(response.content)

        return torch.load(load_path, weights_only=True)

    def delete_checkpoint(self, relative_checkpoint_path: Path) -> None:
        """Delete checkpoint from storage.

        Waits for any pending upload of this same checkpoint to finish first
        — deleting a file a background thread still has open for upload, or
        deleting the cloud copy before the upload that creates it lands,
        would otherwise race.

        Args:
            relative_checkpoint_path: Relative path of the checkpoint file to delete.
        """
        checkpoint_path = self.local_dir / relative_checkpoint_path
        if self.log_to_cloud:
            self._wait_for_pending_upload(checkpoint_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if self.log_to_cloud:
            response = self._delete_request(
                f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}/checkpoints/{relative_checkpoint_path.name}"
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to delete checkpoint {relative_checkpoint_path} "
                    f"from cloud: {response.text}"
                )
                return

    def save_model_artifacts(self, model: nn.Module, output_dir: Path) -> None:
        """Save model artifacts to storage.

        ``create_nc_archive`` regenerates fixed-name files (e.g.
        ``model.nc.zip``) in ``artifacts_dir`` on every call, so — like
        ``save_checkpoint`` — this waits for any artifact upload still in
        flight from a previous call before regenerating them, then uploads
        the new ones on the background worker without blocking the training
        loop. Only artifact-path uploads are waited on here, not checkpoint
        uploads, so the two don't block each other.

        Args:
            model: PyTorch model to save.
            output_dir: Directory to save the artifacts.
        """
        artifacts_dir = self.local_dir / output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.log_to_cloud:
            with self._pending_uploads_lock:
                pending_paths = [
                    path
                    for path in self._pending_uploads
                    if path.is_relative_to(artifacts_dir)
                ]
            for path in pending_paths:
                self._wait_for_pending_upload(path)

        create_nc_archive(
            model=model,
            output_dir=artifacts_dir,
            algorithm_config=self.algorithm_config,
            input_cross_embodiment_description=self.input_cross_embodiment_description,
            output_cross_embodiment_description=self.output_cross_embodiment_description,
            input_preprocessing_config=self.input_preprocessing_config,
            output_preprocessing_config=self.output_preprocessing_config,
        )
        if self.log_to_cloud:
            for file_path in artifacts_dir.glob("*"):
                self._submit_upload(
                    file_path,
                    remote_filepath=str(file_path.name),
                    content_type="application/octet-stream",
                    delete_on_success=False,
                )

    def update_training_progress(self, epoch: int, step: int) -> None:
        """Queue a training epoch/step progress update for cloud storage.

        This is called from inside the training loop, so the HTTP PUT runs on
        a background worker rather than blocking the step. Updates coalesce:
        if a PUT is already in flight, this replaces the payload that will be
        sent next instead of queueing another request.

        Args:
            epoch: Current training epoch.
            step: Current training step.
        """
        if not self.log_to_cloud:
            return

        with self._progress_lock:
            self._pending_progress = (epoch, step)
            if self._progress_worker_running:
                # A worker is already draining; it will observe the payload we
                # just stored. The flag is cleared under this same lock, so a
                # worker that has decided to exit cannot still be marked
                # running here.
                return
            if self._progress_executor is None:
                self._progress_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="nc-training-progress"
                )
            self._progress_worker_running = True
            self._progress_future = self._progress_executor.submit(
                self._drain_progress_updates
            )

    def _drain_progress_updates(self) -> None:
        """Send queued progress updates until none remain."""
        while True:
            with self._progress_lock:
                pending = self._pending_progress
                self._pending_progress = None
                if pending is None:
                    # Clear the flag in the same critical section that observes
                    # the empty queue, so a concurrent enqueue either lands
                    # before this and is drained, or sees the flag cleared and
                    # starts a fresh worker.
                    self._progress_worker_running = False
                    return
            epoch, step = pending
            self._send_training_progress(epoch, step)

    def _send_training_progress(self, epoch: int, step: int) -> None:
        """Send a single progress update, logging rather than raising on failure."""
        try:
            response = self._put_request(
                f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}/update",
                json={"epoch": epoch, "step": step, "error": None},
            )
        except Exception:
            logger.error("Failed to update training progress to cloud.", exc_info=True)
            return
        if response.status_code != 200:
            logger.error(
                f"Failed to update training progress to cloud: {response.text}"
            )

    def wait_for_pending_progress_updates(self) -> None:
        """Block until any queued progress update has been sent.

        Called before the process exits so the final epoch/step reaches the
        backend rather than dying with the worker thread.
        """
        with self._progress_lock:
            future = self._progress_future
        if future is not None:
            try:
                future.result()
            except Exception:
                logger.error("Progress update worker failed.", exc_info=True)

    def report_training_error(self, error: str) -> None:
        """Report a training failure to cloud storage.

        This should be called in exactly one place — the top-level error
        handler in train.py — so that every failure is surfaced regardless
        of where in the training pipeline it originated.

        Args:
            error: Formatted error / traceback string to persist.
        """
        if self.log_to_cloud:
            response = self._put_request(
                f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}/update",
                json={"epoch": None, "step": None, "error": error},
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to report training error to cloud: {response.text}"
                )

    def _put_request(
        self,
        url: str,
        json: dict | None = None,
        data: Any | None = None,
        headers: dict | None = None,
    ) -> requests.Response:
        """Helper method to send a PUT request.

        Args:
            url: The URL to send the request to.
            json: The JSON payload to include in the request.
            data: Optional data to include in the request body.
            headers: Optional headers to include in the request.
        """
        headers = headers or get_auth().get_headers()
        session = thread_local_session(retry_transient=True)
        return session.put(url, headers=headers, json=json, data=data)

    def _get_request(self, url: str, params: dict | None = None) -> requests.Response:
        """Helper method to send a GET request.

        Args:
            url: The URL to send the request to.
            params: Optional parameters to include in the request.
        """
        session = thread_local_session(retry_transient=True)
        return session.get(url, headers=get_auth().get_headers(), params=params)

    def _delete_request(self, url: str) -> requests.Response:
        """Helper method to send a DELETE request.

        Args:
            url: The URL to send the request to.
        """
        session = thread_local_session(retry_transient=True)
        return session.delete(url, headers=get_auth().get_headers())
