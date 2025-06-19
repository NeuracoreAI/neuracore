"""TrainingStorageHandler for managing model training artifacts and checkpoints."""

import logging
from pathlib import Path
from typing import Optional

import requests
import torch
from torch import nn

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.ml.utils.mar import create_mar

FILE_PATH = Path(__file__).parent / "handlers.py"

logger = logging.getLogger(__name__)


class TrainingStorageHandler:
    """Handles storage operations for both local and GCS."""

    def __init__(self, local_dir: Optional[str], training_job_id: Optional[str] = None):
        """Initialize the storage handler.

        Args:
            local_dir: Local directory to save artifacts and checkpoints.
            training_job_id: Optional ID of the training job for cloud logging.
        """
        self.local_dir = Path(local_dir or "./output")
        self.training_job_id = training_job_id
        self.log_to_cloud = self.training_job_id is not None
        self.org_id = get_current_org()
        if self.log_to_cloud:
            response = requests.get(
                f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}",
                headers=get_auth().get_headers(),
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Training job {self.training_job_id} not found or access denied."
                )

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
        auth = get_auth()
        params = {
            "filepath": filepath,
            "content_type": content_type,
        }

        response = requests.get(
            f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}/upload-url",
            params=params,
            headers=auth.get_headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get upload URL for {filepath}: {response.text}"
            )
        return response.json()["url"]

    def _get_download_url(self, filepath: str) -> str:
        """Get a signed download URL for a file in cloud storage.

        Args:
            filepath: Path of the file to download.

        Returns:
            str: Signed URL for downloading the file.

        Raises:
            ValueError: If the request to get the download URL fails.
        """
        auth = get_auth()
        get_current_org()
        response = requests.get(
            f"{API_URL}/org/{self.org_id}/recording/{self.training_job_id}/download-url",
            params={"filepath": filepath},
            headers=auth.get_headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get download URL for {filepath}: {response.text}"
            )
        return response.json()["url"]

    def save_checkpoint(self, checkpoint: dict, checkpoint_name: str) -> None:
        """Save checkpoint to storage.

        Args:
            checkpoint: Checkpoint dictionary to save.
            checkpoint_name: Name of the checkpoint file.
        """
        save_path = self.local_dir / checkpoint_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        if self.log_to_cloud:
            upload_url = self._get_upload_url(
                filepath="checkpoints/" + checkpoint_name,
                content_type="application/octet-stream",
            )
            with open(save_path, "rb") as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "application/octet-stream"},
                )
            if response.status_code != 200:
                logger.error(
                    f"Failed to save checkpoint {checkpoint_name} "
                    f"to cloud: {response.text}"
                )
                return

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
            download_url = self._get_download_url(
                filepath="checkpoints/" + checkpoint_name
            )
            response = requests.get(download_url)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to download checkpoint {checkpoint_name}: {response.text}"
                )
            with open(load_path, "wb") as f:
                f.write(response.content)
        return torch.load(load_path)

    def save_model_artifacts(self, model: nn.Module, output_dir: Path) -> None:
        """Save model artifacts to storage.

        Args:
            model: PyTorch model to save.
            output_dir: Directory to save the artifacts.
        """
        artifacts_dir = self.local_dir / output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        create_mar(model=model, output_dir=artifacts_dir)
        if self.log_to_cloud:
            for file_path in artifacts_dir.glob("*"):
                upload_url = self._get_upload_url(
                    filepath=str(file_path.name),
                    content_type="application/octet-stream",
                )
                with open(file_path, "rb") as f:
                    response = requests.put(
                        upload_url,
                        data=f,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                if response.status_code != 200:
                    logger.error(
                        f"Failed to save artifact {file_path} to cloud: {response.text}"
                    )

    def update_training_metadata(
        self, epoch: int, step: int, error: str | None = None
    ) -> None:
        """Update training metadata in cloud storage.

        Args:
            epoch: Current training epoch.
            step: Current training step.
            error: Optional error message if training failed.
        """
        if self.log_to_cloud:
            response = requests.put(
                f"{API_URL}/org/{self.org_id}/training/jobs/{self.training_job_id}/update",
                headers=get_auth().get_headers(),
                json={"epoch": epoch, "step": step, "error": error},
            )
            if response.status_code != 200:
                logger.error(f"Failed to save epoch {epoch} to cloud: {response.text}")
