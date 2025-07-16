"""Handles storage operations for algorithms."""

import logging
import zipfile
from pathlib import Path
from typing import Optional

import requests
import wget

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.ml.utils.validate import AlgorithmCheck

logger = logging.getLogger(__name__)


class AlgorithmStorageHandler:
    """Handles storage operations for algorithms."""

    def __init__(self, algorithm_id: Optional[str] = None):
        """Initialize the AlgorithmStorageHandler.

        Args:
            algorithm_id: Optional ID of the algorithm to manage.
                If provided, will enable cloud logging and validation.
        """
        self.algorithm_id = algorithm_id
        self.log_to_cloud = self.algorithm_id is not None
        self.org_id = get_current_org()
        if self.log_to_cloud:
            response = requests.get(
                f"{API_URL}/org/{self.org_id}/algorithms/{self.algorithm_id}",
                headers=get_auth().get_headers(),
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Algorithm {self.algorithm_id} not found or access denied."
                )

    def save_algorithm_validation_error(self, error_message: str) -> None:
        """Save error message from failed algorithm validation.

        Args:
            error_message: Error message to save.
        """
        if self.log_to_cloud:
            response = requests.post(
                f"{API_URL}/org/{self.org_id}/algorithms/{self.algorithm_id}/validation-error",
                headers=get_auth().get_headers(),
                json={"error_message": error_message},
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to save algorithm validation error: {response.text}"
                )

    def download_algorithm(self, extract_dir: Path) -> None:
        """Download and extract algorithm code from storage.

        Args:
            extract_dir: Directory to extract algorithm code to.
        """
        response = requests.get(
            f"{API_URL}/org/{self.org_id}/algorithms/download_url/{self.algorithm_id}",
            headers=get_auth().get_headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get download URL for algorithm {self.algorithm_id}: "
                f"{response.text}"
            )
        download_url = response.json()["url"]
        local_zip_path = extract_dir / "algorithm.zip"
        extract_dir.mkdir(parents=True, exist_ok=True)
        wget.download(str(download_url), str(local_zip_path))

        # Extract the zip file
        logger.info(f"Extracting algorithm to {extract_dir}")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up the zip file
        local_zip_path.unlink()

    def save_algorithm_validation_check(
        self, checklist: AlgorithmCheck, error_message: str
    ) -> None:
        """Save algorithm validation checklist and status.

        Args:
            checklist: AlgorithmCheck instance with validation results.
            error_message: Error message if validation failed.
        """
        assert self.algorithm_id is not None, "Algorithm ID not provided"
        # check if all values in model (checklist) are true
        success = all(list(checklist.model_dump(mode="json").values()))
        dict_to_save = {
            "validation_checklist": checklist.model_dump(mode="json"),
        }
        if success:
            dict_to_save["validation_status"] = "available"
        else:
            dict_to_save["validation_status"] = "validation_failed"
            dict_to_save["validation_message"] = error_message
        response = requests.put(
            f"{API_URL}/org/{self.org_id}/algorithms/{self.algorithm_id}/update-algorithm-validation",
            headers=get_auth().get_headers(),
            json=dict_to_save,
        )
        if response.status_code != 200:
            logger.error(f"Failed to save algorithm validation check: {response.text}")
        else:
            logger.info("Algorithm validation check saved successfully.")
