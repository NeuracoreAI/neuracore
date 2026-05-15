"""Storage handler for endpoint inference log uploads."""

import logging
from typing import IO

import requests

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_session import get_session
from neuracore.ml.utils.upload_storage_mixin import UploadStorageMixin

logger = logging.getLogger(__name__)


class EndpointStorageHandler(UploadStorageMixin):
    """Upload helper for endpoint cloud logs."""

    def __init__(self, endpoint_id: str | None = None) -> None:
        """Initialize endpoint storage handler.

        Args:
            endpoint_id: Optional endpoint ID for cloud uploads.
        """
        self.endpoint_id = endpoint_id
        self.log_to_cloud = self.endpoint_id is not None
        self.org_id = get_current_org()
        if self.log_to_cloud:
            response = get_session().get(
                f"{API_URL}/org/{self.org_id}/models/endpoints/{self.endpoint_id}",
                headers=get_auth().get_headers(),
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Endpoint {self.endpoint_id} not found or access denied."
                )

    def _get_upload_url(self, filepath: str, content_type: str) -> str:
        """Get a signed upload URL for endpoint logs."""
        assert self.endpoint_id is not None, "Endpoint ID not provided"
        response = get_session().get(
            f"{API_URL}/org/{self.org_id}/models/endpoints/{self.endpoint_id}/upload-url",
            headers=get_auth().get_headers(),
            params={"filepath": filepath, "content_type": content_type},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get upload URL for {filepath}: {response.text}"
            )
        return response.json()["url"]

    def _execute_upload(
        self,
        upload_url: str,
        data: bytes | IO[bytes],
        content_type: str,
    ) -> requests.Response:
        return get_session().put(
            upload_url,
            data=data,
            headers={"Content-Type": content_type},
        )

    def report_endpoint_error(self, error: str) -> None:
        """Report endpoint startup/runtime failures to cloud metadata."""
        if not self.log_to_cloud:
            return

        assert self.endpoint_id is not None, "Endpoint ID not provided"
        response = get_session().put(
            f"{API_URL}/org/{self.org_id}/models/endpoints/{self.endpoint_id}/update",
            headers=get_auth().get_headers(),
            json={"error": error},
        )
        if response.status_code != 200:
            logger.error("Failed to report endpoint error to cloud: %s", response.text)
