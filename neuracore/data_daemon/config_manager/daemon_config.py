"""Pydantic models for Neuracore data daemon configuration."""

from pydantic import BaseModel


class DaemonConfig(BaseModel):
    """Configuration options for a Neuracore data daemon instance.

    Attributes:
        storage_limit: maximum storage the daemon may use locally, in bytes.
        bandwidth_limit: maximum upload bandwidth, in bytes per second.
        path_to_store_record: directory where the daemon writes recording files.
        num_threads: number of worker threads used by the daemon.
        keep_wakelock_while_upload: whether to keep a wakelock while uploading data.
        offline: when true, disable uploads and only store data locally.
        api_key: Neuracore API key for authentication.
        current_org_id: Organization ID for the authenticated user.
    """

    storage_limit: int | None = None
    bandwidth_limit: int | None = None
    path_to_store_record: str | None = None
    num_threads: int | None = None
    keep_wakelock_while_upload: bool | None = None
    offline: bool | None = None
    api_key: str | None = None
    current_org_id: str | None = None
