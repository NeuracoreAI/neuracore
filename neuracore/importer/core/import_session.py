"""Client for the import sessions that own a dataset import's recording sources."""

import logging
import threading

import requests
from neuracore_types import (
    ImportSession,
    ImportSessionCloseRequest,
    ImportSessionCreateRequest,
    ImportSessionStatus,
)

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_errors import extract_error_detail, extract_error_type
from neuracore.core.utils.http_session import thread_local_session

from .exceptions import ImporterError

logger = logging.getLogger(__name__)

_HEARTBEATS_PER_TIMEOUT = 10
_TERMINAL_HEARTBEAT_STATUS_CODES = frozenset({404, 409})


class ImportSessionError(ImporterError):
    """Raised when an import session request fails."""


def _sessions_url(org_id: str) -> str:
    return f"{API_URL}/org/{org_id}/import-sessions"


def _raise_for_status(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        detail = extract_error_detail(response)
        error_type = extract_error_type(response)
        message = f"Failed to {action}: HTTP {response.status_code}"
        if error_type:
            message += f" {error_type}"
        if detail:
            message += f": {detail}"
        if response.status_code in (404, 405) and detail in (
            "Not Found",
            "Method Not Allowed",
        ):
            message += (
                ". The Neuracore backend does not support import sessions; "
                "upgrade the backend or use an older neuracore release."
            )
        raise ImportSessionError(message) from e


def create_import_session(
    robot_id: str, dataset_id: str, worker_count: int
) -> ImportSession:
    """Create an import session that reserves recording sources for the workers.

    Args:
        robot_id: Robot the imported recordings belong to.
        dataset_id: Dataset the imported recordings are saved to.
        worker_count: Number of import workers that will record concurrently.

    Returns:
        The active import session.

    Raises:
        ImportSessionError: If the session could not be created.
    """
    request = ImportSessionCreateRequest(
        robot_id=robot_id, dataset_id=dataset_id, worker_count=worker_count
    )
    try:
        # Creating a session is not idempotent, so transient failures are not
        # retried: a retry could allocate a second, orphaned range.
        response = thread_local_session().post(
            _sessions_url(get_current_org()),
            json=request.model_dump(mode="json"),
            headers=get_auth().get_headers(),
        )
    except requests.exceptions.RequestException as e:
        raise ImportSessionError(f"Failed to create import session: {e}") from e
    _raise_for_status(response, "create import session")
    return ImportSession.model_validate(response.json())


def heartbeat_import_session(session_id: str) -> ImportSession:
    """Keep an import session usable for new recordings.

    Raises:
        ImportSessionError: If the heartbeat was rejected or failed.
    """
    try:
        response = thread_local_session(retry_transient=True).post(
            f"{_sessions_url(get_current_org())}/{session_id}/heartbeat",
            headers=get_auth().get_headers(),
        )
    except requests.exceptions.RequestException as e:
        raise ImportSessionError(f"Failed to heartbeat import session: {e}") from e
    _raise_for_status(response, "heartbeat import session")
    return ImportSession.model_validate(response.json())


def close_import_session(session_id: str, status: ImportSessionStatus) -> ImportSession:
    """Close an import session so it accepts no new recordings.

    Recordings already started keep uploading after the session closes.

    Raises:
        ImportSessionError: If the session could not be closed.
    """
    request = ImportSessionCloseRequest(status=status)
    try:
        response = thread_local_session(retry_transient=True).post(
            f"{_sessions_url(get_current_org())}/{session_id}/close",
            json=request.model_dump(mode="json"),
            headers=get_auth().get_headers(),
        )
    except requests.exceptions.RequestException as e:
        raise ImportSessionError(f"Failed to close import session: {e}") from e
    _raise_for_status(response, "close import session")
    return ImportSession.model_validate(response.json())


class ImportSessionHeartbeat:
    """Background thread that heartbeats an import session until stopped."""

    def __init__(self, session: ImportSession) -> None:
        """Prepare a heartbeat for the session.

        Args:
            session: The session to keep usable.
        """
        self._session_id = session.id
        self._interval_s = session.heartbeat_timeout_s / _HEARTBEATS_PER_TIMEOUT
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="import-session-heartbeat", daemon=True
        )

    def start(self) -> None:
        """Start sending heartbeats."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sending heartbeats and wait for the thread to exit."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                heartbeat_import_session(self._session_id)
            except ImportSessionError as e:
                cause = e.__cause__
                if (
                    isinstance(cause, requests.exceptions.HTTPError)
                    and cause.response is not None
                    and cause.response.status_code in _TERMINAL_HEARTBEAT_STATUS_CODES
                ):
                    logger.error(
                        "Import session %s is no longer usable, so new recordings "
                        "will be rejected: %s",
                        self._session_id,
                        e,
                    )
                    return
                logger.warning(
                    "Import session %s heartbeat failed: %s", self._session_id, e
                )
