"""Client for the import sessions that track a dataset import's saved items."""

import os
from pathlib import Path

import requests
from neuracore_types import (
    ImportSession,
    ImportSessionCreateRequest,
    ImportSessionItemKeysPage,
    ImportSessionItemState,
)
from pydantic import BaseModel, ValidationError

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_errors import extract_error_detail, extract_error_type
from neuracore.core.utils.http_session import thread_local_session

from .exceptions import ImporterError


class ImportSessionError(ImporterError):
    """Raised when an import session request fails."""


class ImportSessionManifest(BaseModel):
    """The session and items an import selected, kept so a re-run resumes them."""

    import_session_id: str
    robot_id: str
    dataset_id: str
    item_keys: list[str]


def load_import_session_manifest(path: Path) -> ImportSessionManifest | None:
    """Load the manifest of an earlier import, or None if there is none.

    Raises:
        ImporterError: If the manifest exists but cannot be read.
    """
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ImporterError(f"Failed to read import manifest '{path}': {e}") from e
    try:
        return ImportSessionManifest.model_validate_json(contents)
    except ValidationError as e:
        raise ImporterError(
            f"Import manifest '{path}' is corrupt. Delete it to start a new "
            f"import of this dataset: {e}"
        ) from e


def save_import_session_manifest(path: Path, manifest: ImportSessionManifest) -> None:
    """Write the manifest atomically so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    os.replace(tmp_path, path)


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
    robot_id: str, dataset_id: str, expected_recording_count: int
) -> ImportSession:
    """Create an import session for the items selected for import.

    Args:
        robot_id: Robot the imported recordings belong to.
        dataset_id: Dataset the imported recordings are saved to.
        expected_recording_count: Number of items selected for import; the
            session is complete once that many distinct items are saved.

    Returns:
        The new import session.

    Raises:
        ImportSessionError: If the session could not be created.
    """
    request = ImportSessionCreateRequest(
        robot_id=robot_id,
        dataset_id=dataset_id,
        expected_recording_count=expected_recording_count,
    )
    try:
        # Creating a session is not idempotent, so transient failures are not
        # retried: a retry could leave an orphaned session behind.
        response = thread_local_session().post(
            _sessions_url(get_current_org()),
            json=request.model_dump(mode="json"),
            headers=get_auth().get_headers(),
        )
    except requests.exceptions.RequestException as e:
        raise ImportSessionError(f"Failed to create import session: {e}") from e
    _raise_for_status(response, "create import session")
    return ImportSession.model_validate(response.json())


def get_import_session(session_id: str) -> ImportSession:
    """Get an import session and how many of its items are saved.

    Raises:
        ImportSessionError: If the session could not be fetched.
    """
    try:
        response = thread_local_session(retry_transient=True).get(
            f"{_sessions_url(get_current_org())}/{session_id}",
            headers=get_auth().get_headers(),
        )
    except requests.exceptions.RequestException as e:
        raise ImportSessionError(f"Failed to get import session: {e}") from e
    _raise_for_status(response, "get import session")
    return ImportSession.model_validate(response.json())


def list_import_session_item_keys(
    session_id: str, state: ImportSessionItemState
) -> set[str]:
    """List every saved or in-flight item key of an import session.

    Raises:
        ImportSessionError: If a page could not be fetched.
    """
    url = f"{_sessions_url(get_current_org())}/{session_id}/item-keys"
    item_keys: set[str] = set()
    cursor: str | None = None
    while True:
        params = {"state": state.value}
        if cursor is not None:
            params["cursor"] = cursor
        try:
            response = thread_local_session(retry_transient=True).get(
                url, params=params, headers=get_auth().get_headers()
            )
        except requests.exceptions.RequestException as e:
            raise ImportSessionError(
                f"Failed to list import session item keys: {e}"
            ) from e
        _raise_for_status(response, "list import session item keys")
        page = ImportSessionItemKeysPage.model_validate(response.json())
        item_keys.update(page.item_keys)
        if page.next_cursor is None:
            return item_keys
        cursor = page.next_cursor
