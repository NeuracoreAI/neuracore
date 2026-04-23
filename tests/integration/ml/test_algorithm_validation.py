"""Integration tests for algorithm validation via the full GCP pipeline.

Uploads AllTypesModel to the Neuracore API, which triggers a GCE validation
job on the backend (GCE instance → run_in_venv → run_validation), then polls
until the backend reports a final validation status.
"""

import io
import logging
import time
import zipfile
from pathlib import Path

import requests

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL

logger = logging.getLogger(__name__)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "all_types_model"
ALGORITHM_NAME = "AllTypesModel Integration Test"
VALIDATION_TIMEOUT_MINUTES = 30
POLLING_INTERVAL_SECONDS = 60

EXPECTED_VALIDATION_CHECKLIST = {
    "successfully_loaded_file": True,
    "successfully_initialized_model": True,
    "successfully_configured_optimizer": True,
    "successfully_forward_pass": True,
    "successfully_backward_pass": True,
    "successfully_optimiser_step": True,
    "successfully_exported_model": True,
}


def _raise_for_status_with_detail(response: requests.Response) -> None:
    """Raise HTTP errors with the backend response body attached."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"{exc}\nResponse body: {response.text}",
            response=response,
            request=response.request,
        ) from exc


def _zip_fixture_directory(fixture_directory: Path) -> bytes:
    """Pack a directory tree into a ZIP archive and return the raw bytes.

    Excludes __pycache__ directories to keep the archive clean.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zip_archive:
        for file_path in sorted(fixture_directory.rglob("*")):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                archive_path = file_path.relative_to(fixture_directory)
                zip_archive.write(file_path, archive_path)
    return zip_buffer.getvalue()


def _upload_algorithm(
    org_id: str,
    algorithm_zip_bytes: bytes,
    algorithm_name: str,
) -> dict:
    """Upload a ZIP-packaged algorithm to the Neuracore API.

    Returns the Algorithm dict from the upload response. The backend will
    immediately spin up a GCE validation job and return status "validating".
    """
    auth = get_auth()
    response = requests.post(
        f"{API_URL}/org/{org_id}/algorithms",
        headers=auth.get_headers(),
        files={
            "algorithm_file": (
                f"{algorithm_name}.zip",
                algorithm_zip_bytes,
                "application/zip",
            )
        },
        data={
            "name": algorithm_name,
            "description": "AllTypesModel integration test — tests all data types",
        },
    )
    _raise_for_status_with_detail(response)
    return response.json()


def _get_algorithm(org_id: str, algorithm_id: str) -> dict:
    """Fetch the current Algorithm metadata from the Neuracore API."""
    auth = get_auth()
    response = requests.get(
        f"{API_URL}/org/{org_id}/algorithms/{algorithm_id}",
        headers=auth.get_headers(),
    )
    _raise_for_status_with_detail(response)
    return response.json()


def _delete_algorithm(org_id: str, algorithm_id: str) -> None:
    """Delete an algorithm by ID via the Neuracore API."""
    auth = get_auth()
    response = requests.delete(
        f"{API_URL}/org/{org_id}/algorithms/{algorithm_id}",
        headers=auth.get_headers(),
    )
    _raise_for_status_with_detail(response)


def _poll_until_validation_complete(
    org_id: str,
    algorithm_id: str,
    timeout_minutes: int,
    polling_interval_seconds: int,
) -> dict:
    """Poll the algorithm endpoint until validation leaves the "validating" state.

    Returns the final Algorithm dict once status is "available" or
    "validation_failed". Raises TimeoutError if the deadline is exceeded.
    """
    deadline = time.time() + timeout_minutes * 60
    algorithm_data: dict = {}
    while time.time() < deadline:
        algorithm_data = _get_algorithm(org_id, algorithm_id)
        current_status = algorithm_data["validation_status"]
        logger.info(
            "Algorithm %s — validation status: %s", algorithm_id, current_status
        )
        if current_status != "validating":
            return algorithm_data
        time.sleep(polling_interval_seconds)

    raise TimeoutError(
        f"Algorithm validation did not complete within {timeout_minutes} minutes. "
        f"Last known status: {algorithm_data.get('validation_status', 'unknown')}"
    )


def test_all_types_model_passes_gcp_validation() -> None:
    """Upload AllTypesModel and verify it passes the full GCP validation pipeline.

    Submits the fixture to the Neuracore API, waits for the backend GCE
    validation job to complete, and asserts the resulting checklist.
    """
    nc.login()
    org_id = get_current_org()

    algorithm_zip_bytes = _zip_fixture_directory(FIXTURE_DIR)
    upload_response = _upload_algorithm(
        org_id=org_id,
        algorithm_zip_bytes=algorithm_zip_bytes,
        algorithm_name=ALGORITHM_NAME,
    )
    algorithm_id = upload_response["id"]
    logger.info(
        "Uploaded algorithm — id: %s, initial status: %s",
        algorithm_id,
        upload_response["validation_status"],
    )

    try:
        final_algorithm_data = _poll_until_validation_complete(
            org_id=org_id,
            algorithm_id=algorithm_id,
            timeout_minutes=VALIDATION_TIMEOUT_MINUTES,
            polling_interval_seconds=POLLING_INTERVAL_SECONDS,
        )

        logger.info("Final algorithm data: %s", final_algorithm_data)

        final_validation_status = final_algorithm_data["validation_status"]
        final_validation_checklist = final_algorithm_data.get(
            "validation_checklist", {}
        )

        assert final_validation_status == "available", (
            f"Validation completed with unexpected status '{final_validation_status}'. "
            f"Checklist: {final_validation_checklist}"
        )
        assert final_validation_checklist == EXPECTED_VALIDATION_CHECKLIST, (
            f"Validation checklist mismatch.\n"
            f"  Expected: {EXPECTED_VALIDATION_CHECKLIST}\n"
            f"  Actual:   {final_validation_checklist}"
        )

    finally:
        logger.info("Cleaning up — deleting algorithm %s", algorithm_id)
        _delete_algorithm(org_id, algorithm_id)
