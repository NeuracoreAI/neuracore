"""Data synthesis job management utilities.

Data synthesis runs as its own service, reached under the /synthesis prefix
rather than the backend's org routes.

Data synthesis expands a dataset by regenerating the imagery of its recordings
while leaving every other modality untouched, so the actions recorded against
those frames stay valid.
"""

from typing import Any

import requests
from neuracore_types import (
    DataSynthesisAlgorithmConfig,
    DataSynthesisJobRequest,
    DataSynthesisPreviewRequest,
)

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.utils.http_errors import extract_error_detail
from neuracore.core.utils.http_session import thread_local_session

from ..core.auth import get_auth
from ..core.const import SYNTHESIS_API_URL
from ..core.data.dataset import Dataset
from .training import _resolve_next_name


def start_data_synthesis_run(
    name: str,
    dataset_name: str,
    algorithm_name: str,
    algorithm_config: dict[str, Any],
    scaling_factor: int = 2,
    output_dataset_name: str | None = None,
    include_original: bool = False,
    name_auto_increment: bool = False,
) -> dict:
    """Start a new data synthesis run.

    Args:
        name: Name of the synthesis job.
        dataset_name: Name of the dataset to expand.
        algorithm_name: Name of the algorithm to use, as
            get_data_synthesis_algorithms names it.
        algorithm_config: Configuration parameters for the algorithm. What each
            algorithm takes, and what it defaults, is described by
            get_data_synthesis_algorithms; anything with a default may be left
            out. The service checks these against the algorithm and says what
            is wrong with them.
        scaling_factor: How many recordings to produce per original. A whole
            number from 1 to 10.
        output_dataset_name: Name of the dataset to write results into.
            Defaults to the source dataset's name with a suffix.
        include_original: Whether the output dataset should also contain the
            source recordings. They are referenced, never copied.
        name_auto_increment: If True and a job with this name already exists,
            use name_1, name_2, ... instead of duplicating the name.

    Returns:
        dict: Job data including job ID and status.

    Raises:
        ValueError: If the dataset is not found, or the algorithm and its
            configuration are not something the service can run.
        requests.exceptions.HTTPError: If the API request fails.
        requests.exceptions.RequestException: If there is a network problem.
        ConfigError: If there is an error trying to get the current org.
    """
    if name_auto_increment:
        jobs = get_data_synthesis_jobs()
        existing_names = {
            job["name"] for job in jobs if isinstance(job.get("name"), str)
        }
        name = _resolve_next_name(name, existing_names)

    dataset = Dataset.get_by_name(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    data = DataSynthesisJobRequest(
        name=name,
        dataset_id=dataset.id,
        output_dataset_name=output_dataset_name or f"{dataset.name} (synthetic)",
        algorithm=DataSynthesisAlgorithmConfig(
            name=algorithm_name, parameters=algorithm_config
        ),
        scaling_factor=scaling_factor,
        include_original=include_original,
    )

    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.post(
        f"{SYNTHESIS_API_URL}/org/{org_id}/jobs",
        headers=auth.get_headers(),
        json=data.model_dump(mode="json"),
        timeout=(5.0, 60.0),
    )
    if not response.ok:
        detail = extract_error_detail(response)
        raise ValueError(f"Error starting data synthesis run: {detail}")
    return response.json()


def generate_data_synthesis_preview(
    dataset_name: str,
    algorithm_name: str,
    algorithm_config: dict[str, Any],
    recording_index: int = 0,
    frame_index: int = 0,
) -> dict:
    """Generate example frames showing what a synthesis run would produce.

    Every camera of the sampled recording is generated, because a prompt that
    segments the right object from one viewpoint can miss from another.

    Args:
        dataset_name: Name of the dataset to sample from.
        algorithm_name: Name of the algorithm to use, as
            get_data_synthesis_algorithms names it.
        algorithm_config: Configuration parameters for that algorithm.
        recording_index: Which recording of the dataset to sample from.
        frame_index: Which frame of each camera's video to sample.

    Returns:
        dict: A ``frames`` entry per camera, each holding the camera name and
        URLs of the original and generated frames, plus a digest of the
        configuration that produced them.

    Raises:
        ValueError: If the dataset is not found, the algorithm and its
            configuration are not something the service can run, or generating
            the preview failed.
        requests.exceptions.RequestException: If there is a network problem.
        ConfigError: If there is an error trying to get the current org.
    """
    dataset = Dataset.get_by_name(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    data = DataSynthesisPreviewRequest(
        dataset_id=dataset.id,
        algorithm=DataSynthesisAlgorithmConfig(
            name=algorithm_name, parameters=algorithm_config
        ),
        recording_index=recording_index,
        frame_index=frame_index,
    )

    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    # A diffusion preview runs the model on CPU once per camera, so give it
    # several minutes before giving up.
    response = session.post(
        f"{SYNTHESIS_API_URL}/org/{org_id}/preview",
        headers=auth.get_headers(),
        json=data.model_dump(mode="json"),
        timeout=(5.0, 900.0),
    )
    if not response.ok:
        detail = extract_error_detail(response)
        raise ValueError(f"Error generating data synthesis preview: {detail}")
    return response.json()


def get_data_synthesis_algorithms() -> list[dict]:
    """List the algorithms a data synthesis run can be started with.

    Read out of the package the synthesis worker runs, so this is the only
    thing that says what is on offer and what each algorithm takes.

    Returns:
        One entry per algorithm, each with its ``name``, a ``title`` and a
        ``description``, and its parameters as JSON Schema under
        ``json_schema`` -- their types, defaults, choices, which are required,
        and what each one means.

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.get(
        f"{SYNTHESIS_API_URL}/org/{org_id}/algorithms",
        headers=auth.get_headers(),
    )
    response.raise_for_status()
    return response.json()


def get_data_synthesis_jobs() -> list[dict]:
    """List all data synthesis jobs for the current organization.

    Returns:
        List of job dicts (id, name, status, etc.).

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.get(
        f"{SYNTHESIS_API_URL}/org/{org_id}/jobs",
        headers=auth.get_headers(),
    )
    response.raise_for_status()
    return response.json()


def get_data_synthesis_job_data(job_id: str) -> dict:
    """Retrieve complete data for a data synthesis job.

    Args:
        job_id: The ID of the job.

    Returns:
        dict: Complete job data including status, configuration, and progress.

    Raises:
        ValueError: If the job is not found.
        requests.exceptions.HTTPError: If the API request returns an error status.
        requests.exceptions.RequestException: If there is a network problem.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    try:
        response = session.get(
            f"{SYNTHESIS_API_URL}/org/{org_id}/jobs/{job_id}",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            raise ValueError("Job not found") from error
        raise


def get_data_synthesis_job_status(job_id: str) -> str:
    """Get the current status of a data synthesis job.

    Args:
        job_id: The ID of the job.

    Returns:
        str: Current status, e.g. "RUNNING", "COMPLETED" or "FAILED".

    Raises:
        ValueError: If the job is not found.
        requests.exceptions.RequestException: If there is a network problem.
    """
    return get_data_synthesis_job_data(job_id)["status"]


def cancel_data_synthesis_job(job_id: str) -> None:
    """Stop a running data synthesis job and release its machine.

    Args:
        job_id: The ID of the job to cancel.

    Raises:
        ValueError: If the job is not found or cancelling it failed.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.post(
        f"{SYNTHESIS_API_URL}/org/{org_id}/jobs/{job_id}/cancel",
        headers=auth.get_headers(),
    )
    if not response.ok:
        detail = extract_error_detail(response)
        raise ValueError(f"Error cancelling data synthesis job: {detail}")


def delete_data_synthesis_job(job_id: str) -> None:
    """Delete a data synthesis job and release its machine.

    The dataset the job produced is left in place.

    Args:
        job_id: The ID of the job to delete.

    Raises:
        ValueError: If the job is not found or deleting it failed.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.delete(
        f"{SYNTHESIS_API_URL}/org/{org_id}/jobs/{job_id}",
        headers=auth.get_headers(),
    )
    if not response.ok:
        detail = extract_error_detail(response)
        raise ValueError(f"Error deleting data synthesis job: {detail}")
