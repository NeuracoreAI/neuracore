"""Shared utilities for ML integration tests."""

import logging
import uuid
from dataclasses import dataclass

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_session import thread_local_session

logger = logging.getLogger(__name__)

COMPLETED_STATUS = "COMPLETED"
_DATASETS_LIST_PAGE_SIZE = 30


def unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class SelectedRun:
    """A training run chosen for inference."""

    name: str
    id: str
    launch_time: float


def resolve_latest_completed_run(name_prefix: str) -> SelectedRun:
    """Find the most recent COMPLETED training run under ``name_prefix``.

    Scans all training jobs, keeps the ones that are COMPLETED and whose name
    starts with ``name_prefix``, and returns the one with the latest
    ``launch_time``. The selection (and the candidates considered) is logged
    prominently so it is unmistakable which run inference is using.

    Raises:
        AssertionError: If no COMPLETED run matching the prefix is found.
    """
    jobs = nc.get_training_jobs()

    completed_candidates: list[SelectedRun] = []
    rejected: list[str] = []
    for job in jobs:
        name = job.get("name")
        job_id = job.get("id")
        status = job.get("status")
        launch_time = job.get("launch_time") or 0.0
        if not isinstance(name, str) or not name.startswith(name_prefix):
            continue
        if status != COMPLETED_STATUS:
            rejected.append(f"name={name} id={job_id} status={status} (non-COMPLETED)")
            continue
        completed_candidates.append(
            SelectedRun(name=name, id=job_id, launch_time=float(launch_time))
        )

    logger.info(
        "[INFERENCE] Scanned %d training jobs for prefix %r: "
        "%d COMPLETED candidate(s), %d rejected.",
        len(jobs),
        name_prefix,
        len(completed_candidates),
        len(rejected),
    )
    for candidate in completed_candidates:
        logger.info(
            "[INFERENCE] COMPLETED candidate: name=%s id=%s launch_time=%s",
            candidate.name,
            candidate.id,
            candidate.launch_time,
        )
    for rejection in rejected:
        logger.info("[INFERENCE] Rejected candidate: %s", rejection)

    assert completed_candidates, (
        f"No COMPLETED training run found with name prefix {name_prefix!r}. "
        "The end-to-end flow test must run (and complete training) before the "
        "inference tests. Candidate names matching the prefix: "
        f"{[r for r in rejected] or 'none'}"
    )

    selected = max(completed_candidates, key=lambda run: run.launch_time)
    logger.info(
        "===== [INFERENCE] SELECTED TRAINING RUN: name=%s id=%s status=%s "
        "launch_time=%s (chosen from %d COMPLETED candidates) =====",
        selected.name,
        selected.id,
        COMPLETED_STATUS,
        selected.launch_time,
        len(completed_candidates),
    )
    return selected


def prune_training_runs_except(name_prefix: str, keep_id: str) -> None:
    """Delete training runs matching ``name_prefix`` except ``keep_id``.

    The kept run is intentionally retained (not deleted) so it remains
    available as a known-good model for future, separate test sessions: if a
    later run's training fails, the inference test can fall back to this
    previously successful run. All other prefixed runs (older COMPLETED runs
    that have been superseded, plus FAILED leftovers) are pruned to avoid
    unbounded accumulation of training jobs.
    """
    jobs = nc.get_training_jobs()
    pruned = 0
    for job in jobs:
        name = job.get("name")
        job_id = job.get("id")
        if not isinstance(name, str) or not name.startswith(name_prefix):
            continue
        if job_id == keep_id:
            continue
        try:
            nc.delete_training_job(job_id)
            pruned += 1
            logger.info(
                "[INFERENCE] Pruned superseded training run name=%s id=%s",
                name,
                job_id,
            )
        except Exception:
            logger.warning(f"Failed to prune training job {job_id}", exc_info=True)
    logger.info(
        "[INFERENCE] Retained known-good run id=%s; pruned %d superseded run(s) "
        "matching prefix %r.",
        keep_id,
        pruned,
        name_prefix,
    )


def list_org_datasets(org_id: str) -> list[dict]:
    """Fetch all datasets for an org via the paginated list API."""
    auth = get_auth()
    session = thread_local_session()
    headers = auth.get_headers()
    datasets: list[dict] = []
    start_after: dict | None = None

    while True:
        response = session.post(
            f"{API_URL}/org/{org_id}/datasets/list",
            params={"limit": _DATASETS_LIST_PAGE_SIZE},
            json=start_after,
            headers=headers,
        )
        response.raise_for_status()
        page = response.json()
        batch = page.get("data", [])
        if not batch:
            break
        datasets.extend(batch)
        if len(batch) < page.get("limit", _DATASETS_LIST_PAGE_SIZE):
            break
        start_after = page.get("start_after") or batch[-1]

    return datasets


def prune_datasets_by_name_prefix(name_prefix: str) -> int:
    """Delete datasets whose name starts with ``{name_prefix}_``.

    Returns the number of datasets deleted.
    """
    org_id = get_current_org()
    datasets = list_org_datasets(org_id)

    deleted = 0
    prefix = f"{name_prefix}_"
    for dataset_json in datasets:
        name = dataset_json.get("name")
        dataset_id = dataset_json.get("id")
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        if not isinstance(dataset_id, str):
            continue
        try:
            nc.get_dataset(id=dataset_id).delete()
            deleted += 1
            logger.info(
                "[RESUME] Deleted training-flow dataset name=%s id=%s",
                name,
                dataset_id,
            )
        except Exception:
            logger.warning(f"Failed to delete dataset {dataset_id}", exc_info=True)
    return deleted


def cleanup_training_flow_datasets(
    *,
    collected_prefix: str,
    merged_prefix: str,
) -> None:
    """Delete collected and merged datasets created by the training flow test."""
    deleted = 0
    for prefix in (collected_prefix, merged_prefix):
        deleted += prune_datasets_by_name_prefix(prefix)
    logger.info("[RESUME] Cleaned up %d training-flow dataset(s)", deleted)
