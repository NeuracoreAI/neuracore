"""End-to-end coverage for deleting recordings from a large cloned dataset.

This test deliberately exercises the real Neuracore API. It requires valid
credentials and the shared Freiburg dataset to be available to the test org.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import neuracore as nc
from neuracore import Dataset
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import PAGE_SIZE
from neuracore.core.utils.http_session import thread_local_session

SOURCE_DATASET_NAME = "Freiburg Franka Play - (TACO Play)"
DELETE_TIMEOUT_S = 180.0
POLL_INTERVAL_S = 2.0


def delete_recording(dataset: Dataset, recording_id: str) -> None:
    """Delete one recording through the API and surface the backend error."""
    response = thread_local_session().delete(
        f"{API_URL}/org/{dataset.org_id}/datasets/"
        f"{dataset.id}/recording/{recording_id}",
        headers=get_auth().get_headers(),
        timeout=30,
    )

    if not response.ok:
        # Include response content in CI output before raise_for_status discards
        # useful backend context.
        print(response.status_code, response.text)

    response.raise_for_status()


def recording_signature(recording: Any) -> tuple[Any, ...]:
    """Return all stable recording metadata that a clone must preserve."""
    return (
        recording.name,
        str(recording.robot_id),
        recording.instance,
        recording.start_time,
        recording.end_time,
        recording.end_time - recording.start_time,
        recording.total_bytes,
        frozenset(recording.data_types),
        # JSON produces a deterministic, hashable representation for Counter.
        json.dumps(recording.metadata, sort_keys=True, default=str),
    )


def dataset_recording_state(dataset: Dataset) -> tuple[set[str], Counter]:
    """Materialize a fresh dataset view as IDs plus metadata signatures."""
    recordings = list(dataset)
    return (
        {str(recording.id) for recording in recordings},
        Counter(recording_signature(recording) for recording in recordings),
    )


def wait_for_dataset_ids(
    dataset_id: str,
    expected_ids: set[str],
    *,
    timeout_s: float = DELETE_TIMEOUT_S,
) -> Dataset:
    """Poll fresh Dataset objects until eventual deletion is observable.

    Refetching is important: the Dataset used for deletion has already cached
    its recordings and therefore is not a reliable view of backend state.
    """
    deadline = time.monotonic() + timeout_s
    last_ids: set[str] | None = None

    while time.monotonic() < deadline:
        fresh_dataset = nc.get_dataset(id=dataset_id)
        last_ids, _ = dataset_recording_state(fresh_dataset)
        if last_ids == expected_ids:
            return fresh_dataset
        time.sleep(POLL_INTERVAL_S)

    missing = sorted(expected_ids - (last_ids or set()))[:10]
    unexpected = sorted((last_ids or set()) - expected_ids)[:10]
    raise AssertionError(
        f"Dataset {dataset_id} did not reach the expected recording state "
        f"within {timeout_s}s; expected={len(expected_ids)}, "
        f"actual={len(last_ids or set())}, missing={missing}, "
        f"unexpected={unexpected}"
    )


def cleanup_dataset(get_dataset: Callable[[], Dataset | None], name: str) -> None:
    """Best-effort cleanup that does not hide the test's original failure."""
    try:
        dataset = get_dataset()
        if dataset is not None and not dataset.deleted:
            dataset.delete()
    except Exception as exc:  # noqa: BLE001 - cleanup must preserve test failure
        warnings.warn(f"Failed to clean up dataset {name!r}: {exc}", stacklevel=2)


def test_delete_recordings_from_cloned_large_dataset() -> None:
    """Clone, partially delete, fully delete, merge, and verify isolation."""
    nc.login()

    # Unique names make reruns and concurrent CI jobs independent.
    run_id = uuid4().hex
    clone_name = f"delete-large-clone-{run_id}"
    merged_name = f"delete-large-merged-{run_id}"
    clone: Dataset | None = None
    merged: Dataset | None = None

    source_dataset = nc.get_dataset(SOURCE_DATASET_NAME)

    # Snapshot the source once so concurrent reads and later comparisons do not
    # accidentally rely on a mutable in-memory cache.
    source_recordings = list(source_dataset)
    source_ids = {str(recording.id) for recording in source_recordings}
    source_signatures = Counter(
        recording_signature(recording) for recording in source_recordings
    )
    source_signatures_by_id = {
        str(recording.id): recording_signature(recording)
        for recording in source_recordings
    }
    source_tags = Counter(source_dataset.tags)
    source_data_types = Counter(source_dataset.data_types)
    source_description = source_dataset.description
    source_size_bytes = source_dataset.size_bytes

    # The scenario only provides large-dataset/pagination coverage if it spans
    # more than one Dataset API page.
    assert len(source_ids) > PAGE_SIZE, (
        f"Expected {SOURCE_DATASET_NAME!r} to contain more than one page "
        f"({PAGE_SIZE} recordings), but found {len(source_ids)}"
    )

    try:
        clone = nc.clone_dataset(clone_name, source_dataset=source_dataset)

        # Dataset identity must be new while user-visible metadata and every
        # recording in the source snapshot must be preserved.
        assert clone.id != source_dataset.id
        assert clone.name == clone_name
        assert Counter(clone.tags) == source_tags
        assert Counter(clone.data_types) == source_data_types
        assert clone.description == source_description
        assert clone.size_bytes == source_size_bytes

        clone_ids, clone_signatures = dataset_recording_state(clone)
        assert len(clone) == len(source_ids)
        assert clone_ids == source_ids
        assert clone_signatures == source_signatures

        # Delete records around page boundaries plus both ends. This detects
        # off-by-one errors and deletion bugs hidden by pagination cursors.
        ordered_clone_ids = [str(recording.id) for recording in clone]
        selected_indexes = {
            0,
            PAGE_SIZE - 1,
            PAGE_SIZE,
            len(ordered_clone_ids) - 1,
        }
        selected_ids = {ordered_clone_ids[index] for index in selected_indexes}
        for recording_id in selected_ids:
            delete_recording(clone, recording_id)

        expected_remaining_ids = source_ids - selected_ids
        partially_deleted_clone = wait_for_dataset_ids(
            str(clone.id), expected_remaining_ids
        )
        partial_ids, partial_signatures = dataset_recording_state(
            partially_deleted_clone
        )
        assert partial_ids == expected_remaining_ids
        assert len(partially_deleted_clone) == len(source_ids) - len(selected_ids)
        assert partial_signatures == Counter(
            source_signatures_by_id[recording_id]
            for recording_id in expected_remaining_ids
        )

        # Delete all remaining recordings using the fresh post-mutation view.
        for recording_id in partial_ids:
            delete_recording(partially_deleted_clone, recording_id)

        empty_clone = wait_for_dataset_ids(str(clone.id), set())
        assert len(empty_clone) == 0
        assert list(empty_clone) == []

        # Merging an empty clone with the source should reproduce the source.
        # The direct empty-clone assertions above prevent merge deduplication
        # from masking failed DELETE requests.
        merged = nc.merge_datasets(
            name=merged_name,
            dataset_names=[clone_name, SOURCE_DATASET_NAME],
        )
        merged_ids, merged_signatures = dataset_recording_state(merged)
        assert merged.id not in {clone.id, source_dataset.id}
        assert merged.name == merged_name
        assert Counter(merged.tags) == source_tags
        assert len(merged) == len(source_ids)
        assert merged_ids == source_ids
        assert merged_signatures == source_signatures

        # Refetching the original verifies clone deletion and merge operations
        # never mutated the shared source dataset.
        fresh_source = nc.get_dataset(id=str(source_dataset.id))
        final_source_ids, final_source_signatures = dataset_recording_state(
            fresh_source
        )
        assert Counter(fresh_source.tags) == source_tags
        assert Counter(fresh_source.data_types) == source_data_types
        assert fresh_source.description == source_description
        assert fresh_source.size_bytes == source_size_bytes
        assert final_source_ids == source_ids
        assert final_source_signatures == source_signatures
    finally:
        # Always remove cloud resources, including after an assertion failure.
        cleanup_dataset(lambda: merged, merged_name)
        cleanup_dataset(
            lambda: nc.get_dataset(id=str(clone.id)) if clone is not None else None,
            clone_name,
        )
