"""Trace status updater for updating trace status on the backend."""

import asyncio
import logging
from dataclasses import dataclass

import aiohttp
from neuracore_types import (
    BatchedUpdateTraceRequest,
    RecordingDataTraceStatus,
    TraceStatusUpdates,
)

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TraceUpdateBatch:
    """Represents a batch of trace updates.

    Attributes:
        traces: Set of trace IDs in the batch.
        batch_full: Event that is set when the batch is full.
        includes_completed_trace: Event that is set when the batch includes a
            completed trace.
        completed: Future that is set when the batch update is complete.
    """

    traces: set[str]
    batch_full: asyncio.Event
    includes_completed_trace: asyncio.Event
    completed: asyncio.Future[bool]


class TraceStatusUpdater:
    """Trace status updater for updating trace status on the backend.

    The performance of the batching is a compromise between a few factors:
    we want large batches but we don't want to wait too long for them to fill


    So the logic is the batch is ready to send if any of the following is true:
     - it is full (50)
     - it has waited the cap of 4 seconds
     - it includes a completed trace and has waited 0.2 seconds

    Batches with a completed trace are treated as higher priority.


    Batches and deduplicates requests with updates for the same trace together.
    """

    MAXIMUM_REQUEST_ATTEMPTS = 2
    # Minimum request interval in seconds for the same trace

    # Updates with completion are more important, don't wait too long but we don't
    # need to spam for normal updates
    MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 4
    MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.2

    MAXIMUM_UPDATE_BATCH_SIZE = 50

    def __init__(
        self,
        client_session: aiohttp.ClientSession,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Initialize the trace status updater.

        Args:
            client_session: The aiohttp ClientSession to use for HTTP requests.
            loop: The event loop to use for async operations.
        """
        self._loop = loop or asyncio.get_running_loop()
        self._client_session = client_session
        self._auth = get_auth()
        self._org_id: str | None = None
        self._auth_headers: dict | None = None

        # Map of recording_id x trace_id -> TraceStatusUpdates
        self._in_progress_updates: dict[tuple[str, str], TraceStatusUpdates] = {}

        # Map of recording_id -> TraceUpdateBatch
        self._pending_update_batch: dict[str, TraceUpdateBatch] = {}

    async def _get_org_id(self) -> str:
        """Get the current org ID."""
        if self._org_id is None:
            self._org_id = await self._loop.run_in_executor(None, get_current_org)
        return self._org_id

    async def _get_auth_headers(self) -> dict:
        """Get the auth headers from the auth provider."""
        if self._auth_headers is None:
            self._auth_headers = await self._loop.run_in_executor(
                None, self._auth.get_headers
            )
        return self._auth_headers

    async def _wait_for_batch_ready(self, update_batch: TraceUpdateBatch) -> None:
        """Logic for waiting for a batch to be ready to send.

        if the batch includes a completed trace it must wait at least
        MINIMUM_REQUEST_INTERVAL_COMPLETE_S before sending the request

        or if the batch is full it can be sent

        of if the batch has waited MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S

        """

        async def _wait_for_batch_has_complete_trace() -> None:
            await asyncio.sleep(self.MINIMUM_REQUEST_INTERVAL_COMPLETE_S)
            await update_batch.includes_completed_trace.wait()

        await asyncio.wait(
            (
                asyncio.create_task(_wait_for_batch_has_complete_trace()),
                asyncio.create_task(update_batch.batch_full.wait()),
            ),
            timeout=self.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S,
            return_when=asyncio.FIRST_COMPLETED,
        )

    async def _send_batch_update_request(
        self, recording_id: str, body: BatchedUpdateTraceRequest
    ) -> bool:
        """Send a batch update request to the backend.

        Args:
            recording_id: The recording ID
            body: The batch update request
        """
        try:
            request_body = body.model_dump(mode="json", exclude_defaults=True)

            for attempt in range(self.MAXIMUM_REQUEST_ATTEMPTS):
                org_id, headers = await asyncio.gather(
                    self._get_org_id(),
                    self._get_auth_headers(),
                )

                async with self._client_session.put(
                    (
                        f"{API_URL}/org/{org_id}"
                        f"/recording/{recording_id}/traces/batch-update"
                    ),
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    assert isinstance(response, aiohttp.ClientResponse)
                    if response.status == 401 and attempt == 0:
                        logger.debug("Access token expired, refreshing token")
                        await self._loop.run_in_executor(None, self._auth.login)
                        self._auth_headers = None
                        continue

                    if response.status >= 400:
                        error = await response.text()
                        logger.warning(
                            f"Failed to update data trace: "
                            f"HTTP {response.status}: {error}"
                        )
                        return False

                    logger.debug(
                        f"Updated trace(s) {len(body.updates)}",
                    )
                    return True

            return False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Failed to update data trace: %s: %s", type(e).__name__, e)
            return False

    async def _update_backend_trace_record_batch(
        self,
        recording_id: str,
        update_batch: TraceUpdateBatch,
    ) -> None:
        """Update fields of a backend DataTrace.

        This method waits for the batch to be ready before sending the request and
        records its success.

        Args:
            recording_id: The recording ID
            update_batch: The batch of updates to apply
        """
        await self._wait_for_batch_ready(update_batch)

        if self._pending_update_batch.get(recording_id) is update_batch:
            self._pending_update_batch.pop(recording_id)

        # Collect the updates into a batch
        updates: dict[str, TraceStatusUpdates] = {}
        for trace_id in update_batch.traces:
            updates[trace_id] = self._in_progress_updates.pop((recording_id, trace_id))

        success = await self._send_batch_update_request(
            recording_id=recording_id, body=BatchedUpdateTraceRequest(updates=updates)
        )
        update_batch.completed.set_result(success)

    def _record_updates(
        self, recording_id: str, trace_id: str, updates: TraceStatusUpdates
    ) -> None:
        """Record updates for a trace.

        Stacks updates for the same trace.

        Args:
            recording_id: The recording ID
            trace_id: The trace ID
            updates: The updates to record
        """
        update_key = (recording_id, trace_id)
        existing_updates = self._in_progress_updates.get(update_key, None)

        if existing_updates:
            self._in_progress_updates[update_key] = existing_updates.stack(updates)
        else:
            self._in_progress_updates[update_key] = updates

    def _ensure_update_in_batch(
        self, recording_id: str, trace_id: str, updates: TraceStatusUpdates
    ) -> TraceUpdateBatch:
        """Ensure that there is a batch that includes this trace update.

        Args:
            recording_id: The recording ID
            trace_id: The trace ID
            updates: The updates to record
        """
        existing_batch = self._pending_update_batch.get(recording_id, None)

        if existing_batch:
            existing_batch.traces.add(trace_id)
            if len(existing_batch.traces) >= self.MAXIMUM_UPDATE_BATCH_SIZE:
                existing_batch.batch_full.set()
                self._pending_update_batch.pop(recording_id)
            if updates.status == RecordingDataTraceStatus.UPLOAD_COMPLETE:
                existing_batch.includes_completed_trace.set()
            return existing_batch

        new_batch = TraceUpdateBatch(
            traces={trace_id},
            batch_full=asyncio.Event(),
            completed=asyncio.Future(),
            includes_completed_trace=asyncio.Event(),
        )
        if updates.status == RecordingDataTraceStatus.UPLOAD_COMPLETE:
            new_batch.includes_completed_trace.set()

        self._pending_update_batch[recording_id] = new_batch

        asyncio.create_task(
            self._update_backend_trace_record_batch(
                recording_id=recording_id, update_batch=new_batch
            )
        )

        return new_batch

    async def _update_backend_trace_record(
        self,
        recording_id: str,
        trace_id: str,
        updates: TraceStatusUpdates,
        wait_for_completion: bool = False,
    ) -> bool:
        """Update fields of a backend DataTrace.

        Args:
            updates: JSON payload fields to update on backend DataTrace
            wait_for_completion: Whether to wait for the backend to be informed before
                continuing

        Returns:
            True if update succeeded within timeout
        """
        self._record_updates(
            recording_id=recording_id, trace_id=trace_id, updates=updates
        )
        update_batch = self._ensure_update_in_batch(
            recording_id=recording_id, trace_id=trace_id, updates=updates
        )

        if wait_for_completion:
            return await asyncio.shield(update_batch.completed)

        return True

    async def update_trace_progress(
        self,
        recording_id: str,
        trace_id: str,
        uploaded_bytes: int,
        wait_for_completion: bool = False,
    ) -> bool:
        """Update backend upload byte counters without changing status.

        Args:
            recording_id: The recording ID
            trace_id: The trace ID
            uploaded_bytes: The number of bytes uploaded

        Returns:
            True if update succeeded within timeout
        """
        return await self._update_backend_trace_record(
            recording_id=recording_id,
            trace_id=trace_id,
            updates=TraceStatusUpdates(
                uploaded_bytes=uploaded_bytes,
            ),
            wait_for_completion=wait_for_completion,
        )

    async def update_trace_completed(
        self,
        recording_id: str,
        trace_id: str,
        total_bytes: int,
        wait_for_completion: bool = True,
    ) -> bool:
        """Mark backend trace status as UPLOAD_COMPLETE with final counters.

        Called once the uploads is verified as complete.

        Args:
            recording_id: The recording ID
            trace_id: The trace ID
            total_bytes: The total number of bytes uploaded,
            wait_for_completion: Whether to wait for the backend to be informed before
                continuing

        Returns:
            True if update succeeded within timeout
        """
        return await self._update_backend_trace_record(
            recording_id=recording_id,
            trace_id=trace_id,
            updates=TraceStatusUpdates(
                status=RecordingDataTraceStatus.UPLOAD_COMPLETE,
                uploaded_bytes=total_bytes,
                total_bytes=total_bytes,
            ),
            wait_for_completion=wait_for_completion,
        )
