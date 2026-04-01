"""Batch trace registration orchestration.

This module defines a queue-less RegistrationManager that treats persisted state
as the source of truth for registration backlog. The manager wakes on signal or
timer, claims a bounded batch of traces from state, calls a backend batch
registration API, then writes outcomes back through the state API.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

import aiohttp
from neuracore_types import DataType

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.data_daemon.const import API_URL
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import TraceRegistrationErrorCode, get_content_type

logger = logging.getLogger(__name__)

STATUS_TO_REGISTRATION_ERROR_CODE: dict[int, TraceRegistrationErrorCode] = {
    400: TraceRegistrationErrorCode.STREAM_REGISTRATION_ERROR,
    404: TraceRegistrationErrorCode.PENDING_RECORDING_NOT_FOUND,
    500: TraceRegistrationErrorCode.REGISTER_DATA_TRACE_FAILED,
}

# Retry policy for batch registration requests.
REGISTRATION_MAX_RETRIES = 5
REGISTRATION_MAX_BACKOFF_SECONDS = 16
REGISTRATION_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

LOSSY_VIDEO_NAME = "lossy.mp4"
LOSSLESS_VIDEO_NAME = "lossless.mp4"
TRACE_FILE = "trace.json"

# Backend contract for batch registration response.
REGISTERED_TRACES_KEY = "registered_traces"
FAILED_TRACES_KEY = "failed_traces"


def get_cloud_file_list(
    data_type: DataType, data_type_name: str
) -> list[dict[str, str]]:
    """Derive the cloud file list for a trace from its data type."""
    prefix = f"{data_type.value}/{data_type_name}"
    files = []
    if get_content_type(data_type) == "RGB":
        files.append(
            {"filepath": f"{prefix}/{LOSSY_VIDEO_NAME}", "content_type": "video/mp4"}
        )
        files.append(
            {"filepath": f"{prefix}/{LOSSLESS_VIDEO_NAME}", "content_type": "video/mp4"}
        )
    files.append(
        {"filepath": f"{prefix}/{TRACE_FILE}", "content_type": "application/json"}
    )
    return files


@dataclass(frozen=True)
class RegistrationCandidate:
    """Trace payload needed for backend registration."""

    trace_id: str
    recording_id: str
    data_type: DataType
    data_type_name: str


@dataclass(frozen=True)
class RegistrationBatchOutcome:
    """Outcome of a backend batch registration attempt."""

    registered_trace_ids: list[str]
    failed_trace_ids: list[str]
    error_code: TraceRegistrationErrorCode | None = None
    error_message: str | None = None
    upload_session_uris: dict[str, dict[str, str]] | None = None


class RegistrationStateAPI(Protocol):
    """Narrow state interface needed by RegistrationManager."""

    async def claim_traces_for_registration(
        self, limit: int, max_wait_s: float
    ) -> list[RegistrationCandidate]:
        """Atomically claim traces ready for registration."""

    async def mark_traces_registered(self, trace_ids: list[str]) -> list[str]:
        """Mark traces as successfully registered and return updated IDs."""

    async def mark_traces_registration_failed(
        self, trace_ids: list[str], error_message: str
    ) -> None:
        """Mark traces as registration failed/retrying."""

    async def emit_ready_for_upload(
        self,
        trace_ids: list[str],
        upload_session_uris: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Emit upload-ready events for traces that passed registration."""

    async def mark_traces_registering(self, trace_ids: list[str]) -> list[str]:
        """Mark traces as registering and return updated IDs."""


class RegistrationManager:
    """Background manager for batched trace registration."""

    def __init__(
        self,
        *,
        client_session: aiohttp.ClientSession,
        state_api: RegistrationStateAPI,
        batch_size: int = 200,
        max_wait_s: float = 1.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        """Initialize a queue-less registration manager.

        Args:
            client_session: Shared HTTP session used for backend registration calls.
            state_api: State facade that owns DB access and transitions.
            batch_size: Max traces to claim/register per drain iteration.
            max_wait_s: Max age threshold before flushing short batches.
            poll_interval_s: Safety polling interval when no wake signal arrives.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_wait_s < 0:
            raise ValueError("max_wait_s must be >= 0")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")

        self._client_session = client_session
        self._state_api = state_api
        self._batch_size = batch_size
        self._max_wait_s = max_wait_s
        self._poll_interval_s = poll_interval_s

        self._wake_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

        self._emitter = get_emitter()
        self._emitter.on(
            Emitter.TRACE_REGISTRATION_AVAILABLE, self.notify_work_available
        )

    def start(self) -> None:
        """Start the background worker."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        logger.info(
            (
                "RegistrationManager starting (batch_size=%d, max_wait_s=%.2f, "
                "poll_interval_s=%.2f)"
            ),
            self._batch_size,
            self._max_wait_s,
            self._poll_interval_s,
        )
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._run(), name="registration-manager")
        self._worker_task.add_done_callback(self._log_worker_exit)
        # Drain any backlog that may already be present at startup.
        self._wake_event.set()

    def _log_worker_exit(self, task: asyncio.Task) -> None:
        """Log unexpected worker exits so failures are visible in daemon logs."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.info("RegistrationManager worker cancelled")
            return
        if exc is not None:
            logger.exception("RegistrationManager worker failed", exc_info=exc)

    async def shutdown(self) -> None:
        """Stop the background worker and wait for clean exit."""
        self._shutdown_event.set()
        self._wake_event.set()

        if self._worker_task is None:
            return

        try:
            await self._worker_task
        finally:
            self._worker_task = None

    def notify_work_available(self) -> None:
        """Wake worker after new traces become registration-eligible."""
        logger.debug("RegistrationManager wake requested")
        self._wake_event.set()

    async def _run(self) -> None:
        """Worker loop: wake then drain all currently-claimable work."""
        while not self._shutdown_event.is_set():
            await self._wait_for_wakeup_or_poll()
            if self._shutdown_event.is_set():
                break
            await self._drain_claimable_traces()

        logger.info("RegistrationManager stopped")

    async def _wait_for_wakeup_or_poll(self) -> None:
        try:
            await asyncio.wait_for(
                self._wake_event.wait(), timeout=self._poll_interval_s
            )
            logger.debug("RegistrationManager woke from event signal")
        except asyncio.TimeoutError:
            logger.debug("RegistrationManager woke from poll timeout")
            return
        finally:
            self._wake_event.clear()

    async def _drain_claimable_traces(self) -> None:
        """Claim/register batches until state has no more eligible traces."""
        while not self._shutdown_event.is_set():
            traces = await self._state_api.claim_traces_for_registration(
                limit=self._batch_size,
                max_wait_s=self._max_wait_s,
            )
            if not traces:
                logger.debug("RegistrationManager claim returned no traces")
                return
            logger.debug(
                "RegistrationManager claimed %d traces for batch registration",
                len(traces),
            )

            await self._register_and_record_outcome(traces)

    async def _register_data_trace_batch(
        self, traces: list[RegistrationCandidate]
    ) -> RegistrationBatchOutcome:
        """Register a batch of backend DataTraces.

        Args:
            traces: Traces to register.

        Returns:
            RegistrationBatchOutcome with success/failure partition.
        """
        if not traces:
            return RegistrationBatchOutcome(
                registered_trace_ids=[],
                failed_trace_ids=[],
                error_code=None,
                error_message=None,
            )

        try:
            loop = asyncio.get_running_loop()
            auth = get_auth()
            org_id, headers = await asyncio.gather(
                loop.run_in_executor(None, get_current_org),
                loop.run_in_executor(None, auth.get_headers),
            )

            requested_trace_ids = {trace.trace_id for trace in traces}
            logger.debug(
                "Submitting registration batch (size=%d, sample_ids=%s)",
                len(requested_trace_ids),
                sorted(list(requested_trace_ids))[:5],
            )
            payload = {
                "traces": [
                    {
                        "recording_id": trace.recording_id,
                        "data_type": trace.data_type.value,
                        "trace_id": str(trace.trace_id),
                        "cloud_files": get_cloud_file_list(
                            trace.data_type, trace.data_type_name
                        ),
                    }
                    for trace in traces
                ]
            }
            endpoint = f"{API_URL}/org/{org_id}/recording/traces/batch-register"
            refreshed_auth = False

            for attempt in range(REGISTRATION_MAX_RETRIES):
                try:
                    async with self._client_session.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        assert isinstance(response, aiohttp.ClientResponse)
                        if response.status == 401 and not refreshed_auth:
                            logger.debug("Access token expired, refreshing token")
                            await loop.run_in_executor(None, auth.login)
                            headers = await loop.run_in_executor(None, auth.get_headers)
                            refreshed_auth = True
                            continue

                        if response.status >= 400:
                            error = await response.text()
                            logger.warning(
                                (
                                    "Registration batch failed with HTTP %d "
                                    "(size=%d, body=%s, attempt=%d/%d)"
                                ),
                                response.status,
                                len(requested_trace_ids),
                                error[:300],
                                attempt + 1,
                                REGISTRATION_MAX_RETRIES,
                            )
                            if (
                                response.status in REGISTRATION_RETRYABLE_STATUS_CODES
                                and attempt < REGISTRATION_MAX_RETRIES - 1
                            ):
                                delay = min(
                                    2**attempt, REGISTRATION_MAX_BACKOFF_SECONDS
                                )
                                await asyncio.sleep(delay)
                                continue
                            return RegistrationBatchOutcome(
                                registered_trace_ids=[],
                                failed_trace_ids=list(requested_trace_ids),
                                error_code=STATUS_TO_REGISTRATION_ERROR_CODE.get(
                                    response.status, TraceRegistrationErrorCode.UNKNOWN
                                ),
                                error_message=error,
                            )

                        response_payload = await response.json()
                        if not isinstance(response_payload, dict):
                            return RegistrationBatchOutcome(
                                registered_trace_ids=[],
                                failed_trace_ids=list(requested_trace_ids),
                                error_code=TraceRegistrationErrorCode.UNKNOWN,
                                error_message=(
                                    "Invalid batch-register response payload "
                                    f"type={type(response_payload).__name__}"
                                ),
                            )

                        registered_traces = response_payload.get(
                            REGISTERED_TRACES_KEY, []
                        )
                        failed_traces = response_payload.get(FAILED_TRACES_KEY, [])

                        registered_trace_ids = [
                            str(entry["trace_id"])
                            for entry in registered_traces
                            if str(entry.get("trace_id")) in requested_trace_ids
                        ]
                        upload_session_uris: dict[str, dict[str, str]] = {
                            str(entry["trace_id"]): entry["upload_session_uris"]
                            for entry in registered_traces
                            if str(entry.get("trace_id")) in requested_trace_ids
                            and entry.get("upload_session_uris")
                        }
                        failed_trace_ids = [
                            str(entry["trace_id"])
                            for entry in failed_traces
                            if str(entry.get("trace_id")) in requested_trace_ids
                        ]
                        failed_errors = [
                            entry.get("error", "Unknown error")
                            for entry in failed_traces
                            if str(entry.get("trace_id")) in requested_trace_ids
                        ]

                        unresolved = (
                            requested_trace_ids
                            - set(registered_trace_ids)
                            - set(failed_trace_ids)
                        )
                        if unresolved:
                            failed_trace_ids.extend(sorted(unresolved))

                        error_summary = (
                            "; ".join(failed_errors) if failed_errors else None
                        )

                        return RegistrationBatchOutcome(
                            registered_trace_ids=registered_trace_ids,
                            failed_trace_ids=failed_trace_ids,
                            error_code=(
                                TraceRegistrationErrorCode.UNKNOWN
                                if failed_trace_ids
                                else None
                            ),
                            error_message=(error_summary if failed_trace_ids else None),
                            upload_session_uris=upload_session_uris or None,
                        )
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        (
                            "Registration batch request error "
                            "(size=%d, attempt=%d/%d, error=%s)"
                        ),
                        len(requested_trace_ids),
                        attempt + 1,
                        REGISTRATION_MAX_RETRIES,
                        exc,
                    )
                    if attempt < REGISTRATION_MAX_RETRIES - 1:
                        delay = min(2**attempt, REGISTRATION_MAX_BACKOFF_SECONDS)
                        await asyncio.sleep(delay)
                        continue
                    return RegistrationBatchOutcome(
                        registered_trace_ids=[],
                        failed_trace_ids=list(requested_trace_ids),
                        error_code=TraceRegistrationErrorCode.NETWORK_ERROR,
                        error_message=str(exc),
                    )

            return RegistrationBatchOutcome(
                registered_trace_ids=[],
                failed_trace_ids=list(requested_trace_ids),
                error_code=TraceRegistrationErrorCode.NETWORK_ERROR,
                error_message=(
                    "Unable to register traces after "
                    f"{REGISTRATION_MAX_RETRIES} attempts"
                ),
            )

        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            trace_ids = [trace.trace_id for trace in traces]
            logger.error(f"Failed to register data trace batch: {e}")
            return RegistrationBatchOutcome(
                registered_trace_ids=[],
                failed_trace_ids=trace_ids,
                error_code=TraceRegistrationErrorCode.NETWORK_ERROR,
                error_message=str(e),
            )

    async def _register_and_record_outcome(
        self, traces: list[RegistrationCandidate]
    ) -> None:
        trace_ids = [trace.trace_id for trace in traces]
        logger.debug(
            (
                "Processing registration outcome for claimed traces "
                "(size=%d, sample_ids=%s)"
            ),
            len(trace_ids),
            trace_ids[:5],
        )

        try:
            outcome = await self._register_data_trace_batch(traces)
        except Exception as exc:
            logger.exception("Batch registration failed unexpectedly")
            await self._state_api.mark_traces_registration_failed(
                trace_ids=trace_ids,
                error_message=(
                    f"{TraceRegistrationErrorCode.UNKNOWN.value}: "
                    f"Registration batch exception: {exc}"
                ),
            )
            return

        if outcome.registered_trace_ids:
            registered_ids = await self._state_api.mark_traces_registered(
                outcome.registered_trace_ids
            )
            logger.debug(
                "Registration batch success persisted (registered=%d, sample_ids=%s)",
                len(registered_ids),
                registered_ids[:5],
            )
            if registered_ids:
                await self._state_api.emit_ready_for_upload(
                    registered_ids, outcome.upload_session_uris
                )
                logger.debug(
                    "Emitted READY_FOR_UPLOAD for registered traces (count=%d)",
                    len(registered_ids),
                )

        if outcome.failed_trace_ids:
            logger.warning(
                (
                    "Registration batch had failed traces "
                    "(count=%d, sample_ids=%s, error=%s)"
                ),
                len(outcome.failed_trace_ids),
                outcome.failed_trace_ids[:5],
                outcome.error_message,
            )
            error_code = outcome.error_code or TraceRegistrationErrorCode.UNKNOWN
            await self._state_api.mark_traces_registration_failed(
                trace_ids=outcome.failed_trace_ids,
                error_message=(
                    f"{error_code.value}: "
                    f"{outcome.error_message or 'Registration failed'}"
                ),
            )
