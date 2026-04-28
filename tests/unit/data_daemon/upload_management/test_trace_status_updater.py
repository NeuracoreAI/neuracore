import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio
from neuracore_types import RecordingDataTraceStatus

from neuracore.data_daemon.upload_management.trace_status_updater import (
    TraceStatusUpdater,
)

# --- Fixtures ---


@pytest.fixture
def mock_auth_and_org():
    """
    Fixtures to handle the global imports inside the class.
    Adapting your provided logic to the core modules used by the class.
    """
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_status_updater.get_auth"
        ) as mock_get_auth,
        patch(
            "neuracore.data_daemon.upload_management.trace_status_updater.get_current_org",
            return_value="test-org",
        ) as mock_get_org,
    ):
        auth_instance = MagicMock()
        auth_instance.get_headers.return_value = {"Authorization": "Bearer test-token"}
        auth_instance.login = MagicMock()
        mock_get_auth.return_value = auth_instance

        yield {
            "auth": auth_instance,
            "get_org": mock_get_org,
            "mock_get_auth": mock_get_auth,
        }


@pytest.fixture
def mock_client_session():
    # Use an AsyncMock to represent the session
    session = AsyncMock(spec=aiohttp.ClientSession)
    # Mock the .put() context manager
    session.put.return_value.__aenter__.return_value = AsyncMock(
        spec=aiohttp.ClientResponse
    )
    yield session


@pytest_asyncio.fixture
async def trace_status_updater(mock_client_session, mock_auth_and_org):
    # Initialize the actual class with the mocked session
    updater = TraceStatusUpdater(client_session=mock_client_session)
    yield updater
    await updater.shutdown()


# --- Tests ---


@pytest.mark.asyncio
async def test_update_trace_progress_batches_requests(
    trace_status_updater, mock_client_session
):
    """Verify that multiple calls are batched into one network request."""
    # Setup response
    response = mock_client_session.put.return_value.__aenter__.return_value
    response.status = 200

    # Act: Fire two updates for the same recording
    # We don't wait for completion on the first one so the batch can accumulate

    await trace_status_updater.update_trace_progress(
        "rec-1", "trace-1", 100, wait_for_completion=False
    )
    await trace_status_updater.update_trace_progress(
        "rec-1", "trace-2", 200, wait_for_completion=True
    )

    # Assert: Only one PUT request should have been made for that recording_id
    assert mock_client_session.put.call_count == 1

    # Check that the body contains both trace updates
    call_args = mock_client_session.put.call_args
    json_data = call_args.kwargs["json"]
    assert "trace-1" in json_data["updates"]
    assert "trace-2" in json_data["updates"]
    assert json_data["updates"]["trace-1"]["uploaded_bytes"] == 100


@pytest.mark.asyncio
async def test_batch_is_sent_even_if_not_full(
    trace_status_updater, mock_client_session
):
    # Speed up automatic batch sending
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.01
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 0.01
    # Setup response
    response = mock_client_session.put.return_value.__aenter__.return_value
    response.status = 200

    # Act: Fire two updates for the same recording
    # We don't wait for completion on the first one so the batch can accumulate
    await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 100, wait_for_completion=False
        ),
        timeout=0.01,
    )

    await asyncio.sleep(2 * TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S)

    assert mock_client_session.put.call_count == 1

    await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 100, wait_for_completion=False
        ),
        timeout=0.01,
    )

    await asyncio.sleep(2 * TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S)

    assert mock_client_session.put.call_count == 2


@pytest.mark.asyncio
async def test_batch_is_sent_once_full(trace_status_updater, mock_client_session):
    # Disable automatic batch sending
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 1000
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 1000
    # Setup response
    response = mock_client_session.put.return_value.__aenter__.return_value
    response.status = 200

    num_batches = 3
    # Act
    try:
        for batch in range(num_batches):
            for i in range(TraceStatusUpdater.MAXIMUM_UPDATE_BATCH_SIZE):
                last_trace = (
                    i == TraceStatusUpdater.MAXIMUM_UPDATE_BATCH_SIZE - 1
                    and batch == num_batches - 1
                )
                await asyncio.wait_for(
                    trace_status_updater.update_trace_progress(
                        "rec-1",
                        f"trace-{i}-batch-{batch}",
                        100,
                        wait_for_completion=last_trace,
                    ),
                    timeout=0.01,
                )
    except:
        raise

    # Assert
    assert mock_client_session.put.call_count == num_batches


@pytest.mark.asyncio
async def test_update_trace_completed_waits(
    trace_status_updater: TraceStatusUpdater, mock_client_session
):
    """Verify that (default update_trace_completed=True) waits for network response."""
    # Setup it so we can control when the request is made and completed
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.01
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 0.01
    put_called = asyncio.Event()
    put_response = asyncio.Future()

    async def enter_mock():
        put_called.set()
        return await put_response

    mock_client_session.put.return_value.__aenter__.side_effect = enter_mock

    # Act
    task = asyncio.create_task(
        trace_status_updater.update_trace_completed("rec-1", "trace-1", 500)
    )

    await asyncio.wait_for(put_called.wait(), timeout=0.1)
    await asyncio.sleep(0.1)
    assert not task.done(), "update should not be done before request is completed"
    put_response.set_result(AsyncMock(spec=aiohttp.ClientResponse, status=200))
    # Assert

    await asyncio.wait_for(task, timeout=0.1)
    assert task.result() is True
    mock_client_session.put.assert_called_once()
    json_data = mock_client_session.put.call_args.kwargs["json"]
    assert (
        json_data["updates"]["trace-1"]["status"]
        == RecordingDataTraceStatus.UPLOAD_COMPLETE
    )


@pytest.mark.asyncio
async def test_auth_refresh_on_401(
    trace_status_updater, mock_client_session, mock_auth_and_org
):
    """Verify that the updater attempts to login/refresh when receiving a 401."""
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.01
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 0.01

    # First response is 401, second is 200
    resp401 = AsyncMock(spec=aiohttp.ClientResponse, status=401)
    resp200 = AsyncMock(spec=aiohttp.ClientResponse, status=200)

    mock_client_session.put.return_value.__aenter__.side_effect = [resp401, resp200]

    # Act
    success = await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 10, wait_for_completion=True
        ),
        timeout=0.1,
    )

    # Assert
    # Check that login was called
    mock_auth_and_org["auth"].login.assert_called_once()
    # Check that put was called twice (retry logic)
    assert mock_client_session.put.call_count == 2
    assert success


@pytest.mark.asyncio
async def test_failed_request_returns_false(trace_status_updater, mock_client_session):
    """Verify that a 500 error returns False and logs warning."""
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.01
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 0.01

    response = mock_client_session.put.return_value.__aenter__.return_value
    response.status = 500
    response.text = AsyncMock(return_value="Internal Server Error")

    # Act
    success = await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 10, wait_for_completion=True
        ),
        timeout=0.1,
    )
    # Assert
    assert success is False


@pytest.mark.asyncio
async def test_error_request_returns_false(trace_status_updater, mock_client_session):
    """Verify that a 500 error returns False and logs warning."""
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.01
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 0.01

    mock_client_session.put.side_effect = aiohttp.ClientError(
        "Connection reset by peer"
    )

    # Act
    success = await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 10, wait_for_completion=True
        ),
        timeout=0.1,
    )
    # Assert
    assert success is False


@pytest.mark.asyncio
async def test_stacking_updates_on_same_trace(
    trace_status_updater, mock_client_session
):
    """Verify that multiple updates to the SAME trace_id combine to one request."""
    response = mock_client_session.put.return_value.__aenter__.return_value
    response.status = 200

    await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 100, wait_for_completion=False
        ),
        timeout=0.01,
    )
    await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 200, wait_for_completion=False
        ),
        timeout=0.01,
    )
    # Manually trigger batch processing or use a wait_for_completion call
    await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 300, wait_for_completion=True
        ),
        timeout=0.1,
    )

    json_data = mock_client_session.put.call_args.kwargs["json"]
    # The final value depends on how .stack() is implemented, but verifies the logic
    # path
    assert "trace-1" in json_data["updates"]
    assert json_data["updates"]["trace-1"]["uploaded_bytes"] == 300


@pytest.mark.asyncio
async def test_completed_update_is_sent_earlier_than_in_progress(
    trace_status_updater, mock_client_session
):
    # Make timing difference obvious
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 0.01
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 1000.0

    response = mock_client_session.put.return_value.__aenter__.return_value
    response.status = 200

    # Fire an in-progress update (should wait basically forever if alone)
    progress_update_task = asyncio.create_task(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 100, wait_for_completion=True
        )
    )

    await asyncio.sleep(0.1)
    assert isinstance(progress_update_task, asyncio.Task)
    assert (
        not progress_update_task.done()
    ), "update should wait for MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S"

    # both should finish once the complete task is included
    await asyncio.wait_for(
        trace_status_updater.update_trace_completed(
            "rec-1", "trace-2", 200, wait_for_completion=True
        ),
        timeout=0.1,
    )
    await asyncio.wait_for(progress_update_task, timeout=0.1)

    # Assert: request already sent (did NOT wait 1s)
    assert mock_client_session.put.call_count == 1


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_batch_tasks(
    trace_status_updater: TraceStatusUpdater,
) -> None:
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_COMPLETE_S = 1000.0
    TraceStatusUpdater.MINIMUM_REQUEST_INTERVAL_IN_PROGRESS_S = 1000.0

    await asyncio.wait_for(
        trace_status_updater.update_trace_progress(
            "rec-1", "trace-1", 100, wait_for_completion=False
        ),
        timeout=0.01,
    )

    assert len(trace_status_updater._batch_update_tasks) == 1
    await trace_status_updater.shutdown()

    assert trace_status_updater._pending_update_batch == {}
    assert trace_status_updater._in_progress_updates == {}
    assert trace_status_updater._batch_update_tasks == set()
