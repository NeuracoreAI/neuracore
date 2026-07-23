"""Tests for CloudTrainingLogger cloud sync behaviour."""

from unittest.mock import MagicMock, patch

from neuracore.ml.logging.cloud_training_logger import CloudTrainingLogger

MODULE = "neuracore.ml.logging.cloud_training_logger"


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.put.return_value = MagicMock()
    return session


def _response(status_code: int) -> MagicMock:
    """Build a response mock with the given status code."""
    response = MagicMock()
    response.status_code = status_code
    if status_code != 200:
        response.raise_for_status.side_effect = Exception("boom")
    return response


def _make_logger(session: MagicMock) -> CloudTrainingLogger:
    """Build a logger with its background sync thread stopped and joined.

    Callers must have the cloud dependencies patched. The sync thread is
    stopped so each test drives close() deterministically without the loop
    racing on the same store.
    """
    logger_obj = CloudTrainingLogger(training_id="job123", sync_interval=10)
    logger_obj._stop_sync.set()
    if logger_obj._sync_thread is not None:
        logger_obj._sync_thread.join(timeout=5)
    session.reset_mock()
    return logger_obj


def test_close_swallows_persistent_sync_error(caplog) -> None:
    session = _mock_session()

    with (
        patch(f"{MODULE}.thread_local_session", return_value=session),
        patch(f"{MODULE}.get_current_org", return_value="org123"),
        patch(f"{MODULE}.get_auth", return_value=MagicMock(get_headers=lambda: {})),
    ):
        logger_obj = _make_logger(session)
        logger_obj.log_scalar("train/loss", 1.0, 0)
        session.put.return_value = _response(503)

        with caplog.at_level("WARNING", logger=MODULE):
            logger_obj.close()  # must not raise

    assert any("Final cloud metric sync failed" in r.message for r in caplog.records)


def test_sync_requests_a_retrying_session() -> None:
    session = _mock_session()

    with (
        patch(
            f"{MODULE}.thread_local_session", return_value=session
        ) as mock_thread_local_session,
        patch(f"{MODULE}.get_current_org", return_value="org123"),
        patch(f"{MODULE}.get_auth", return_value=MagicMock(get_headers=lambda: {})),
    ):
        logger_obj = _make_logger(session)
        logger_obj.log_scalar("train/loss", 1.0, 0)
        session.put.return_value = _response(200)
        logger_obj.close()

    mock_thread_local_session.assert_called_with(retry_transient=True)
