from unittest.mock import MagicMock, patch

import requests

from neuracore.ml.logging.cloud_training_logger import CloudTrainingLogger

MODULE = "neuracore.ml.logging.cloud_training_logger"


def _make_logger(session: MagicMock) -> CloudTrainingLogger:
    """Build a logger with its background sync thread stopped and joined.

    Callers must have the cloud dependencies patched. The sync thread is
    stopped so each test drives ``close()`` deterministically without the loop
    racing on the same store.
    """
    logger_obj = CloudTrainingLogger(training_id="job123", sync_interval=10)
    logger_obj._stop_sync.set()
    if logger_obj._sync_thread is not None:
        logger_obj._sync_thread.join(timeout=5)
    session.reset_mock()
    return logger_obj


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.put.return_value = MagicMock()
    return session


@patch(f"{MODULE}.time.sleep")
def test_close_swallows_persistent_sync_error(mock_sleep, caplog) -> None:
    session = _mock_session()
    session.put.return_value.raise_for_status.side_effect = (
        requests.exceptions.HTTPError("503")
    )

    with patch(f"{MODULE}.thread_local_session", return_value=session), patch(
        f"{MODULE}.get_current_org", return_value="org123"
    ), patch(f"{MODULE}.get_auth", return_value=MagicMock(get_headers=lambda: {})):
        logger_obj = _make_logger(session)
        logger_obj.log_scalar("train/loss", 1.0, 0)

        with caplog.at_level("WARNING", logger=MODULE):
            logger_obj.close()  # must not raise

    assert session.put.call_count == 3
    assert any("Final cloud metric sync failed" in r.message for r in caplog.records)


@patch(f"{MODULE}.time.sleep")
def test_close_succeeds_after_transient_error(mock_sleep, caplog) -> None:
    session = _mock_session()
    session.put.return_value.raise_for_status.side_effect = [
        requests.exceptions.HTTPError("503"),
        requests.exceptions.HTTPError("503"),
        None,
    ]

    with patch(f"{MODULE}.thread_local_session", return_value=session), patch(
        f"{MODULE}.get_current_org", return_value="org123"
    ), patch(f"{MODULE}.get_auth", return_value=MagicMock(get_headers=lambda: {})):
        logger_obj = _make_logger(session)
        logger_obj.log_scalar("train/loss", 1.0, 0)

        with caplog.at_level("WARNING", logger=MODULE):
            logger_obj.close()

    assert session.put.call_count == 3
    assert logger_obj._store == {}  # store cleared on successful sync
    assert not any(
        "Final cloud metric sync failed" in r.message for r in caplog.records
    )
