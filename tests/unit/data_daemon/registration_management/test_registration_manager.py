from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from neuracore_types import DataType

from neuracore.data_daemon.registration_management import registration_manager
from neuracore.data_daemon.registration_management.registration_manager import (
    RegistrationCandidate,
    RegistrationManager,
    get_cloud_file_list,
)

MODULE = registration_manager.__name__


@pytest.fixture
def mock_auth():
    with (
        patch(f"{MODULE}.get_auth") as mock_get_auth,
        patch(f"{MODULE}.get_current_org", return_value="test-org"),
    ):
        auth_instance = MagicMock()
        auth_instance.get_headers = MagicMock(
            return_value={"Authorization": "Bearer test-token"}
        )
        auth_instance.login = MagicMock()
        mock_get_auth.return_value = auth_instance
        yield mock_get_auth


@pytest.fixture
def state_api() -> MagicMock:
    api = MagicMock()
    api.claim_traces_for_registration = AsyncMock(return_value=[])
    api.mark_traces_registered = AsyncMock(return_value=[])
    api.mark_traces_registration_failed = AsyncMock()
    api.emit_ready_for_upload = AsyncMock()
    api.mark_traces_registering = AsyncMock(return_value=[])
    return api


def _make_candidate(
    trace_id: str = "t1",
    recording_id: str = "r1",
    data_type: DataType = DataType.JOINT_POSITIONS,
    data_type_name: str = "joints",
) -> RegistrationCandidate:
    return RegistrationCandidate(
        trace_id=trace_id,
        recording_id=recording_id,
        data_type=data_type,
        data_type_name=data_type_name,
    )


def _mock_http_session(response_payload: dict) -> MagicMock:
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_payload)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_response)
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock(spec=aiohttp.ClientSession)
    session.post = MagicMock(return_value=ctx)
    return session


class TestGetCloudFileList:
    """Tests for the get_cloud_file_list helper that derives cloud paths."""

    def test_rgb_type_returns_lossy_lossless_and_trace(self) -> None:
        files = get_cloud_file_list(DataType.RGB_IMAGES, "cam_front")
        paths = {f["filepath"] for f in files}
        assert paths == {
            "RGB_IMAGES/cam_front/lossy.mp4",
            "RGB_IMAGES/cam_front/lossless.mp4",
            "RGB_IMAGES/cam_front/trace.json",
        }
        content_types = {f["content_type"] for f in files}
        assert content_types == {"video/mp4", "application/json"}

    def test_non_rgb_type_returns_only_trace(self) -> None:
        files = get_cloud_file_list(DataType.JOINT_POSITIONS, "arm")
        assert files == [
            {
                "filepath": "JOINT_POSITIONS/arm/trace.json",
                "content_type": "application/json",
            },
        ]


class TestBatchRegistration:
    """Tests for the batch registration HTTP call and outcome handling."""

    @pytest.mark.asyncio
    async def test_registration_payload_includes_cloud_files(
        self, mock_auth, state_api, emitter
    ) -> None:
        session = _mock_http_session({
            "registered_traces": [{"trace_id": "t1", "upload_session_uris": {}}],
            "failed_traces": [],
        })
        candidate = _make_candidate(data_type=DataType.RGB_IMAGES, data_type_name="cam")

        mgr = RegistrationManager(
            client_session=session,
            state_api=state_api,
            emitter=emitter,
            batch_size=10,
        )
        await mgr._register_data_trace_batch([candidate])

        payload = session.post.call_args.kwargs["json"]
        trace_entry = payload["traces"][0]
        assert trace_entry["cloud_files"] == get_cloud_file_list(
            DataType.RGB_IMAGES, "cam"
        )

    @pytest.mark.asyncio
    async def test_session_uris_forwarded_to_state_api(
        self, mock_auth, state_api, emitter
    ) -> None:
        uris = {"JOINT_POSITIONS/arm/trace.json": "https://storage/sess/1"}
        session = _mock_http_session({
            "registered_traces": [{"trace_id": "t1", "upload_session_uris": uris}],
            "failed_traces": [],
        })
        state_api.mark_traces_registered = AsyncMock(return_value=["t1"])

        mgr = RegistrationManager(
            client_session=session,
            state_api=state_api,
            emitter=emitter,
            batch_size=10,
        )
        await mgr._register_and_record_outcome([_make_candidate()])

        state_api.emit_ready_for_upload.assert_awaited_once_with(["t1"], {"t1": uris})

    @pytest.mark.asyncio
    async def test_empty_session_uris_normalised_to_none(
        self, mock_auth, state_api, emitter
    ) -> None:
        session = _mock_http_session({
            "registered_traces": [{"trace_id": "t1", "upload_session_uris": {}}],
            "failed_traces": [],
        })
        state_api.mark_traces_registered = AsyncMock(return_value=["t1"])

        mgr = RegistrationManager(
            client_session=session,
            state_api=state_api,
            emitter=emitter,
            batch_size=10,
        )
        await mgr._register_and_record_outcome([_make_candidate()])

        state_api.emit_ready_for_upload.assert_awaited_once_with(["t1"], None)

    @pytest.mark.asyncio
    async def test_registration_retries_on_500_then_succeeds(
        self,
        mock_auth,
        state_api,
    ) -> None:
        contexts: list[MagicMock] = []

        for status, body in [
            (500, "server exploded"),
            (500, "still broken"),
            (
                200,
                {
                    "registered_traces": [{
                        "trace_id": "t1",
                        "upload_session_uris": {
                            "JOINT_POSITIONS/joints/trace.json": (
                                "https://storage/sess/1"
                            ),
                        },
                    }],
                    "failed_traces": [],
                },
            ),
        ]:
            response = AsyncMock(spec=aiohttp.ClientResponse)
            response.status = status
            response.text = AsyncMock(
                return_value=body if isinstance(body, str) else ""
            )
            response.json = AsyncMock(
                return_value=body if isinstance(body, dict) else {}
            )

            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=response)
            context.__aexit__ = AsyncMock(return_value=False)
            contexts.append(context)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.post = MagicMock(side_effect=contexts)

        state_api.mark_traces_registered = AsyncMock(return_value=["t1"])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            manager = RegistrationManager(
                client_session=session,
                state_api=state_api,
                emitter=MagicMock(),
                batch_size=10,
            )
            await manager._register_and_record_outcome([_make_candidate()])

        assert session.post.call_count == 3
        state_api.mark_traces_registered.assert_awaited_once_with(["t1"])
        state_api.emit_ready_for_upload.assert_awaited_once_with(
            ["t1"],
            {
                "t1": {
                    "JOINT_POSITIONS/joints/trace.json": "https://storage/sess/1",
                },
            },
        )
        state_api.mark_traces_registration_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_registration_repeated_500_marks_traces_failed(
        self,
        mock_auth,
        state_api,
    ) -> None:
        contexts: list[MagicMock] = []

        for _ in range(registration_manager.REGISTRATION_MAX_RETRIES):
            response = AsyncMock(spec=aiohttp.ClientResponse)
            response.status = 500
            response.text = AsyncMock(return_value="backend down")
            response.json = AsyncMock(return_value={})

            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=response)
            context.__aexit__ = AsyncMock(return_value=False)
            contexts.append(context)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.post = MagicMock(side_effect=contexts)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            manager = RegistrationManager(
                client_session=session,
                state_api=state_api,
                emitter=MagicMock(),
                batch_size=10,
            )
            await manager._register_and_record_outcome([_make_candidate()])

        assert session.post.call_count == registration_manager.REGISTRATION_MAX_RETRIES
        state_api.mark_traces_registered.assert_not_awaited()
        state_api.emit_ready_for_upload.assert_not_awaited()
        state_api.mark_traces_registration_failed.assert_awaited_once()

        call = state_api.mark_traces_registration_failed.await_args
        assert call.kwargs["trace_ids"] == ["t1"]
        assert "backend down" in call.kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_registration_retries_on_timeout_then_succeeds(
        self,
        mock_auth,
        state_api,
    ) -> None:
        success_response = AsyncMock(spec=aiohttp.ClientResponse)
        success_response.status = 200
        success_response.json = AsyncMock(
            return_value={
                "registered_traces": [{
                    "trace_id": "t1",
                    "upload_session_uris": {},
                }],
                "failed_traces": [],
            }
        )

        success_context = MagicMock()
        success_context.__aenter__ = AsyncMock(return_value=success_response)
        success_context.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.post = MagicMock(
            side_effect=[
                asyncio.TimeoutError("timed out"),
                asyncio.TimeoutError("timed out again"),
                success_context,
            ]
        )

        state_api.mark_traces_registered = AsyncMock(return_value=["t1"])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            manager = RegistrationManager(
                client_session=session,
                state_api=state_api,
                emitter=MagicMock(),
                batch_size=10,
            )
            await manager._register_and_record_outcome([_make_candidate()])

        assert session.post.call_count == 3
        state_api.mark_traces_registered.assert_awaited_once_with(["t1"])
        state_api.emit_ready_for_upload.assert_awaited_once_with(["t1"], None)
        state_api.mark_traces_registration_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_registration_partial_batch_success_and_failure(
        self,
        mock_auth,
        state_api,
    ) -> None:
        session = _mock_http_session({
            "registered_traces": [{
                "trace_id": "t1",
                "upload_session_uris": {
                    "JOINT_POSITIONS/joints/trace.json": ("https://storage/sess/1"),
                },
            }],
            "failed_traces": [{
                "trace_id": "t2",
                "error": "bad trace payload",
            }],
        })

        candidates = [
            _make_candidate(trace_id="t1", recording_id="r1"),
            _make_candidate(trace_id="t2", recording_id="r2"),
        ]
        state_api.mark_traces_registered = AsyncMock(return_value=["t1"])

        manager = RegistrationManager(
            client_session=session,
            state_api=state_api,
            emitter=MagicMock(),
            batch_size=10,
        )
        await manager._register_and_record_outcome(candidates)

        state_api.mark_traces_registered.assert_awaited_once_with(["t1"])
        state_api.emit_ready_for_upload.assert_awaited_once_with(
            ["t1"],
            {
                "t1": {
                    "JOINT_POSITIONS/joints/trace.json": "https://storage/sess/1",
                },
            },
        )
        state_api.mark_traces_registration_failed.assert_awaited_once()

        call = state_api.mark_traces_registration_failed.await_args
        assert call.kwargs["trace_ids"] == ["t2"]
        assert "bad trace payload" in call.kwargs["error_message"]
