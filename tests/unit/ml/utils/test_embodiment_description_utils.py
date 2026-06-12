from pathlib import Path

import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotMismatchError
from neuracore.core.utils.embodiment_description_utils import (
    convert_cross_embodiment_description_names_to_ids,
    resolve_embodiment_descriptions,
    resolve_embodiment_descriptions_with_override,
)

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a3"
TEST_ARCHIVE_ROBOT_ID = "11111111-1111-4111-8111-111111111111"


def _indexed_names(*names: str) -> dict[int, str]:
    return dict(enumerate(names))


def _robots_list_requests(mock_auth_requests, mocked_org_id: str):
    list_url = f"{API_URL}/org/{mocked_org_id}/robots/list"
    return [r for r in mock_auth_requests.request_history if r.url.startswith(list_url)]


def _robots_list_page(
    robots: list[dict],
    *,
    total: int | None = None,
) -> dict:
    return {
        "data": robots,
        "metadata": None,
        "total": total if total is not None else len(robots),
        "limit": 30,
        "start_after": None,
    }


@pytest.fixture
def mock_current_org(monkeypatch, mocked_org_id):
    monkeypatch.setattr(
        "neuracore.core.robot.get_current_org",
        lambda: mocked_org_id,
    )


@pytest.fixture
def mock_auth_requests_robots(
    mock_auth_requests,
    temp_config_dir,
    reset_neuracore,
    mocked_org_id,
    mock_current_org,
):
    nc.login("test_api_key")
    robot = {"id": TEST_ROBOT_ID, "name": "robot_name"}
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots/list",
        json=_robots_list_page([robot]),
        status_code=200,
    )


@pytest.fixture
def mock_auth_requests_robots_paginated(
    mock_auth_requests,
    temp_config_dir,
    reset_neuracore,
    mocked_org_id,
    mock_current_org,
):
    nc.login("test_api_key")
    other_robot = {
        "id": "11111111-1111-4111-8111-111111111111",
        "name": "other_robot",
    }
    target_robot = {"id": TEST_ROBOT_ID, "name": "robot_name"}
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots/list",
        [
            {
                "json": _robots_list_page([other_robot], total=2),
                "status_code": 200,
            },
            {
                "json": _robots_list_page([target_robot], total=2),
                "status_code": 200,
            },
            {
                "json": _robots_list_page([]),
                "status_code": 200,
            },
        ],
    )


def test_convert_cross_embodiment_description_names_to_ids_resolves_name_across_pages(
    mock_auth_requests,
    mock_auth_requests_robots_paginated,
    mocked_org_id,
) -> None:
    spec = {"robot_name": {DataType.RGB_IMAGES: _indexed_names("cam")}}
    result = convert_cross_embodiment_description_names_to_ids(spec)

    assert result == {TEST_ROBOT_ID: {DataType.RGB_IMAGES: _indexed_names("cam")}}

    list_requests = _robots_list_requests(mock_auth_requests, mocked_org_id)
    assert len(list_requests) == 3


def test_convert_cross_embodiment_description_names_to_ids_raises_error_on_duplicates(
    mock_auth_requests_robots,
):
    spec = {
        "robot_name": {DataType.RGB_IMAGES: _indexed_names("cam", "cam2")},
        TEST_ROBOT_ID: {DataType.RGB_IMAGES: _indexed_names("cam2", "cam3")},
    }
    with pytest.raises(Exception):
        convert_cross_embodiment_description_names_to_ids(spec)


def test_convert_cross_embodiment_description_names_to_ids_raises_on_ambiguous_name(
    mock_auth_requests_robots,
):
    spec = {"dup_name": {DataType.RGB_IMAGES: _indexed_names("cam")}}
    with pytest.raises(Exception):
        convert_cross_embodiment_description_names_to_ids(spec)


def test_convert_cross_embodiment_description_names_to_ids_raises_on_name_id_collision(
    mock_auth_requests_robots,
):
    spec = {"robot_id_1": {DataType.RGB_IMAGES: _indexed_names("cam")}}
    with pytest.raises(Exception):
        convert_cross_embodiment_description_names_to_ids(spec)


def test_resolve_embodiment_descriptions_raises_mismatch_with_robot_names(
    requests_mock, monkeypatch: pytest.MonkeyPatch, mocked_org_id: str
) -> None:
    class _Auth:
        is_authenticated = True

        def get_headers(self):
            return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.robot.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_current_org",
        lambda: mocked_org_id,
    )
    monkeypatch.setattr(
        "neuracore.core.robot.get_current_org",
        lambda: mocked_org_id,
    )
    trained_robot_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    requested_robot_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    requests_mock.post(
        f"{API_URL}/org/{mocked_org_id}/robots/names",
        json={
            "robots": [
                {"id": trained_robot_id, "name": "lab-franka"},
                {"id": requested_robot_id, "name": "lab-franka-2"},
            ],
            "missing_robot_ids": [],
        },
        status_code=200,
    )
    input_cross = {trained_robot_id: {DataType.JOINT_POSITIONS: {0: "joint1"}}}
    output_cross = {trained_robot_id: {DataType.JOINT_TARGET_POSITIONS: {0: "joint1"}}}

    with pytest.raises(RobotMismatchError) as exc_info:
        resolve_embodiment_descriptions(
            input_cross_embodiment_description=input_cross,
            output_cross_embodiment_description=output_cross,
            robot_id=requested_robot_id,
        )

    message = str(exc_info.value)
    assert "This model cannot run on the requested robot record." in message
    assert 'Requested robot name: "lab-franka-2"' in message
    assert f"Requested robot ID: {requested_robot_id}" in message
    assert f"Current organization: {mocked_org_id}" in message
    assert '"lab-franka"' in message
    assert "globally unique robot record ID" in message
    assert "same hardware in another organization" in message


def test_resolve_embodiment_descriptions_raises_mismatch_with_partial_name_resolution(
    requests_mock, monkeypatch: pytest.MonkeyPatch, mocked_org_id: str
) -> None:
    class _Auth:
        is_authenticated = True

        def get_headers(self):
            return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.robot.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_current_org",
        lambda: mocked_org_id,
    )
    monkeypatch.setattr(
        "neuracore.core.robot.get_current_org",
        lambda: mocked_org_id,
    )
    trained_robot_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    requested_robot_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    requests_mock.post(
        f"{API_URL}/org/{mocked_org_id}/robots/names",
        json={
            "robots": [
                {"id": requested_robot_id, "name": "org-b-franka"},
            ],
            "missing_robot_ids": [trained_robot_id],
        },
        status_code=200,
    )
    input_cross = {trained_robot_id: {DataType.JOINT_POSITIONS: {0: "joint1"}}}
    output_cross = {trained_robot_id: {DataType.JOINT_TARGET_POSITIONS: {0: "joint1"}}}

    with pytest.raises(RobotMismatchError) as exc_info:
        resolve_embodiment_descriptions(
            input_cross_embodiment_description=input_cross,
            output_cross_embodiment_description=output_cross,
            robot_id=requested_robot_id,
        )

    message = str(exc_info.value)
    assert 'Requested robot name: "org-b-franka"' in message
    assert f"Requested robot ID: {requested_robot_id}" in message
    assert f"Current organization: {mocked_org_id}" in message
    assert trained_robot_id in message
    assert "model.nc.zip file is downloaded from one organization" in message
    assert "robot name may resolve successfully" in message


def test_resolve_embodiments_with_override_returns_explicit_descriptions() -> None:
    input_emb = {DataType.JOINT_POSITIONS: _indexed_names("joint1")}
    output_emb = {DataType.JOINT_TARGET_POSITIONS: _indexed_names("joint1")}

    resolved_input, resolved_output = resolve_embodiment_descriptions_with_override(
        input_embodiment_description=input_emb,
        output_embodiment_description=output_emb,
        robot_id=None,
    )

    assert resolved_input == input_emb
    assert resolved_output == output_emb


def test_resolve_embodiments_with_override_loads_from_job_metadata(
    requests_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Auth:
        is_authenticated = True

        def get_headers(self):
            return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_current_org",
        lambda: "test-org-id",
    )
    requests_mock.get(
        f"{API_URL}/org/test-org-id/training/jobs/job-123",
        json={
            "input_cross_embodiment_description": {
                TEST_ARCHIVE_ROBOT_ID: {"JOINT_POSITIONS": {"0": "joint1"}}
            },
            "output_cross_embodiment_description": {
                TEST_ARCHIVE_ROBOT_ID: {"JOINT_TARGET_POSITIONS": {"0": "joint1"}}
            },
        },
        status_code=200,
    )

    resolved_input, resolved_output = resolve_embodiment_descriptions_with_override(
        input_embodiment_description=None,
        output_embodiment_description=None,
        robot_id=TEST_ARCHIVE_ROBOT_ID,
        job_id="job-123",
    )

    assert resolved_input == {DataType.JOINT_POSITIONS: {"0": "joint1"}}
    assert resolved_output == {DataType.JOINT_TARGET_POSITIONS: {"0": "joint1"}}


def test_resolve_embodiments_with_override_loads_from_model_archive(
    monkeypatch,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        "neuracore.ml.utils.nc_archive."
        "load_cross_embodiment_descriptions_from_nc_archive",
        lambda model_file: (
            {TEST_ARCHIVE_ROBOT_ID: {"JOINT_POSITIONS": {"0": "joint1"}}},
            {TEST_ARCHIVE_ROBOT_ID: {"JOINT_TARGET_POSITIONS": {"0": "joint1"}}},
        ),
    )

    resolved_input, resolved_output = resolve_embodiment_descriptions_with_override(
        input_embodiment_description=None,
        output_embodiment_description=None,
        robot_id=TEST_ARCHIVE_ROBOT_ID,
        model_file=Path("dummy.nc.zip"),
    )

    assert resolved_input == {DataType.JOINT_POSITIONS: {"0": "joint1"}}
    assert resolved_output == {DataType.JOINT_TARGET_POSITIONS: {"0": "joint1"}}


def test_resolve_embodiments_with_override_raises_when_training_job_not_found(
    requests_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job metadata fetch returns 404, so cross-embodiment specs are never loaded."""

    class _Auth:
        is_authenticated = True

        def get_headers(self):
            return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_current_org",
        lambda: "test-org-id",
    )
    requests_mock.get(
        f"{API_URL}/org/test-org-id/training/jobs/job-123",
        status_code=404,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Failed to load input_cross_embodiment_description and "
            "output_cross_embodiment_description"
        ),
    ):
        resolve_embodiment_descriptions_with_override(
            robot_id=TEST_ARCHIVE_ROBOT_ID,
            job_id="job-123",
        )


def test_resolve_embodiments_override_raises_when_job_lacks_cross_embodiment(
    requests_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Auth:
        is_authenticated = True

        def get_headers(self):
            return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.utils.embodiment_description_utils.get_current_org",
        lambda: "test-org-id",
    )
    requests_mock.get(
        f"{API_URL}/org/test-org-id/training/jobs/job-123",
        json={"id": "job-123", "name": "run"},
        status_code=200,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Failed to load input_cross_embodiment_description and "
            "output_cross_embodiment_description"
        ),
    ):
        resolve_embodiment_descriptions_with_override(
            robot_id=TEST_ARCHIVE_ROBOT_ID,
            job_id="job-123",
        )


def test_resolve_embodiments_override_raises_when_archive_lacks_cross_embodiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        "neuracore.ml.utils.nc_archive."
        "load_cross_embodiment_descriptions_from_nc_archive",
        lambda model_file: (None, None),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Failed to load input_cross_embodiment_description and "
            "output_cross_embodiment_description"
        ),
    ):
        resolve_embodiment_descriptions_with_override(
            robot_id=TEST_ARCHIVE_ROBOT_ID,
            model_file=Path("dummy.nc.zip"),
        )


def test_resolve_embodiments_with_override_raises_when_nothing_provided() -> None:
    with pytest.raises(ValueError, match="Failed to resolve embodiment descriptions"):
        resolve_embodiment_descriptions_with_override()


def test_resolve_embodiments_with_override_raises_when_only_output_provided() -> None:
    with pytest.raises(ValueError, match="Failed to resolve embodiment descriptions"):
        resolve_embodiment_descriptions_with_override(
            output_embodiment_description={
                DataType.JOINT_TARGET_POSITIONS: _indexed_names("joint1")
            },
        )


def test_resolve_embodiments_with_override_raises_when_incomplete() -> None:
    with pytest.raises(ValueError, match="Failed to resolve embodiment descriptions"):
        resolve_embodiment_descriptions_with_override(
            input_embodiment_description={
                DataType.JOINT_POSITIONS: _indexed_names("j1")
            },
            output_embodiment_description=None,
            robot_id=None,
        )
