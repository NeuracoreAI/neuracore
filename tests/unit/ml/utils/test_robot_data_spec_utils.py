from pathlib import Path

import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.utils.robot_data_spec_utils import (
    convert_cross_embodiment_description_names_to_ids,
    resolve_embodiment_descriptions_with_override,
)

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a3"


def _indexed_names(*names: str) -> dict[int, str]:
    return dict(enumerate(names))


@pytest.fixture
def mock_auth_requests_robots(mock_auth_requests):
    nc.login("test_api_key")
    mocked_org_id = "test-org-id"
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/robots?is_shared=false",
        json=[{"id": TEST_ROBOT_ID, "name": "robot_name"}],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/robots?is_shared=true",
        json=[{"id": TEST_ROBOT_ID, "name": "robot_name"}],
        status_code=200,
    )


def test_convert_robot_data_spec_names_to_ids_raises_error_on_duplicates(
    mock_auth_requests_robots,
):
    spec = {
        "robot_name": {DataType.RGB_IMAGES: _indexed_names("cam", "cam2")},
        TEST_ROBOT_ID: {DataType.RGB_IMAGES: _indexed_names("cam2", "cam3")},
    }
    with pytest.raises(Exception):
        convert_cross_embodiment_description_names_to_ids(spec)


def test_convert_robot_data_spec_names_to_ids_raises_on_ambiguous_name(
    mock_auth_requests_robots,
):
    spec = {"dup_name": {DataType.RGB_IMAGES: _indexed_names("cam")}}
    with pytest.raises(Exception):
        convert_cross_embodiment_description_names_to_ids(spec)


def test_convert_robot_data_spec_names_to_ids_raises_on_name_id_collision(
    mock_auth_requests_robots,
):
    spec = {"robot_id_1": {DataType.RGB_IMAGES: _indexed_names("cam")}}
    with pytest.raises(Exception):
        convert_cross_embodiment_description_names_to_ids(spec)


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
    requests_mock, monkeypatch
) -> None:
    class _Auth:
        def get_headers(self):
            return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        "neuracore.core.utils.robot_data_spec_utils.get_auth",
        lambda: _Auth(),
    )
    monkeypatch.setattr(
        "neuracore.core.utils.robot_data_spec_utils.get_current_org",
        lambda: "test-org-id",
    )
    requests_mock.get(
        f"{API_URL}/org/test-org-id/training/jobs/job-123",
        json={
            "input_cross_embodiment_description": {
                "robot-1": {"JOINT_POSITIONS": {"0": "joint1"}}
            },
            "output_cross_embodiment_description": {
                "robot-1": {"JOINT_TARGET_POSITIONS": {"0": "joint1"}}
            },
        },
        status_code=200,
    )

    resolved_input, resolved_output = resolve_embodiment_descriptions_with_override(
        input_embodiment_description=None,
        output_embodiment_description=None,
        robot_id="robot-1",
        job_id="job-123",
    )

    assert resolved_input == {DataType.JOINT_POSITIONS: {"0": "joint1"}}
    assert resolved_output == {DataType.JOINT_TARGET_POSITIONS: {"0": "joint1"}}


def test_resolve_embodiments_with_override_loads_from_model_archive(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "neuracore.core.utils.robot_data_spec_utils.load_cross_embodiment_descriptions_from_nc_archive",
        lambda model_file: (
            {"robot-1": {"JOINT_POSITIONS": {"0": "joint1"}}},
            {"robot-1": {"JOINT_TARGET_POSITIONS": {"0": "joint1"}}},
        ),
    )

    resolved_input, resolved_output = resolve_embodiment_descriptions_with_override(
        input_embodiment_description=None,
        output_embodiment_description=None,
        robot_id="robot-1",
        model_file=Path("dummy.nc.zip"),
    )

    assert resolved_input == {DataType.JOINT_POSITIONS: {"0": "joint1"}}
    assert resolved_output == {DataType.JOINT_TARGET_POSITIONS: {"0": "joint1"}}


def test_resolve_embodiments_with_override_raises_when_incomplete() -> None:
    with pytest.raises(
        ValueError, match="Must provide both input_embodiment_description"
    ):
        resolve_embodiment_descriptions_with_override(
            input_embodiment_description={
                DataType.JOINT_POSITIONS: _indexed_names("j1")
            },
            output_embodiment_description=None,
            robot_id=None,
        )
