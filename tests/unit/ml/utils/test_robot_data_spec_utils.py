import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.utils.robot_data_spec_utils import (
    convert_robot_data_spec_names_to_ids,
)

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a3"


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
        "robot_name": {DataType.RGB_IMAGES: ["cam", "cam2"]},
        TEST_ROBOT_ID: {DataType.RGB_IMAGES: ["cam2", "cam3"]},
    }
    with pytest.raises(Exception):
        convert_robot_data_spec_names_to_ids(spec)


def test_convert_robot_data_spec_names_to_ids_raises_on_ambiguous_name(
    mock_auth_requests_robots,
):
    spec = {"dup_name": {DataType.RGB_IMAGES: ["cam"]}}
    with pytest.raises(Exception):
        convert_robot_data_spec_names_to_ids(spec)


def test_convert_robot_data_spec_names_to_ids_raises_on_name_id_collision(
    mock_auth_requests_robots,
):
    spec = {"robot_id_1": {DataType.RGB_IMAGES: ["cam"]}}
    with pytest.raises(Exception):
        convert_robot_data_spec_names_to_ids(spec)
