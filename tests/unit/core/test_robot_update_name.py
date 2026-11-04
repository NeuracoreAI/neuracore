import pytest

import neuracore as nc
from neuracore.core import robot as robot_module
from neuracore.core.const import API_URL
from neuracore.core.exceptions import AuthenticationError, RobotError

ROBOT_ID = "robot-id-1"
NEW_NAME = "new-name"


def test_update_robot_name_raises_authentication_error_when_not_authenticated(
    monkeypatch,
):
    auth = robot_module.Auth()
    monkeypatch.setattr(auth, "_access_token", None)
    monkeypatch.setattr(robot_module, "get_auth", lambda: auth)

    with pytest.raises(AuthenticationError, match="Not authenticated"):
        robot_module.update_robot_name("old-name", "new-name")


@pytest.fixture
def logged_in(temp_config_dir, mock_auth_requests, reset_neuracore):
    nc.login("test_api_key")


def _rename_url(org_id: str) -> str:
    return f"{API_URL}/org/{org_id}/robots/{ROBOT_ID}/name"


def _connect_url(org_id: str) -> str:
    return f"{API_URL}/org/{org_id}/robots"


@pytest.mark.usefixtures("logged_in")
@pytest.mark.parametrize(("shared", "is_shared"), [(False, "false"), (True, "true")])
def test_rename_puts_to_the_name_route_and_never_posts_to_connect(
    mock_auth_requests, mocked_org_id, shared, is_shared
):
    rename = mock_auth_requests.put(_rename_url(mocked_org_id), status_code=200)
    connect = mock_auth_requests.post(
        _connect_url(mocked_org_id),
        status_code=409,
        json={"detail": {"error": "archived", "error_type": "ROBOT_ARCHIVED"}},
    )

    assert robot_module.update_robot_name(ROBOT_ID, NEW_NAME, shared=shared) == ROBOT_ID

    assert rename.last_request.json() == {"name": NEW_NAME}
    assert rename.last_request.qs == {"is_shared": [is_shared]}
    assert not connect.called


@pytest.mark.usefixtures("logged_in")
@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (404, "Robot not found"),
        (409, "Another robot in this organization already has this name."),
    ],
)
def test_rename_route_errors_are_reported_with_the_server_message(
    mock_auth_requests, mocked_org_id, status_code, message
):
    mock_auth_requests.put(
        _rename_url(mocked_org_id),
        status_code=status_code,
        json={"detail": {"error": message, "status": status_code}},
    )

    with pytest.raises(RobotError) as raised:
        robot_module.update_robot_name(ROBOT_ID, NEW_NAME)

    assert str(raised.value) == f"Failed to update robot name: {message}"
