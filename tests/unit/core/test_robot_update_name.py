import pytest

from neuracore.core import robot as robot_module
from neuracore.core.exceptions import AuthenticationError


def test_update_robot_name_raises_authentication_error_when_not_authenticated(
    monkeypatch,
):
    auth = robot_module.Auth()
    monkeypatch.setattr(auth, "_access_token", None)
    monkeypatch.setattr(robot_module, "get_auth", lambda: auth)

    with pytest.raises(AuthenticationError, match="Not authenticated"):
        robot_module.update_robot_name("old-name", "new-name")
