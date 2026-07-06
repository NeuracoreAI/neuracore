# ruff: noqa: E402
import os
import pathlib
import re
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
import requests_mock

TEST_API_KEY = "test_api_key"
MOCKED_ORG_ID = "test-org-id"

os.environ["NEURACORE_API_KEY"] = TEST_API_KEY
os.environ["NEURACORE_ORG_ID"] = MOCKED_ORG_ID
os.environ["NEURACORE_API_URL"] = "http://127.0.0.1:9/api"

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core.config import config_manager
from neuracore.core.const import API_URL
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_consume_live_data_enabled_manager,
    get_provide_live_data_enabled_manager,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


@pytest.fixture
def test_api_key():
    """Return the API key used by mocked authentication."""
    return TEST_API_KEY


@pytest.fixture
def mocked_org_id():
    """Return the organization id used by mocked API endpoints."""
    return MOCKED_ORG_ID


@pytest.fixture(autouse=True, scope="session")
def isolated_config_dir(tmp_path_factory):
    """Redirect the Neuracore config directory to a per-session temp dir."""
    tmpdir = tmp_path_factory.mktemp("nc_config")
    with patch.object(config_manager, "CONFIG_DIR", tmpdir):
        yield tmpdir


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory for a single test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        with patch.object(config_manager, "CONFIG_DIR", tmpdir):
            yield tmpdir


@pytest.fixture
def mock_auth_requests() -> Generator[requests_mock.Mocker, None, None]:
    """Mock authentication and API requests."""
    get_provide_live_data_enabled_manager().disable()
    get_consume_live_data_enabled_manager().disable()

    with requests_mock.Mocker(real_http=True) as m:
        # Mock API Key Verification
        m.post(
            f"{API_URL}/auth/verify-api-key",
            json={"access_token": "test_token"},
            status_code=200,
        )

        # Mock token generation
        m.post(
            f"{API_URL}/auth/token",
            json={"access_token": "test_token"},
            status_code=200,
        )

        m.get(f"{API_URL}/auth/verify-version", status_code=200)

        # Mock robots endpoint
        m.get(f"{API_URL}/org/{MOCKED_ORG_ID}/robots", json=[], status_code=200)

        # Mock robots upload endpoint
        m.put(
            re.compile(f"{API_URL}/org/{MOCKED_ORG_ID}/robots/.*/package"),
            json={"status": "success"},
            status_code=200,
        )

        # Mock dataset endpoint
        m.get(f"{API_URL}/org/{MOCKED_ORG_ID}/datasets", json=[], status_code=200)
        m.get(
            f"{API_URL}/org/{MOCKED_ORG_ID}/datasets/shared", json=[], status_code=200
        )

        # Mock models/endpoints endpoint
        m.get(
            f"{API_URL}/org/{MOCKED_ORG_ID}/models/endpoints", json=[], status_code=200
        )

        # Mock List Organizations
        m.get(
            f"{API_URL}/org-management/my-orgs",
            json=[{"org": {"id": MOCKED_ORG_ID, "name": "test organization"}}],
        )
        m.get(
            f"{API_URL}/org/{MOCKED_ORG_ID}/datasets/dataset_123/robot_ids",
            json=[],
            status_code=200,
        )

        yield m


@pytest.fixture
def mock_login(mock_auth_requests):
    """Log in against the mocked authentication endpoints."""
    nc.login(TEST_API_KEY)
    yield


@pytest.fixture
def reset_neuracore():
    """Reset Neuracore global state between tests."""
    original_auth = nc.core.auth._auth
    global_state = GlobalSingleton()
    original_active_robot = global_state._active_robot
    original_active_dataset_id = global_state._active_dataset_id
    original_has_validated_version = global_state._has_validated_version

    nc.api._active_robot = None
    nc.api._active_dataset_id = None
    nc.api._active_recording_id = None
    global_state._active_robot = None
    global_state._active_dataset_id = None
    global_state._has_validated_version = False

    nc.core.auth._auth = nc.core.auth.Auth()

    yield

    nc.core.auth._auth = original_auth
    global_state._active_robot = original_active_robot
    global_state._active_dataset_id = original_active_dataset_id
    global_state._has_validated_version = original_has_validated_version


@pytest.fixture
def mock_urdf(tmp_path):
    """Create a mock URDF file for testing."""
    meshes_dir = tmp_path / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    dummy_mesh_files = ["gripper.stl", "base.stl", "link1.stl", "link2.stl"]
    for mesh_file in dummy_mesh_files:
        mesh_path = meshes_dir / mesh_file
        mesh_path.write_bytes(b"Dummy mesh content")

    urdf_content = """<?xml version="1.0"?>
    <robot name="test_robot">
        <link name="base_link">
            <visual>
                <geometry>
                    <mesh filename="package://meshes/base.stl"/>
                </geometry>
            </visual>
        </link>
        <link name="gripper_link">
            <visual>
                <geometry>
                    <mesh filename="package://meshes/gripper.stl"/>
                </geometry>
            </visual>
        </link>
        <link name="link1">
            <visual>
                <geometry>
                    <mesh filename="package://meshes/link1.stl"/>
                </geometry>
            </visual>
        </link>
        <link name="link2">
            <visual>
                <geometry>
                    <mesh filename="package://meshes/link2.stl"/>
                </geometry>
            </visual>
        </link>
    </robot>"""

    urdf_path = tmp_path / "test_robot.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def mock_model_mar(tmp_path):
    """Create a mock model.nc.zip file for testing."""
    model_path = tmp_path / "model.nc.zip"
    model_path.write_bytes(b"dummy model content")
    return str(model_path)


@pytest.fixture
def mock_session():
    """Return a MagicMock usable as a context-managed session."""
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    return session
