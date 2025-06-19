import pathlib
import re
import tempfile

import pytest
import requests_mock

import neuracore
from neuracore.core.config import config_manager
from neuracore.core.const import API_URL


@pytest.fixture
def temp_config_dir(monkeypatch):
    """Fixture to create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock home directory for testing
        monkeypatch.setattr("pathlib.Path.home", lambda: pathlib.Path(tmpdir))
        config_manager.CONFIG_DIR = pathlib.Path(tmpdir) / ".neuracore"
        yield tmpdir


MOCKED_ORG_ID = "test-org-id"


@pytest.fixture
def mocked_org_id():
    return MOCKED_ORG_ID


@pytest.fixture
def mock_auth_requests():
    """Fixture to mock authentication and API requests."""
    with requests_mock.Mocker() as m:
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

        m.get(
            f"{API_URL}/auth/verify-version",
            status_code=200,
        )
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

        yield m


@pytest.fixture
def mock_urdf(tmp_path):
    """Create a mock URDF file for testing."""
    # Create meshes directory
    meshes_dir = tmp_path / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy mesh files to satisfy URDF reference
    dummy_mesh_files = ["gripper.stl", "base.stl", "link1.stl", "link2.stl"]

    for mesh_file in dummy_mesh_files:
        mesh_path = meshes_dir / mesh_file
        mesh_path.write_bytes(b"Dummy mesh content")

    # Updated URDF content with relative mesh paths
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
    """Create a mock model.mar file for testing."""
    model_path = tmp_path / "model.mar"
    model_path.write_bytes(b"dummy model content")
    return str(model_path)


@pytest.fixture
def reset_neuracore():
    """Reset Neuracore global state between tests."""
    # Store the original authentication instance
    original_auth = neuracore.core.auth._auth

    # Reset global variables in core
    neuracore.api._active_robot = None
    neuracore.api._active_dataset_id = None
    neuracore.api._active_recording_id = None

    # Reset authentication
    neuracore.core.auth._auth = neuracore.core.auth.Auth()

    yield

    # Restore the original authentication instance
    neuracore.core.auth._auth = original_auth
