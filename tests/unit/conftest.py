# ruff: noqa: E402
import os
import pathlib
import re
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import imagecodecs
import numpy as np
import pytest
import requests_mock

TEST_API_KEY = "test_api_key"
MOCKED_ORG_ID = "test-org-id"

os.environ["NEURACORE_API_KEY"] = TEST_API_KEY
os.environ["NEURACORE_ORG_ID"] = MOCKED_ORG_ID
os.environ["NEURACORE_API_URL"] = "http://127.0.0.1:9/api"

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.api.globals import GlobalSingleton
from neuracore.core.config import config_manager
from neuracore.core.const import API_URL
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_consume_live_data_enabled_manager,
    get_provide_live_data_enabled_manager,
)
from neuracore.data_daemon import bridge


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


@pytest.fixture(autouse=True)
def stub_data_bridge(monkeypatch):
    """Stub the data daemon's native extension for every unit test.

    Tests import ``neuracore`` from the source tree, which holds no compiled
    ``_data_bridge``; only the installed wheel does. Tests that assert on the
    daemon patch ``_load_native`` themselves.
    """
    monkeypatch.setattr(bridge, "_load_native", lambda: MagicMock())
    monkeypatch.setattr(api_core, "ensure_daemon_running", lambda: 0)


class FakeRegistry:
    """Plays the data daemon's robot and dataset registry in unit tests.

    Every row is confirmed at once with a server id: by default
    ``cloud-dataset-<name>`` / ``cloud-robot-<name>``, or the one set in
    :attr:`dataset_ids` / :attr:`robot_ids`. A test sets a row's state by name
    to hold or fail it instead.
    """

    def __init__(self) -> None:
        self.dataset_ids: dict[str, str] = {}
        self.robot_ids: dict[str, str] = {}
        self.dataset_specs: list[dict] = []
        self.robot_specs: list[dict] = []
        self._dataset_rows: dict[tuple, int] = {}
        self._robot_rows: dict[tuple, int] = {}
        self._dataset_states: dict[str, bridge.RowState] = {}
        self._robot_states: dict[str, bridge.RowState] = {}
        self._names: dict[int, str] = {}

    def set_dataset_state(
        self, name: str, status: str, reason: str | None = None
    ) -> None:
        self._dataset_states[name] = bridge.RowState(status, None, reason)

    def set_robot_state(
        self, name: str, status: str, reason: str | None = None
    ) -> None:
        self._robot_states[name] = bridge.RowState(status, None, reason)

    def select_dataset_row(self, *, org_id, name, shared, description, tags):
        self.dataset_specs.append(
            dict(
                org_id=org_id,
                name=name,
                shared=shared,
                description=description,
                tags=list(tags),
            )
        )
        key = (org_id, name, shared)
        index = self._dataset_rows.setdefault(key, 100 + len(self._dataset_rows))
        self._names[index] = name
        return index

    def dataset_row_for_cloud_id(self, *, org_id, cloud_dataset_id):
        key = ("cloud-id", org_id, cloud_dataset_id)
        index = self._dataset_rows.setdefault(key, 100 + len(self._dataset_rows))
        self._names[index] = cloud_dataset_id
        self.dataset_ids[cloud_dataset_id] = cloud_dataset_id
        return index

    def dataset_row_state(self, dataset_index):
        name = self._names.get(dataset_index)
        if name is None:
            return None
        if name in self._dataset_states:
            return self._dataset_states[name]
        return bridge.RowState(
            "done", self.dataset_ids.get(name, f"cloud-dataset-{name}"), None
        )

    def connect_robot_row(
        self, *, org_id, name, shared, instance, overwrite_urdf, package_path
    ):
        self.robot_specs.append(
            dict(
                org_id=org_id,
                name=name,
                shared=shared,
                instance=instance,
                overwrite_urdf=overwrite_urdf,
                package_path=package_path,
            )
        )
        key = (org_id, name, shared, instance)
        index = self._robot_rows.setdefault(key, 200 + len(self._robot_rows))
        self._names[index] = name
        return index

    def robot_row_state(self, robot_index):
        name = self._names.get(robot_index)
        if name is None:
            return None
        if name in self._robot_states:
            return self._robot_states[name]
        return bridge.RowState(
            "done", self.robot_ids.get(name, f"cloud-robot-{name}"), None
        )


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch) -> FakeRegistry:
    """Stand in for the daemon's robot and dataset registry in every unit test."""
    registry = FakeRegistry()
    for name in (
        "select_dataset_row",
        "dataset_row_for_cloud_id",
        "dataset_row_state",
        "connect_robot_row",
        "robot_row_state",
    ):
        monkeypatch.setattr(bridge, name, getattr(registry, name))
    return registry


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
    original_has_validated_version = global_state._has_validated_version
    original_robots = dict(nc.core.robot._robots)
    original_robot_name_id_mapping = dict(nc.core.robot._robot_name_id_mapping)
    original_connected_robots = dict(nc.core.robot._connected_robots)
    original_active_dataset = global_state._active_dataset

    nc.api._active_robot = None
    nc.api._active_recording_id = None
    global_state._active_robot = None
    global_state._has_validated_version = False
    global_state._active_dataset = None
    nc.core.robot._robots.clear()
    nc.core.robot._robot_name_id_mapping.clear()
    nc.core.robot._connected_robots.clear()

    nc.core.auth._auth = nc.core.auth.Auth()

    yield

    nc.core.auth._auth = original_auth
    global_state._active_robot = original_active_robot
    global_state._has_validated_version = original_has_validated_version
    nc.core.robot._robots.clear()
    nc.core.robot._robots.update(original_robots)
    nc.core.robot._robot_name_id_mapping.clear()
    nc.core.robot._robot_name_id_mapping.update(original_robot_name_id_mapping)
    nc.core.robot._connected_robots.clear()
    nc.core.robot._connected_robots.update(original_connected_robots)
    global_state._active_dataset = original_active_dataset


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


@pytest.fixture
def depth_frames_fn():
    """Return a builder of depth frames files holding JPEG-XL frames."""

    def _depth_frames(frames: np.ndarray) -> tuple[bytes, list[tuple[int, int]]]:
        payloads = [imagecodecs.jpegxl_encode(frame, lossless=True) for frame in frames]
        ranges, offset = [], 0
        for payload in payloads:
            ranges.append((offset, len(payload)))
            offset += len(payload)
        return b"".join(payloads), ranges

    return _depth_frames
