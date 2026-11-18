from __future__ import annotations

import pytest

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.exceptions import ValidationError
from neuracore.core.robot import Robot

ORG_ID = "org-1"
ROBOT_NAME = "robot"


def _write_urdf(tmp_path, link_geometries: str) -> str:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        f'<robot name="r"><link name="base">{link_geometries}</link></robot>'
    )
    return str(urdf_path)


def _visual(filename: str) -> str:
    return f'<visual><geometry><mesh filename="{filename}"/></geometry></visual>'


def _collision(filename: str) -> str:
    return f'<collision><geometry><mesh filename="{filename}"/></geometry></collision>'


def _make_robot(urdf_path: str) -> Robot:
    return Robot(ROBOT_NAME, instance=0, urdf_path=urdf_path, org_id=ORG_ID)


class TestRobotMeshFormats:
    """Tests for rejecting URDF visual meshes the viewer cannot draw."""

    def test_obj_visual_mesh_is_rejected(self, tmp_path):
        """An OBJ visual mesh raises with its filename in the message."""
        urdf_path = _write_urdf(tmp_path, _visual("meshes/base.obj"))

        with pytest.raises(ValidationError) as error:
            _make_robot(urdf_path)

        assert "meshes/base.obj" in str(error.value)

    @pytest.mark.parametrize(
        "filename",
        ["meshes/base.glb", "meshes/base.OBJ", "C:\\robot\\meshes\\base.obj"],
    )
    def test_unsupported_extension_is_rejected(self, tmp_path, filename):
        """An unsupported extension raises and the message lists supported ones."""
        urdf_path = _write_urdf(tmp_path, _visual(filename))

        with pytest.raises(ValidationError) as error:
            _make_robot(urdf_path)

        assert filename in str(error.value)
        assert ".stl, .dae" in str(error.value)

    @pytest.mark.parametrize(
        "filename",
        [
            "meshes/base.STL",
            "meshes/base.dae",
            "/abs/meshes/base.stl",
            "package://robot_description/meshes/base.stl",
        ],
    )
    def test_supported_extension_is_accepted(self, tmp_path, filename):
        """STL and DAE visual meshes in any path form do not raise."""
        _make_robot(_write_urdf(tmp_path, _visual(filename)))

    def test_obj_visuals_with_stl_collisions_are_rejected(self, tmp_path):
        """Each unsupported visual mesh is listed once, in document order."""
        urdf_path = _write_urdf(
            tmp_path,
            _visual("meshes/base.obj")
            + _collision("meshes/base.stl")
            + _visual("meshes/arm.obj")
            + _visual("meshes/base.obj"),
        )

        with pytest.raises(ValidationError) as error:
            _make_robot(urdf_path)

        message = str(error.value)
        assert "meshes/base.obj, meshes/arm.obj\n" in message
        assert "meshes/base.stl" not in message

    def test_obj_collision_with_stl_visual_is_accepted(self, tmp_path):
        """Collision meshes are not checked because the viewer does not load them."""
        _make_robot(
            _write_urdf(
                tmp_path,
                _visual("meshes/base.stl") + _collision("meshes/base.obj"),
            )
        )

    def test_urdf_without_meshes_is_accepted(self, tmp_path):
        """A URDF with only primitive geometry does not raise."""
        _make_robot(
            _write_urdf(
                tmp_path, '<visual><geometry><box size="1 1 1"/></geometry></visual>'
            )
        )


@pytest.mark.usefixtures("mock_login", "reset_neuracore")
def test_connect_robot_rejects_obj_visual_mesh_before_creating_robot(
    tmp_path, mock_auth_requests, mocked_org_id
):
    """connect_robot raises before it sends a robot creation request."""
    create_robot = mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "robot-id-1", "has_urdf": False},
    )
    urdf_path = _write_urdf(tmp_path, _visual("meshes/base.obj"))

    with pytest.raises(ValidationError):
        nc.connect_robot(ROBOT_NAME, urdf_path=urdf_path)

    assert not create_robot.called
