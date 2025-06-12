"""MJCF to URDF conversion utility for robot model formats.

This module provides functionality to convert MuJoCo XML format (MJCF) robot
models to Universal Robot Description Format (URDF). It handles the conversion
of kinematic structures, inertial properties, joint configurations, and mesh
geometries while creating appropriate STL files for visual elements.
"""

from pathlib import Path
from typing import List, Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from stl import mesh


def _array2str(arr: list) -> str:
    """Convert numeric array to space-separated string for XML attributes.

    Args:
        arr: List or array of numeric values.

    Returns:
        Space-separated string representation of the array values.
    """
    return " ".join([str(x) for x in arr])


def _create_body(
    xml_root: ET.Element,
    name: str,
    inertial_pos: list,
    inertial_rpy: list,
    mass: float,
    ixx: float,
    iyy: float,
    izz: float,
) -> ET.Element:
    """Create a URDF link element with specified mass and inertia properties.

    Args:
        xml_root: Parent XML element to attach the new link to.
        name: Name identifier for the link.
        inertial_pos: Position offset of the center of mass [x, y, z].
        inertial_rpy: Orientation of the inertial frame in roll-pitch-yaw [r, p, y].
        mass: Mass of the link in kilograms.
        ixx: Moment of inertia about the x-axis.
        iyy: Moment of inertia about the y-axis.
        izz: Moment of inertia about the z-axis.

    Returns:
        The created link XML element.
    """
    # create XML element for this body
    body = ET.SubElement(xml_root, "link", {"name": name})

    # add inertial element
    inertial = ET.SubElement(body, "inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": _array2str(inertial_pos), "rpy": _array2str(inertial_rpy)},
    )
    ET.SubElement(inertial, "mass", {"value": str(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(ixx),
            "iyy": str(iyy),
            "izz": str(izz),
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )
    return body


def _create_dummy_body(xml_root: ET.Element, name: str) -> ET.Element:
    """Create a lightweight dummy link with negligible mass and inertia.

    Used for creating intermediate joint bodies in the URDF kinematic chain
    where MJCF and URDF joint representations differ.

    Args:
        xml_root: Parent XML element to attach the new link to.
        name: Name identifier for the dummy link.

    Returns:
        The created dummy link XML element.
    """
    mass = 0.001
    mass_moi = mass * (0.001**2)  # mass moment of inertia
    return _create_body(
        xml_root, name, np.zeros(3), np.zeros(3), mass, mass_moi, mass_moi, mass_moi
    )


def _create_joint(
    xml_root: ET.Element,
    name: str,
    parent: str,
    child: str,
    pos: list,
    rpy: list,
    axis: Optional[List] = None,
    jnt_range: Optional[List] = None,
    jnt_type: str = "fixed",
) -> ET.Element:
    """Create a URDF joint element connecting two links.

    Creates joints of various types including fixed, revolute, and prismatic.
    Joint parameters are automatically configured based on the provided axis
    and range information.

    Args:
        xml_root: Parent XML element to attach the new joint to.
        name: Name identifier for the joint.
        parent: Name of the parent link.
        child: Name of the child link.
        pos: Position offset from parent to child [x, y, z].
        rpy: Orientation offset in roll-pitch-yaw [r, p, y].
        axis: Joint axis of rotation/translation [x, y, z]. None for fixed joints.
        jnt_range: Joint limits [min, max]. None for fixed joints.
        jnt_type: Type of joint ("fixed", "revolute", or "prismatic").

    Returns:
        The created joint XML element.
    """
    if axis is None:
        assert jnt_range is None
        joint_type = "fixed"
    else:
        joint_type = jnt_type

    # create joint element connecting this to parent
    jnt_element = ET.SubElement(xml_root, "joint", {"type": joint_type, "name": name})
    ET.SubElement(jnt_element, "parent", {"link": parent})
    ET.SubElement(jnt_element, "child", {"link": child})
    ET.SubElement(
        jnt_element, "origin", {"xyz": _array2str(pos), "rpy": _array2str(rpy)}
    )
    if axis is not None:
        ET.SubElement(jnt_element, "axis", {"xyz": _array2str(axis)})
        if jnt_range is not None:
            ET.SubElement(
                jnt_element,
                "limit",
                {
                    "lower": str(jnt_range[0]),
                    "upper": str(jnt_range[1]),
                    "effort": "100",
                    "velocity": "100",
                },
            )
    return jnt_element


def convert(mjcf_file: str, urdf_file: Path, asset_file_prefix: str = "") -> None:
    """Convert a MuJoCo MJCF file to URDF format.

    Performs a comprehensive conversion from MJCF to URDF including:
    - Kinematic structure and joint relationships
    - Mass and inertia properties
    - Joint types and limits (revolute, prismatic, fixed)
    - Visual mesh geometries converted to STL format
    - Proper coordinate frame transformations

    The conversion handles the structural differences between MJCF and URDF
    by creating intermediate joint bodies when necessary to maintain equivalent
    kinematic behavior.

    Args:
        mjcf_file: Path to the input MJCF file to convert.
        urdf_file: Path where the output URDF file will be saved.
        asset_file_prefix: Optional prefix for generated STL mesh files.
            Useful for organizing assets in subdirectories.

    Raises:
        AssertionError: If a body has more than one joint, which is not supported.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    root = ET.Element("robot", {"name": "converted_robot"})

    for id in range(model.nbody):
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)
        parent_id = model.body_parentid[id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        parentbody2childbody_pos = model.body_pos[id]
        parentbody2childbody_quat = model.body_quat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        parentbody2childbody_quat = [
            parentbody2childbody_quat[1],
            parentbody2childbody_quat[2],
            parentbody2childbody_quat[3],
            parentbody2childbody_quat[0],
        ]
        parentbody2childbody_Rot = Rotation.from_quat(
            parentbody2childbody_quat
        ).as_matrix()
        parentbody2childbody_rpy = Rotation.from_matrix(
            parentbody2childbody_Rot
        ).as_euler("xyz")

        # read inertial info
        mass = model.body_mass[id]
        inertia = model.body_inertia[id]
        childbody2childinertia_pos = model.body_ipos[id]
        childbody2childinertia_quat = model.body_iquat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        childbody2childinertia_quat = [
            childbody2childinertia_quat[1],
            childbody2childinertia_quat[2],
            childbody2childinertia_quat[3],
            childbody2childinertia_quat[0],
        ]
        childbody2childinertia_Rot = Rotation.from_quat(
            childbody2childinertia_quat
        ).as_matrix()
        childbody2childinertia_rpy = Rotation.from_matrix(
            childbody2childinertia_Rot
        ).as_euler("xyz")

        jntnum = model.body_jntnum[id]
        assert jntnum <= 1, "only one joint per body supported"

        if jntnum == 1:
            # load joint info
            jntid = model.body_jntadr[id]
            jnt_type = model.jnt_type[jntid]

            # Handle different joint types
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                urdf_jnt_type = "revolute"
            elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                urdf_jnt_type = "prismatic"
            else:
                # only support hinge and slide joints
                continue

            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jntid)
            jnt_range = model.jnt_range[jntid]  # [min, max]
            jnt_axis_childbody = model.jnt_axis[jntid]  # [x, y, z]
            childbody2jnt_pos = model.jnt_pos[jntid]  # [x, y, z]
            parentbody2jnt_axis = jnt_axis_childbody
        else:
            # create a fixed joint instead
            jnt_name = f"{parent_name}2{child_name}_fixed"
            urdf_jnt_type = "fixed"
            jnt_range = None
            childbody2jnt_pos = np.zeros(3)
            parentbody2jnt_axis = None

        # create child body
        body_element = _create_body(
            root,
            child_name,
            childbody2childinertia_pos,
            childbody2childinertia_rpy,
            mass,
            inertia[0],
            inertia[1],
            inertia[2],
        )

        # read geom info and add it child body
        geomnum = model.body_geomnum[id]
        for geomnum_i in range(geomnum):
            geomid = model.body_geomadr[id] + geomnum_i
            if model.geom_type[geomid] != mujoco.mjtGeom.mjGEOM_MESH:
                # only support mesh geoms
                continue
            geom_dataid = model.geom_dataid[geomid]  # id of geom's mesh
            geom_pos = model.geom_pos[geomid]
            geom_quat = model.geom_quat[geomid]  # [w, x, y, z]
            # change to [x, y, z, w]
            geom_quat = [geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]
            geom_rpy = Rotation.from_quat(geom_quat).as_euler("xyz")
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, geom_dataid)

            # create visual element within body element
            visual_element = ET.SubElement(body_element, "visual", {"name": mesh_name})
            ET.SubElement(
                visual_element,
                "origin",
                {"xyz": _array2str(geom_pos), "rpy": _array2str(geom_rpy)},
            )
            geometry_element = ET.SubElement(visual_element, "geometry")
            ET.SubElement(
                geometry_element,
                "mesh",
                {"filename": f"{asset_file_prefix}converted_{mesh_name}.stl"},
            )
            material_element = ET.SubElement(
                visual_element, "material", {"name": "white"}
            )

            # create STL
            vertadr = model.mesh_vertadr[geom_dataid]
            vertnum = model.mesh_vertnum[geom_dataid]
            vert = model.mesh_vert[vertadr : vertadr + vertnum]
            faceadr = model.mesh_faceadr[geom_dataid]
            facenum = model.mesh_facenum[geom_dataid]
            face = model.mesh_face[faceadr : faceadr + facenum]
            data = np.zeros(facenum, dtype=mesh.Mesh.dtype)
            for i in range(facenum):
                data["vectors"][i] = vert[face[i]]
            m = mesh.Mesh(data, remove_empty_areas=False)
            urdf_base = urdf_file.parent
            if asset_file_prefix:
                stl_file = urdf_base / asset_file_prefix / f"converted_{mesh_name}.stl"
            else:
                stl_file = urdf_base / f"converted_{mesh_name}.stl"
            stl_file.parent.mkdir(parents=True, exist_ok=True)
            m.save(stl_file)

        if child_name == "world":
            # there is no joint connecting the world to anything, since it is the root
            assert parent_name == "world"
            assert jntnum == 0
            continue  # skip adding joint element or parent body

        # create dummy body for joint
        jnt_body_name = f"{jnt_name}_jointbody"
        _create_dummy_body(root, jnt_body_name)
        # connect parent to joint body with appropriate joint type
        parentbody2jnt_pos = (
            parentbody2childbody_pos + parentbody2childbody_Rot @ childbody2jnt_pos
        )
        parentbody2jnt_rpy = parentbody2childbody_rpy
        _create_joint(
            root,
            jnt_name,
            parent_name,
            jnt_body_name,
            parentbody2jnt_pos,
            parentbody2jnt_rpy,
            parentbody2jnt_axis,
            jnt_range,
            urdf_jnt_type,
        )
        # connect joint body to child body with fixed joint
        jnt2childbody_pos = -childbody2jnt_pos
        jnt2childbody_rpy = np.zeros(3)
        _create_joint(
            root,
            f"{jnt_name}_offset",
            jnt_body_name,
            child_name,
            jnt2childbody_pos,
            jnt2childbody_rpy,
        )

    # define white material
    material_element = ET.SubElement(root, "material", {"name": "white"})
    ET.SubElement(material_element, "color", {"rgba": "1 1 1 1"})

    # write to file with pretty printing
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with urdf_file.open("w") as f:
        f.write(xmlstr)
