"""MCAP round trip configured with the constants below.

Exports, imports with the existing importer, and checks every recording lands.
"""

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml
from neuracore_types import DataType
from neuracore_types.nc_data import (
    DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS,
    DatasetImportConfig,
)

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from neuracore.core.organizations import list_my_orgs
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

ORG_NAME = "Service Account's Project"  # Enter your organization name.
DATASET_NAME = "Data Upload Integrity Check"


def _import_config(recording, dataset_name: str, robot_name: str):
    """Map every exported sensor to its raw value; never omit a modality."""

    fields = {
        DataType.JOINT_POSITIONS: "value",
        DataType.JOINT_TARGET_POSITIONS: "value",
        DataType.JOINT_VELOCITIES: "value",
        DataType.JOINT_TORQUES: "value",
        DataType.VISUAL_JOINT_POSITIONS: "value",
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: "open_amount",
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: "open_amount",
        DataType.POSES: "pose",
        DataType.END_EFFECTOR_POSES: "pose",
        DataType.CUSTOM_1D: "data",
        DataType.LANGUAGE: "text",
        # Image messages carry compressed pixels; map the complete message.
        DataType.RGB_IMAGES: "",
        DataType.DEPTH_IMAGES: "",
        DataType.POINT_CLOUDS: "",
    }
    mappings = {}
    for kind, names in recording.sensor_manifest.items():
        field = fields[kind]  # Unknown future modalities must fail, not disappear.
        mappings[kind.value] = {
            "mapping": [
                {
                    "name": name,
                    "source_name": f"/neuracore/{kind.value}/"
                    f"{quote(name, safe='').replace('.', '%2E')}"
                    + (f".{field}" if field else ""),
                }
                for name in names
            ]
        }
    for kind in (DataType.POSES, DataType.END_EFFECTOR_POSES):
        if kind.value in mappings:
            mappings[kind.value]["format"] = {
                "pose_type": "POSITION_ORIENTATION",
                "orientation": {"type": "QUATERNION", "quaternion_order": "XYZW"},
            }
            for item in mappings[kind.value]["mapping"]:
                item["index_range"] = {"start": 0, "end": 7}
    config = DatasetImportConfig(
        input_dataset_name=recording.dataset.name,
        dataset_type="MCAP",
        output_dataset={"name": dataset_name},
        robot={"name": robot_name},
        data_import_config={
            DataType(kind): DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS[DataType(kind)](
                **config
            )
            for kind, config in mappings.items()
        },
    )

    # Scalar topic fields are already selected; remove automatic array indexes.
    for kind, mapping in config.data_import_config.items():
        if fields[kind] in ("value", "open_amount"):
            for item in mapping.mapping:
                item.index = None
    return config


def _write_synthetic_urdf(joint_names: set[str], path: Path) -> None:
    """Write a throwaway serial-chain URDF so joints have a robot model.

    This fixture's joints have no real kinematics or limits; a minimal
    continuous-joint chain is enough for the importer to accept them.
    """
    links = ['  <link name="link_0"/>']
    joints = []
    for i, name in enumerate(sorted(joint_names)):
        links.append(f'  <link name="link_{i + 1}"/>')
        joints.append(
            f'  <joint name="{name}" type="continuous">\n'
            f'    <parent link="link_{i}"/>\n'
            f'    <child link="link_{i + 1}"/>\n'
            f'    <axis xyz="0 0 1"/>\n'
            "  </joint>"
        )
    urdf = (
        '<robot name="mcap_integrity_robot">\n'
        + "\n".join(links)
        + "\n"
        + "\n".join(joints)
        + "\n</robot>\n"
    )
    path.write_text(urdf)


def test_mcap_dataset_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export every recording to MCAP, re-import it, and check it all lands."""

    assert ORG_NAME, "Set ORG_NAME at the top of this file"

    monkeypatch.setenv("NCD_VIDEO_CODEC", "h264_lossless")

    nc.login()

    matches = [org for org in list_my_orgs() if org.name == ORG_NAME]
    assert (
        len(matches) == 1
    ), f"Expected one organization named {ORG_NAME!r}, found {len(matches)}"
    monkeypatch.setenv("NEURACORE_ORG_ID", matches[0].id)

    source = Dataset.get_by_name(DATASET_NAME)
    originals = list(source)
    assert originals, "An empty source dataset cannot test integrity"

    neuracore_cli = shutil.which("neuracore")
    assert neuracore_cli is not None, "neuracore CLI entry point not on PATH"

    export_dir = tmp_path / "export"
    export_command = [
        neuracore_cli,
        "export",
        "mcap",
        "--output",
        str(export_dir),
        "--dataset",
        DATASET_NAME,
    ]
    print("Running:", " ".join(export_command))
    subprocess.run(export_command, check=True)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    source_ids = [r.id for r in originals]
    assert manifest["status"] == "succeeded"
    assert manifest["completed_recording_ids"] == source_ids
    assert {entry["recording_id"] for entry in manifest["files"]} == set(source_ids)

    run_id = uuid.uuid4().hex
    dataset_name = f"mcap_integrity_{run_id}"
    robot_name = f"mcap_integrity_robot_{run_id}"

    # All recordings share one config; the CLI imports every MCAP file in
    # export_dir in a single run, one recording per file.
    config = _import_config(originals[0], dataset_name, robot_name)
    config_path = tmp_path / "import_config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))

    # This synthetic fixture has named joints but no real URDF; generate a
    # throwaway one so the CLI's robot model recognizes every joint name.
    joint_names = {
        name
        for recording in originals
        for kind, names in recording.sensor_manifest.items()
        if "JOINT" in kind.value
        for name in names
    }
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    _write_synthetic_urdf(joint_names, robot_dir / "robot.urdf")

    import_command = [
        neuracore_cli,
        "importer",
        "import",
        "--dataset-config",
        str(config_path),
        "--dataset-dir",
        str(export_dir),
        "--robot-dir",
        str(robot_dir),
        "--max-workers",
        "1",
        "--skip-on-error",
        "all",
        "--no-validation-warnings",
    ]
    print("Running:", " ".join(import_command))

    try:
        with online_daemon_running():
            subprocess.run(import_command, check=True)

            deadline = time.monotonic() + 300
            uploaded_count = len(Dataset.get_by_name(dataset_name))
            while uploaded_count != len(originals) and time.monotonic() < deadline:
                time.sleep(2)
                uploaded_count = len(Dataset.get_by_name(dataset_name))
            assert uploaded_count == len(originals), (
                f"Expected {len(originals)} uploaded recordings, "
                f"found {uploaded_count} after 300s"
            )
    finally:
        target = Dataset.get_by_name(dataset_name, non_exist_ok=True)
        if target is not None:
            target.delete()
        nc.connect_robot(robot_name).delete()
