import json
import zipfile

from neuracore_types import DataType
from omegaconf import OmegaConf

from neuracore.ml.utils.json_serialization import to_json_serializable
from neuracore.ml.utils.nc_archive import (
    load_cross_embodiment_descriptions_from_nc_archive,
)


class ModelDumpWithOmegaConf:
    def model_dump(self, mode):
        assert mode == "json"
        return {"model_init": OmegaConf.create({"hidden_dim": 128})}


def test_to_json_serializable_handles_nested_omegaconf_config():
    payload = {
        "algorithm_config": OmegaConf.create(
            {"optimizer": {"name": "adam", "lr": 1e-4}}
        ),
        "input_cross_embodiment_description": {
            "robot_id": {
                DataType.JOINT_POSITIONS: OmegaConf.create(
                    {0: "joint_positions_0", 1: "joint_positions_1"}
                )
            }
        },
    }

    result = to_json_serializable(payload)

    json.dumps(result)
    assert result == {
        "algorithm_config": {"optimizer": {"name": "adam", "lr": 1e-4}},
        "input_cross_embodiment_description": {
            "robot_id": {
                DataType.JOINT_POSITIONS: {
                    0: "joint_positions_0",
                    1: "joint_positions_1",
                }
            }
        },
    }


def test_to_json_serializable_recurses_after_model_dump():
    result = to_json_serializable(ModelDumpWithOmegaConf())

    json.dumps(result)
    assert result == {"model_init": {"hidden_dim": 128}}


def test_load_cross_embodiment_descriptions_normalizes_json_index_keys(tmp_path):
    archive_path = tmp_path / "model.nc.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "input_cross_embodiment_description.json",
            json.dumps({
                "robot_id": {
                    "JOINT_POSITIONS": {
                        "0": "joint_positions_0",
                        "1": "joint_positions_1",
                    }
                }
            }),
        )
        archive.writestr(
            "output_cross_embodiment_description.json",
            json.dumps({"robot_id": {"JOINT_TARGET_POSITIONS": {"0": "target_0"}}}),
        )

    input_description, output_description = (
        load_cross_embodiment_descriptions_from_nc_archive(archive_path)
    )

    assert input_description == {
        "robot_id": {
            DataType.JOINT_POSITIONS: {
                0: "joint_positions_0",
                1: "joint_positions_1",
            }
        }
    }
    assert output_description == {
        "robot_id": {DataType.JOINT_TARGET_POSITIONS: {0: "target_0"}}
    }
