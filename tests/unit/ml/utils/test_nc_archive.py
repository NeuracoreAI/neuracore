import json
import zipfile
from unittest.mock import MagicMock, patch

from neuracore_types import DataType
from omegaconf import OmegaConf

from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.preprocessing.methods import ColorJitter, ResizePad
from neuracore.ml.utils.json_serialization import to_json_serializable
from neuracore.ml.utils.nc_archive import (
    create_nc_archive,
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


class _ArchiveOnlyModel:
    model_init_description: dict = {}

    def state_dict(self) -> dict:
        return {}


def test_create_nc_archive_writes_grouped_preprocessing_json(tmp_path):
    algorithm_loader = MagicMock(algorithm_dir=tmp_path)
    algorithm_loader.get_all_files.return_value = []
    algorithm_loader.find_dependency_file.return_value = None

    with patch(
        "neuracore.ml.utils.nc_archive.AlgorithmLoader", return_value=algorithm_loader
    ):
        archive_path = create_nc_archive(
            model=_ArchiveOnlyModel(),
            output_dir=tmp_path,
            input_preprocessing_config=PreprocessingConfiguration(
                {DataType.RGB_IMAGES: [ResizePad(size=[32, 32]), ColorJitter(hue=0.1)]}
            ),
            output_preprocessing_config=PreprocessingConfiguration(
                {DataType.DEPTH_IMAGES: [ResizePad(size=[16, 16])]}
            ),
        )

    with zipfile.ZipFile(archive_path) as archive:
        input_preprocessing_json = json.loads(
            archive.read("input_preprocessing_config.json")
        )
        output_preprocessing_json = json.loads(
            archive.read("output_preprocessing_config.json")
        )

    assert input_preprocessing_json == {
        "RGB_IMAGES": {
            "geometric": [{"resize_pad": {"size": [32, 32]}}],
            "photometric": [{
                "color_jitter": {
                    "brightness": 0.3,
                    "contrast": 0.3,
                    "saturation": 0.3,
                    "hue": 0.1,
                }
            }],
        }
    }
    assert output_preprocessing_json == {
        "DEPTH_IMAGES": {
            "geometric": [{"resize_pad": {"size": [16, 16]}}],
            "photometric": [],
        }
    }
