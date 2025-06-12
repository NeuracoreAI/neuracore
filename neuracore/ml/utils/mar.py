"""Model archive (MAR) creation utility for Neuracore model deployment.

This module provides functionality to package Neuracore models into TorchServe
Model Archive (.mar) files for deployment. It handles model serialization,
dependency management, and packaging of all required files for inference.
"""

import inspect
import json
from pathlib import Path

import torch
from model_archiver.model_archiver import ModelArchiver
from model_archiver.model_archiver_config import ModelArchiverConfig

from neuracore.ml.core.neuracore_model import NeuracoreModel
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader


def create_mar(
    model: NeuracoreModel, output_dir: Path, algorithm_config: dict = {}
) -> None:
    """Create a TorchServe Model Archive (MAR) file from a Neuracore model.

    Packages a trained Neuracore model into a deployable MAR file that includes
    the model weights, algorithm code, configuration metadata, and dependencies.
    The resulting MAR file can be deployed to TorchServe for inference.

    Args:
        model: Trained Neuracore model instance to package for deployment.
        output_dir: Directory path where the MAR file and temporary artifacts
            will be created.
        algorithm_config: Custom configuration for the algorithm.
    """
    algorithm_file = Path(inspect.getfile(model.__class__))
    algorithm_loader = AlgorithmLoader(algorithm_file.parent)
    algo_files = algorithm_loader.get_all_files()

    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "model_init_description.json", "w") as f:
        json.dump(model.model_init_description.model_dump(), f, indent=2)
    if algorithm_config is not None:
        with open(output_dir / "algorithm_config.json", "w") as f:
            json.dump(algorithm_config, f, indent=2)
    with open(output_dir / "requirements.txt", "w") as f:
        f.write("neuracore\n")

    extra_files = [str(f) for f in algo_files] + [
        str(output_dir / "model_init_description.json"),
        str(output_dir / "algorithm_config.json"),
        str(output_dir / "requirements.txt"),
    ]

    FILE_PATH = Path(__file__).parent / "handlers.py"
    ModelArchiver.generate_model_archive(
        ModelArchiverConfig(
            model_name="model",
            version="1.0",
            model_file=str(algorithm_file),
            serialized_file=str(output_dir / "model.pt"),
            handler=str(FILE_PATH.resolve()),
            export_path=str(output_dir),
            extra_files=",".join(extra_files),
            force=True,
            requirements_file=str(output_dir / "requirements.txt"),
        )
    )
