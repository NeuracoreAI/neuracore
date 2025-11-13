"""Model archive creation utility for Neuracore model deployment.

This module provides functionality to package Neuracore models into simple
ZIP archives (.nc.zip) for deployment. It handles model serialization,
dependency management, and packaging of all required files for inference.
"""

import inspect
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch
from neuracore_types import ModelInitDescription

from neuracore.ml.core.neuracore_model import NeuracoreModel
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.device_utils import get_default_device


def create_nc_archive(
    model: NeuracoreModel, output_dir: Path, algorithm_config: dict = {}
) -> Path:
    """Create a Neuracore model archive (NC.ZIP) file from a Neuracore model.

    Packages a trained Neuracore model into a deployable ZIP file that includes
    the model weights, algorithm code, configuration metadata, and dependencies.
    The resulting NC.ZIP file can be deployed for inference.

    Args:
        model: Trained Neuracore model instance to package for deployment.
        output_dir: Directory path where the NC.ZIP file will be created.
        algorithm_config: Custom configuration for the algorithm.

    Returns:
        Path to the created NC.ZIP file.
    """
    algorithm_file = Path(inspect.getfile(model.__class__))
    algorithm_loader = AlgorithmLoader(algorithm_file.parent)
    algo_files = algorithm_loader.get_all_files()
    requirements_file_path = algorithm_loader.algorithm_dir / "requirements.txt"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the archive filename
    archive_path = output_dir / "model.nc.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save model weights
        torch.save(model.state_dict(), temp_path / "model.pt")

        # Save model initialization description
        with open(temp_path / "model_init_description.json", "w") as f:
            json.dump(model.model_init_description.model_dump(mode="json"), f, indent=2)

        # Save algorithm config (always create file, even if empty)
        with open(temp_path / "algorithm_config.json", "w") as f:
            json.dump(algorithm_config, f, indent=2)

        # Create the ZIP archive
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add model weights
            zip_file.write(temp_path / "model.pt", "model.pt")

            # Add model initialization description
            zip_file.write(
                temp_path / "model_init_description.json", "model_init_description.json"
            )

            # Add algorithm config (always present)
            zip_file.write(temp_path / "algorithm_config.json", "algorithm_config.json")

            # Add all algorithm files
            for algo_file in algo_files:
                # Calculate relative path from algorithm directory
                rel_path = algo_file.relative_to(algorithm_loader.algorithm_dir)
                zip_file.write(algo_file, f"algorithm/{rel_path}")

            # Add requirements file if it exists
            if requirements_file_path.exists():
                zip_file.write(requirements_file_path, "requirements.txt")

    return archive_path


def extract_nc_archive(archive_file: Path, output_dir: Path) -> dict[str, Path]:
    """Extract all contents from a Neuracore model archive (NC.ZIP) file.

    Extracts all files from a NC.ZIP archive including model weights, algorithm code,
    configuration files, and dependencies.

    Args:
        archive_file: Path to the NC.ZIP file to extract.
        output_dir: Directory where extracted files will be saved.

    Returns:
        Dictionary mapping file types to their extracted paths.

    Raises:
        FileNotFoundError: If the archive file doesn't exist.
        zipfile.BadZipFile: If the archive file is corrupted or not a valid ZIP.
    """
    if not archive_file.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: dict[str, Any] = {}

    with zipfile.ZipFile(archive_file, "r") as zip_ref:
        # Extract all files
        zip_ref.extractall(output_dir)

        # Catalog the extracted files
        for file_info in zip_ref.filelist:
            file_path = output_dir / file_info.filename

            # Categorize files based on their names/extensions
            if file_info.filename == "model.pt":
                extracted_files["model_weights"] = file_path
            elif file_info.filename == "model_init_description.json":
                extracted_files["model_init_description"] = file_path
            elif file_info.filename == "algorithm_config.json":
                extracted_files["algorithm_config"] = file_path
            elif file_info.filename == "requirements.txt":
                extracted_files["requirements"] = file_path
            elif file_info.filename.startswith("algorithm/"):
                extracted_files.setdefault("algorithm_files", []).append(file_path)
            else:
                extracted_files.setdefault("other_files", []).append(file_path)

    return extracted_files


def load_model_from_nc_archive(
    archive_file: Path, extract_to: Optional[Path] = None, device: Optional[str] = None
) -> NeuracoreModel:
    """Load a Neuracore model from a NC.ZIP archive file.

    Extracts the archive file and reconstructs the original Neuracore model instance
    with its trained weights and configuration.

    Args:
        archive_file: Path to the NC.ZIP file.
        extract_to: Optional directory to extract files to.
            If None, uses a temporary directory.
        device: Optional device model to be loaded on

    Returns:
        NeuracoreModel: The reconstructed model instance ready for inference.
    """
    use_temp_dir = extract_to is None

    if use_temp_dir:
        temp_dir_context = tempfile.TemporaryDirectory()
        extract_to = Path(temp_dir_context.__enter__())
    else:
        temp_dir_context = None

    assert extract_to is not None

    try:
        # Extract the archive file
        extracted_files = extract_nc_archive(archive_file, extract_to)

        # Load model initialization description
        if "model_init_description" not in extracted_files:
            raise FileNotFoundError("model_init_description.json not found in archive")

        with open(extracted_files["model_init_description"], "r") as f:
            model_init_description = json.load(f)
        model_init_description = ModelInitDescription.model_validate(
            model_init_description
        )

        # Load algorithm config if present
        algorithm_config = {}
        if "algorithm_config" in extracted_files:
            with open(extracted_files["algorithm_config"], "r") as f:
                algorithm_config = json.load(f)

        # Find the algorithm directory
        algorithm_dir = extract_to / "algorithm"
        if not algorithm_dir.exists():
            raise FileNotFoundError("Algorithm directory not found in archive")

        # Load the algorithm using AlgorithmLoader
        algorithm_loader = AlgorithmLoader(algorithm_dir)
        algorithm_loader.install_requirements()
        model_class = algorithm_loader.load_model()

        # Create model instance
        if device:
            device = torch.device(device)
        else:
            device = get_default_device()
        model = model_class(model_init_description, **algorithm_config)
        model = model.to(device)

        # Load trained weights if present
        if "model_weights" in extracted_files:
            state_dict = torch.load(
                extracted_files["model_weights"],
                map_location=model.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

        return model

    finally:
        if use_temp_dir and temp_dir_context:
            temp_dir_context.__exit__(None, None, None)
