import inspect
import json
from pathlib import Path

import torch
from model_archiver.model_archiver import ModelArchiver
from model_archiver.model_archiver_config import ModelArchiverConfig

from neuracore.ml.neuracore_model import NeuracoreModel
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader


def create_mar(model: NeuracoreModel, output_dir: str):
    algorithm_file = Path(inspect.getfile(model.__class__))
    algorithm_loader = AlgorithmLoader(algorithm_file.parent)
    algo_files = algorithm_loader.get_all_files()

    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "model_init_description.json", "w") as f:
        json.dump(model.model_init_description.model_dump(), f, indent=2)

    with open(output_dir / "requirements.txt", "w") as f:
        # TODO: Remove test PyPI for main branch
        f.write("--index-url https://test.pypi.org/simple\n")
        f.write("neuracore\n")

    extra_files = [str(f) for f in algo_files] + [
        str(output_dir / "model_init_description.json"),
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
