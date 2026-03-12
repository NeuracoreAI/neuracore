"""Tests for nc_archive creation utilities."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from neuracore.ml.utils.nc_archive import create_nc_archive
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler


def _fake_torch_save(obj, path, **kwargs):
    """Write an empty file so downstream .stat() calls succeed."""
    Path(path).write_bytes(b"")


def _make_model(tmp_path: Path):
    """Return a minimal mock NeuracoreModel whose class lives in tmp_path."""
    algo_dir = tmp_path / "algorithm"
    algo_dir.mkdir()
    (algo_dir / "model.py").write_text("# stub algorithm\n")

    model = MagicMock()
    model.__class__ = type("StubModel", (), {"__module__": __name__})
    model.state_dict.return_value = {}

    model_init_desc = MagicMock()
    model_init_desc.model_dump.return_value = {}
    model.model_init_description = model_init_desc

    return model, algo_dir


def _extract_metadata(archive_path: Path) -> dict:
    """Read and parse the 'metadata' file from the nc.zip archive."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        with zf.open("metadata") as f:
            return json.load(f)


def _create_archive(tmp_path: Path, **kwargs) -> Path:
    """Helper that wires up the necessary patches and calls create_nc_archive."""
    model, algo_dir = _make_model(tmp_path)
    output_dir = tmp_path / "out"
    with patch(
        "neuracore.ml.utils.nc_archive.inspect.getfile",
        return_value=str(algo_dir / "model.py"),
    ), patch(
        "neuracore.ml.utils.nc_archive.torch.save",
        side_effect=_fake_torch_save,
    ):
        return create_nc_archive(model=model, output_dir=output_dir, **kwargs)


class TestCreateNcArchiveMetadata:
    def test_training_metadata_written_verbatim_to_archive(self, tmp_path):
        training_metadata = {
            "neuracore_version": "1.2.3",
            "neuracore_types_version": "0.5.0",
            "dataset_sync_frequency": 30.0,
        }
        archive_path = _create_archive(tmp_path, training_metadata=training_metadata)
        assert _extract_metadata(archive_path) == training_metadata

    def test_empty_training_metadata_produces_empty_metadata_file(self, tmp_path):
        archive_path = _create_archive(tmp_path)
        assert _extract_metadata(archive_path) == {}


class TestTrainingMetadataFlowFromTrainPy:
    """Verify training_metadata set in train.py flows into the archive metadata.

    train.py calls importlib.metadata.version() directly to obtain package
    versions and bundles them with dataset_sync_frequency into training_metadata,
    which is passed to TrainingStorageHandler. These tests simulate that path
    end-to-end: handler → save_model_artifacts → create_nc_archive → metadata in zip.
    """

    def test_training_metadata_from_cfg_ends_up_in_archive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "neuracore.ml.utils.training_storage_handler.get_current_org",
            lambda: "test-org",
        )
        model, algo_dir = _make_model(tmp_path)
        # Mirrors what train.py builds from cfg and importlib.metadata.version()
        training_metadata = {
            "neuracore_version": "1.0.0",
            "neuracore_types_version": "0.5.0",
            "dataset_sync_frequency": 15.0,
        }
        handler = TrainingStorageHandler(
            local_dir=str(tmp_path),
            training_metadata=training_metadata,
        )

        with patch(
            "neuracore.ml.utils.nc_archive.inspect.getfile",
            return_value=str(algo_dir / "model.py"),
        ), patch(
            "neuracore.ml.utils.nc_archive.torch.save",
            side_effect=_fake_torch_save,
        ):
            handler.save_model_artifacts(model, Path("run_1"))

        archive_path = tmp_path / "run_1" / "artifacts" / "model.nc.zip"
        assert archive_path.exists()
        assert _extract_metadata(archive_path) == training_metadata
