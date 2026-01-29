import argparse
import sys

import pytest

from neuracore.importer.core.exceptions import CLIError
from neuracore.importer.importer import cli_args_validation, parse_args


def test_parse_args_requires_robot_description(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "importer.py",
            "--dataset-config",
            "config.yaml",
            "--dataset-dir",
            "dataset_dir",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        parse_args()

    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err.lower()


def test_cli_args_validation_missing_dataset_config(tmp_path):
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()

    args = argparse.Namespace(
        dataset_config=tmp_path / "missing_config.yaml",
        dataset_dir=dataset_dir,
        robot_dir=tmp_path / "robot_dir",
    )

    with pytest.raises(CLIError) as excinfo:
        cli_args_validation(args)

    assert f"Path does not exist: {args.dataset_config}" in str(excinfo.value)


def test_cli_args_validation_missing_dataset_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    args = argparse.Namespace(
        dataset_config=config_path,
        dataset_dir=tmp_path / "missing_dataset",
        robot_dir=tmp_path / "robot_dir",
    )

    with pytest.raises(CLIError) as excinfo:
        cli_args_validation(args)

    assert f"Path does not exist: {args.dataset_dir}" in str(excinfo.value)


def test_cli_args_validation_missing_robot_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()

    args = argparse.Namespace(
        dataset_config=config_path,
        dataset_dir=dataset_dir,
        robot_dir=tmp_path / "missing_robot",
    )

    with pytest.raises(CLIError) as excinfo:
        cli_args_validation(args)

    assert "Robot description directory does not exist" in str(excinfo.value)


def test_cli_args_validation_valid_args(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()
    robot_dir = tmp_path / "robot_dir"
    robot_dir.mkdir()

    args = argparse.Namespace(
        dataset_config=config_path,
        dataset_dir=dataset_dir,
        robot_dir=robot_dir,
    )

    # Should not raise any exception
    cli_args_validation(args)
