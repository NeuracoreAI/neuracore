import argparse
import importlib
import logging

import pytest
from typer.testing import CliRunner

from neuracore.importer.cli.app import app
from neuracore.importer.core.exceptions import CLIError
from neuracore.importer.importer import cli_args_validation

runner = CliRunner()


def test_import_command_invokes_run_import_with_defaults(monkeypatch, tmp_path):
    called = {}

    def fake_run_import(**kwargs):
        called.update(kwargs)

    config_path = tmp_path / "config.yaml"
    config_path.touch()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cli_module = importlib.import_module("neuracore.importer.cli.app")
    monkeypatch.setattr(cli_module, "_run_import", fake_run_import)

    result = runner.invoke(
        app,
        [
            "--dataset-config",
            str(config_path),
            "--dataset-dir",
            str(dataset_dir),
        ],
        env={"TERM": "dumb", "NO_COLOR": "1"},
    )

    assert result.exit_code == 0
    assert called["dataset_config"] == config_path
    assert called["dataset_dir"] == dataset_dir
    assert called["robot_dir"] is None
    assert called["overwrite"] is False
    assert called["dry_run"] is False
    assert called["skip_on_error"] == "episode"
    assert called["suppress_validation_warnings"] is False


def test_import_command_propagates_flags(monkeypatch, tmp_path):
    captured = {}

    def fake_run_import(**kwargs):
        captured.update(kwargs)

    config_path = tmp_path / "config.yaml"
    config_path.touch()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cli_module = importlib.import_module("neuracore.importer.cli.app")
    monkeypatch.setattr(cli_module, "_run_import", fake_run_import)

    result = runner.invoke(
        app,
        [
            "--dataset-config",
            str(config_path),
            "--dataset-dir",
            str(dataset_dir),
            "--overwrite",
            "--dry-run",
            "--skip-on-error",
            "step",
            "--no-validation-warnings",
        ],
        env={"TERM": "dumb", "NO_COLOR": "1"},
    )

    assert result.exit_code == 0
    assert captured["overwrite"] is True
    assert captured["dry_run"] is True
    assert captured["skip_on_error"] == "step"
    assert captured["suppress_validation_warnings"] is True
    assert captured["robot_dir"] is None


def test_import_command_handles_cli_error(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cli_module = importlib.import_module("neuracore.importer.cli.app")
    monkeypatch.setattr(
        cli_module,
        "_run_import",
        lambda **_: (_ for _ in ()).throw(CLIError("bad input")),
    )

    with caplog.at_level(logging.ERROR):
        result = runner.invoke(
            app,
            [
                "--dataset-config",
                str(config_path),
                "--dataset-dir",
                str(dataset_dir),
            ],
            env={"TERM": "dumb", "NO_COLOR": "1"},
        )

    assert result.exit_code == 1
    assert any("bad input" in message for message in caplog.messages)


def test_import_command_validates_paths_with_click(tmp_path):
    missing_config = tmp_path / "missing.yaml"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    result = runner.invoke(
        app,
        [
            "--dataset-config",
            str(missing_config),
            "--dataset-dir",
            str(dataset_dir),
        ],
        env={"TERM": "dumb", "NO_COLOR": "1"},
    )

    assert result.exit_code == 2
    assert "Invalid value for '--dataset-config'" in result.output


def test_cli_args_validation_missing_dataset_config(tmp_path):
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()

    args = argparse.Namespace(
        dataset_config=tmp_path / "missing_config.yaml",
        dataset_dir=dataset_dir,
        robot_dir=None,
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
        robot_dir=None,
    )

    with pytest.raises(CLIError) as excinfo:
        cli_args_validation(args)

    assert f"Path does not exist: {args.dataset_dir}" in str(excinfo.value)


def test_cli_args_validation_missing_robot_dir_when_provided(tmp_path):
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


def test_cli_args_validation_none_robot_dir_is_allowed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()

    args = argparse.Namespace(
        dataset_config=config_path,
        dataset_dir=dataset_dir,
        robot_dir=None,
    )

    # Should not raise any exception
    cli_args_validation(args)


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
