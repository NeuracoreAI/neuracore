"""Shared command validation and failure reporting for dataset exports."""

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from neuracore.core.exceptions import AuthenticationError, SynchronizationError
from neuracore.exporter import cli


@pytest.fixture(params=[("mcap", []), ("lerobot", ["--fps", "10"])])
def command(request):
    name, options = request.param
    return [name, *options]


@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_validates_before_login(command, existing_output, tmp_path, monkeypatch):
    login = Mock()
    monkeypatch.setattr(cli, "login", login)
    output = tmp_path / "out"
    args = [*command, "--output", str(output)]
    if existing_output:
        output.mkdir()
        args.extend(["--dataset", "Demo"])

    result = CliRunner().invoke(cli.export_app, args)

    assert result.exit_code == 2
    expected = "already exists" if existing_output else "Provide exactly one"
    assert expected in result.output
    login.assert_not_called()


@pytest.mark.parametrize("stage", ["login", "export_recordings"])
def test_cli_reports_failures(command, stage, tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr(cli, "McapExporter", Mock(return_value=exporter))
    monkeypatch.setattr(cli, "LeRobotExporter", Mock(return_value=exporter))
    monkeypatch.setattr(cli, "login", Mock())
    monkeypatch.setattr(cli, "get_dataset", Mock(return_value=[]))
    error = AuthenticationError if stage == "login" else SynchronizationError
    monkeypatch.setattr(cli, stage, Mock(side_effect=error("unavailable")))
    output = tmp_path / "out"

    result = CliRunner().invoke(
        cli.export_app, [*command, "--dataset", "Demo", "--output", str(output)]
    )

    assert result.exit_code == 1
    assert "Export failed: unavailable" in result.output
    assert "Export complete" not in result.output
    assert not output.exists()


def test_cli_reports_invalid_fps_before_login(tmp_path, monkeypatch):
    login = Mock()
    monkeypatch.setattr(cli, "login", login)
    result = CliRunner().invoke(
        cli.export_app,
        ["lerobot", "--fps", "0", "--dataset", "Demo", "-o", str(tmp_path / "out")],
    )

    assert result.exit_code == 1
    assert "fps must be positive" in result.output
    login.assert_not_called()
