from typer.testing import CliRunner

from neuracore import __version__
from neuracore.core.cli.app import app

runner = CliRunner()


def test_neuracore_cli_version() -> None:
    result = runner.invoke(
        app,
        ["--version"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_neuracore_cli_help_includes_subcommands() -> None:
    result = runner.invoke(
        app,
        ["--help"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "login" in result.output
    assert "select-org" in result.output
    assert "launch-server" in result.output


def test_neuracore_login_help_includes_options() -> None:
    result = runner.invoke(
        app,
        ["login", "--help"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )
    assert result.exit_code == 0
    assert "--email" in result.output
    assert "--password" in result.output


def test_neuracore_select_org_help_includes_options() -> None:
    result = runner.invoke(
        app,
        ["select-org", "--help"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "--org-name" in result.output
    assert "--org-id" in result.output


def test_neuracore_launch_server_help_includes_options() -> None:
    result = runner.invoke(
        app,
        ["launch-server", "--help"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "--model_input_order" in result.output
    assert "--model_output_order" in result.output
