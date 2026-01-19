import sys
import types

import pytest
from typer.testing import CliRunner

from neuracore import __version__
from neuracore.core.cli.app import app
from neuracore.core.organizations import Organization

runner = CliRunner()


@pytest.fixture
def setup_torch_availability(monkeypatch, request):
    """Fixture to setup torch availability based on parametrize."""
    torch_available = request.param

    if torch_available:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)


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


@pytest.mark.parametrize(
    "setup_torch_availability",
    [True, False],
    indirect=True,
    ids=["torch_available", "torch_not_available"],
)
def test_neuracore_login_works_regardless_of_torch(
    monkeypatch, setup_torch_availability
) -> None:
    """Test that login works with or without torch available."""

    # Setup mocks for auth
    monkeypatch.setattr(
        "neuracore.core.cli.generate_api_key.generate_api_key",
        lambda email=None, password=None: "test_api_key",
    )

    # Test login command
    result = runner.invoke(
        app,
        ["login", "--email", "user@example.com", "--password", "pw"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "setup_torch_availability",
    [True, False],
    indirect=True,
    ids=["torch_available", "torch_not_available"],
)
def test_neuracore_select_org_works_regardless_of_torch(
    monkeypatch, setup_torch_availability
) -> None:
    """Test that select-org works with or without torch available."""

    # Setup mocks for org operations
    class DummyAuth:
        is_authenticated = True

        def login(self, api_key=None):
            pass

    monkeypatch.setattr(
        "neuracore.core.cli.select_current_org.get_auth", lambda: DummyAuth()
    )
    monkeypatch.setattr(
        "neuracore.core.cli.select_current_org.list_my_orgs",
        lambda: [Organization(id="org-1", name="Test Org")],
    )

    # Test select-org command
    result = runner.invoke(
        app,
        ["select-org", "--org-name", "Test Org"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )
    assert result.exit_code == 0
