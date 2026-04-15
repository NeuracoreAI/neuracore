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
    assert "cache" in result.output
    assert "training" in result.output


def test_neuracore_cache_clear_requires_confirmation(monkeypatch, tmp_path) -> None:
    recording_cache = tmp_path / "recording_cache"
    dataset_cache = tmp_path / "dataset_cache"
    recording_cache.mkdir()
    dataset_cache.mkdir()
    (recording_cache / "frame.png").write_bytes(b"abc")
    (dataset_cache / "stats.json").write_text("{}")

    monkeypatch.setattr(
        "neuracore.core.cli.cache_commands.DEFAULT_RECORDING_CACHE_DIR",
        recording_cache,
    )
    monkeypatch.setattr(
        "neuracore.core.cli.cache_commands.DEFAULT_DATASET_CACHE_DIR",
        dataset_cache,
    )

    result = runner.invoke(
        app,
        ["cache", "clear"],
        input="n\n",
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Clear these cache files?" in result.output
    assert "Aborted." in result.output
    assert (recording_cache / "frame.png").exists()
    assert (dataset_cache / "stats.json").exists()


def test_neuracore_cache_clear_yes_deletes_only_cache_contents(
    monkeypatch, tmp_path
) -> None:
    recording_cache = tmp_path / "recording_cache"
    dataset_cache = tmp_path / "dataset_cache"
    training_runs = tmp_path / "runs"
    recording_cache.mkdir()
    dataset_cache.mkdir()
    training_runs.mkdir()
    (recording_cache / "frame.png").write_bytes(b"abc")
    (dataset_cache / "stats.json").write_text("{}")
    (training_runs / "training_run.json").write_text("{}")

    monkeypatch.setattr(
        "neuracore.core.cli.cache_commands.DEFAULT_RECORDING_CACHE_DIR",
        recording_cache,
    )
    monkeypatch.setattr(
        "neuracore.core.cli.cache_commands.DEFAULT_DATASET_CACHE_DIR",
        dataset_cache,
    )

    result = runner.invoke(
        app,
        ["cache", "clear", "--yes"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Training runs, auth config, and daemon state will not be deleted." in (
        result.output
    )
    assert "Cleared" in result.output
    assert recording_cache.exists()
    assert dataset_cache.exists()
    assert not (recording_cache / "frame.png").exists()
    assert not (dataset_cache / "stats.json").exists()
    assert (training_runs / "training_run.json").exists()


def test_neuracore_cache_clear_dry_run_preserves_files(monkeypatch, tmp_path) -> None:
    recording_cache = tmp_path / "recording_cache"
    dataset_cache = tmp_path / "dataset_cache"
    recording_cache.mkdir()
    dataset_cache.mkdir()
    (recording_cache / "frame.png").write_bytes(b"abc")
    (dataset_cache / "stats.json").write_text("{}")

    monkeypatch.setattr(
        "neuracore.core.cli.cache_commands.DEFAULT_RECORDING_CACHE_DIR",
        recording_cache,
    )
    monkeypatch.setattr(
        "neuracore.core.cli.cache_commands.DEFAULT_DATASET_CACHE_DIR",
        dataset_cache,
    )

    result = runner.invoke(
        app,
        ["cache", "clear", "--dry-run"],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert "Dry run: no files were deleted." in result.output
    assert (recording_cache / "frame.png").exists()
    assert (dataset_cache / "stats.json").exists()


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
    assert "Input embodiment description" in result.output
    assert "Output embodiment description" in result.output
    assert "indexed name mapping" in result.output


def test_neuracore_launch_server_accepts_documented_json(monkeypatch) -> None:
    launched = {}

    class DummyPolicy:
        class ServerProcess:
            def wait(self) -> None:
                launched["waited"] = True

        server_process = ServerProcess()

    def fake_policy_local_server(**kwargs):
        launched.update(kwargs)
        return DummyPolicy()

    monkeypatch.setattr("neuracore.core.cli.launch_server.nc.login", lambda: None)
    monkeypatch.setattr(
        "neuracore.core.cli.launch_server.nc.set_organization", lambda org_id: None
    )
    monkeypatch.setattr(
        "neuracore.core.cli.launch_server.policy_local_server",
        fake_policy_local_server,
    )

    result = runner.invoke(
        app,
        [
            "launch-server",
            "--input_embodiment_description",
            '{"RGB_IMAGES": {"0": "front_cam"}}',
            "--output_embodiment_description",
            '{"JOINT_TARGET_POSITIONS": {"0": "arm"}}',
            "--job_id",
            "run-1",
            "--org_id",
            "org-1",
        ],
        color=False,
        env={"TERM": "dumb", "NO_COLOR": "1", "RICH_DISABLE": "1"},
    )

    assert result.exit_code == 0
    assert launched["input_embodiment_description"]
    assert launched["output_embodiment_description"]
    assert launched["waited"] is True


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
