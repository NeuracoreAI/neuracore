import importlib.metadata
import subprocess
import sys
from unittest.mock import Mock

import pytest

from neuracore.ml.utils import algorithm_loader
from neuracore.ml.utils.algorithm_loader import (
    AlgorithmLoader,
    DependencyInstallError,
    install_packaged_algorithm_requirements,
)


@pytest.fixture
def run_calls(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Would install x", stderr="")

    monkeypatch.setattr(algorithm_loader.subprocess, "run", fake_run)
    return calls


def test_reads_pyproject_dependencies_and_drops_neuracore(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "algo"\nversion = "0"\n'
        'dependencies = ["transformers==4.53.2", "neuracore>=1", "einops"]\n'
    )

    assert AlgorithmLoader(tmp_path)._read_requirements() == [
        "transformers==4.53.2",
        "einops",
    ]


def test_reads_requirements_txt(tmp_path):
    (tmp_path / "requirements.txt").write_text("einops\nneuracore_types\n")

    assert AlgorithmLoader(tmp_path)._read_requirements() == ["einops"]


def test_both_manifests_raise(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "a"\n')
    (tmp_path / "requirements.txt").write_text("einops\n")

    with pytest.raises(DependencyInstallError, match="Keep only one"):
        AlgorithmLoader(tmp_path).install_requirements()


def test_finds_requirements_inside_a_zipped_top_level_folder(tmp_path):
    (tmp_path / "my_algo").mkdir()
    (tmp_path / "my_algo" / "requirements.txt").write_text("einops\n")

    assert AlgorithmLoader(tmp_path)._read_requirements() == ["einops"]


def test_the_shallowest_dependency_file_wins(tmp_path):
    (tmp_path / "requirements.txt").write_text("einops\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "requirements.txt").write_text("scipy\n")

    assert AlgorithmLoader(tmp_path).find_dependency_file() == (
        tmp_path / "requirements.txt"
    )


def test_dependency_files_in_sibling_folders_raise(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "requirements.txt").write_text("einops\n")

    with pytest.raises(DependencyInstallError, match="Keep only one"):
        AlgorithmLoader(tmp_path).find_dependency_file()


def test_dependency_files_below_the_depth_limit_are_ignored(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "requirements.txt").write_text("einops\n")
    (deep.parent / "requirements.txt").write_text("scipy\n")

    assert AlgorithmLoader(tmp_path).find_dependency_file() == (
        deep.parent / "requirements.txt"
    )


def test_hidden_and_double_underscore_folders_are_skipped(tmp_path):
    for name in (".venv", "__MACOSX"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text('[project]\nname = "x"\n')

    assert AlgorithmLoader(tmp_path).find_dependency_file() is None


def test_empty_dependencies_skip_install(tmp_path, run_calls):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "a"\nversion = "0"\ndependencies = []\n'
    )

    assert AlgorithmLoader(tmp_path).install_requirements() is True
    assert run_calls == []


def test_install_pins_core_packages_in_every_command(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("einops\n")
    calls = []

    def fake_run(cmd, **kwargs):
        with open(cmd[cmd.index("-c") + 1]) as f:
            calls.append((cmd, f.read()))
        return subprocess.CompletedProcess(cmd, 0, stdout="Would install x", stderr="")

    monkeypatch.setattr(algorithm_loader.subprocess, "run", fake_run)

    AlgorithmLoader(tmp_path).install_requirements()

    torch_pin = f"torch=={importlib.metadata.version('torch')}"
    assert ["--dry-run" in cmd for cmd, _ in calls] == [True, False]
    assert all(torch_pin in constraints.splitlines() for _, constraints in calls)


@pytest.fixture
def loader_class(monkeypatch):
    install_packaged_algorithm_requirements.cache_clear()
    loader_class = Mock()
    monkeypatch.setattr(algorithm_loader, "AlgorithmLoader", loader_class)
    yield loader_class
    install_packaged_algorithm_requirements.cache_clear()


def test_packaged_install_uses_the_algorithm_folder(loader_class):
    install_packaged_algorithm_requirements(
        "neuracore.ml.algorithms.diffusion_policy.diffusion_policy.DiffusionPolicy"
    )

    folder = loader_class.call_args.args[0]
    assert folder.name == "diffusion_policy"
    assert folder.parent.name == "algorithms"
    loader_class.return_value.install_requirements.assert_called_once_with()


def test_packaged_install_runs_once_per_algorithm(loader_class):
    target = "neuracore.ml.algorithms.act.act.ACT"

    install_packaged_algorithm_requirements(target)
    install_packaged_algorithm_requirements(target)

    loader_class.return_value.install_requirements.assert_called_once_with()


def _write_algorithm(root, class_name):
    """Write an archive-style ``algorithm`` package whose model uses a relative import.

    Returns:
        The ``algorithm`` directory.
    """
    algorithm_dir = root / "algorithm"
    algorithm_dir.mkdir(parents=True)
    (algorithm_dir / "modules.py").write_text(f'NAME = "{class_name}"\n')
    (algorithm_dir / "model.py").write_text(
        "from neuracore.ml.core.neuracore_model import NeuracoreModel\n"
        "from .modules import NAME\n\n\n"
        f"class {class_name}(NeuracoreModel):\n"
        "    def name(self):\n"
        "        return NAME\n"
    )
    (algorithm_dir / "__init__.py").write_text(f"from .model import {class_name}\n")
    return algorithm_dir


@pytest.fixture
def restore_algorithm_modules():
    """Drop the test ``algorithm`` packages from sys.modules and sys.path afterwards."""
    path = list(sys.path)
    yield
    sys.path[:] = path
    for name in [n for n in sys.modules if n.startswith("algorithm")]:
        del sys.modules[name]


def test_second_algorithm_in_one_process_loads_its_own_code(
    tmp_path, restore_algorithm_modules
):
    first = AlgorithmLoader(_write_algorithm(tmp_path / "a", "ModelA")).load_model()
    second = AlgorithmLoader(_write_algorithm(tmp_path / "b", "ModelB")).load_model()

    assert first.__name__ == "ModelA"
    assert second.__name__ == "ModelB"
    # The first model still runs its own code after the second one loads.
    assert first.name(None) == "ModelA"
    assert second.name(None) == "ModelB"


def test_reloading_an_earlier_algorithm_loads_it_again(
    tmp_path, restore_algorithm_modules
):
    dir_a = _write_algorithm(tmp_path / "a", "ModelA")
    AlgorithmLoader(dir_a).load_model()
    AlgorithmLoader(_write_algorithm(tmp_path / "b", "ModelB")).load_model()

    assert AlgorithmLoader(dir_a).load_model().__name__ == "ModelA"
