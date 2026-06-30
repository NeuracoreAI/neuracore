"""Guard that each algorithm hydra config matches its model ``__init__`` defaults.

The ``__init__`` signature is the single source of truth for algorithm
hyperparameters; the ``config/algorithm/*.yaml`` files are an override layer that
``hydra.utils.instantiate`` passes as kwargs. This test fails if the two drift.
"""

import inspect
import math
from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

import neuracore.ml as nc_ml

CONFIG_DIR = Path(nc_ml.__file__).parent / "config" / "algorithm"

CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


def _load_algorithm_block(path: Path) -> dict:
    """Return the ``algorithm:`` mapping from a hydra config yaml.

    Loaded via OmegaConf so values parse exactly as ``hydra.utils.instantiate``
    sees them (e.g. ``1e-4`` is a float, not a string).
    """
    cfg = OmegaConf.load(path)
    block = OmegaConf.to_container(cfg.algorithm, resolve=True)
    assert isinstance(block, dict)
    return block


def _init_defaults(cls: type) -> dict:
    """Return ``{param: default}`` for every defaulted ``__init__`` param."""
    signature = inspect.signature(cls.__init__)
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def _values_equal(yaml_value: object, default_value: object) -> bool:
    """Compare a yaml value with an ``__init__`` default, tolerant of list/tuple
    and float representation differences."""
    if isinstance(default_value, (list, tuple)) or isinstance(
        yaml_value, (list, tuple)
    ):
        yaml_seq = list(yaml_value)  # type: ignore[arg-type]
        default_seq = list(default_value)  # type: ignore[arg-type]
        if len(yaml_seq) != len(default_seq):
            return False
        return all(_values_equal(y, d) for y, d in zip(yaml_seq, default_seq))
    if isinstance(default_value, bool) or isinstance(yaml_value, bool):
        return yaml_value == default_value
    if isinstance(default_value, (int, float)) and isinstance(yaml_value, (int, float)):
        return math.isclose(yaml_value, default_value, rel_tol=1e-9, abs_tol=0.0)
    return yaml_value == default_value


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_hydra_yaml_values_match_init_defaults(config_path: Path) -> None:
    """Every yaml key matches its ``__init__`` default, and every defaulted
    ``__init__`` param is present in the yaml."""
    algorithm = _load_algorithm_block(config_path)
    target = algorithm["_target_"]
    cls = hydra.utils.get_object(target)
    defaults = _init_defaults(cls)

    yaml_keys = {key for key in algorithm if key != "_target_"}

    unknown = yaml_keys - defaults.keys()
    assert (
        not unknown
    ), f"{config_path.name}: keys not in {cls.__name__}.__init__: {sorted(unknown)}"

    missing = defaults.keys() - yaml_keys
    assert (
        not missing
    ), f"{config_path.name}: __init__ params absent from yaml: {sorted(missing)}"

    mismatched = {
        key: (algorithm[key], defaults[key])
        for key in yaml_keys
        if not _values_equal(algorithm[key], defaults[key])
    }
    assert not mismatched, (
        f"{config_path.name}: yaml values differ from {cls.__name__}.__init__ "
        f"defaults (yaml, default): {mismatched}"
    )
