"""Utilities for normalizing Hydra training configuration."""

import copy
import re
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from names_generator import generate_name
from neuracore_types import (
    AlgorithmConfig,
    CrossEmbodimentDescription,
    DataType,
    SynchronizationConfig,
    TrainingConfig,
)
from omegaconf import DictConfig, OmegaConf

import neuracore as nc
from neuracore.api.training import _get_algorithms, get_algorithm
from neuracore.core.const import DEFAULT_CACHE_DIR
from neuracore.core.data.dataset import Dataset
from neuracore.core.utils.embodiment_description_utils import (
    convert_cross_embodiment_description_names_to_ids,
    is_robot_id,
)
from neuracore.core.utils.training_input_args_validation import (
    _get_data_types_for_algorithms,
    get_algorithm_id,
    get_algorithm_name,
    validate_training_params,
)

# Packaged algorithm Hydra fragments live next to the base training config.
ALGORITHM_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "algorithm"

# Class of each packaged algorithm, keyed by its config file stem.
PACKAGED_ALGORITHM_CLASS_PATHS: dict[str, str] = {
    "act": "neuracore.ml.algorithms.act.act.ACT",
    "cnnmlp": "neuracore.ml.algorithms.cnnmlp.cnnmlp.CNNMLP",
    "diffusion_policy": (
        "neuracore.ml.algorithms.diffusion_policy.diffusion_policy.DiffusionPolicy"
    ),
    "groot": "neuracore.ml.algorithms.groot.groot.Groot",
    "pi0": "neuracore.ml.algorithms.pi0.pi0.Pi0",
    "pi05": "neuracore.ml.algorithms.pi05.pi05.Pi05",
}

# Hydra may resolve ``local_output_dir`` more than once during a single process:
#
# 1. Hydra resolves ``hydra.run.dir`` before entering ``main`` and creates that
#    directory.
# 2. Our training setup later merges the default config again, which can resolve
#    the same interpolation a second time.
#
# Without this cache, an auto-incremented name could move from ``run_1`` to
# ``run_2`` after Hydra creates ``run_1``. The key includes the cache root,
# requested name, and auto-increment flag so unrelated runs do not share names.
_RESOLVED_TRAINING_RUN_NAMES: dict[tuple[str, str, bool], str] = {}


def _resolve_training_run_name(
    training_name: str | None,
    training_name_auto_increment: bool | str = False,
) -> str:
    """Return the final directory name for a training run.

    This resolver is intentionally scoped to the *name* segment only. The full
    path remains regular Hydra/OmegaConf interpolation in ``config.yaml``:
    ``$HOME/.neuracore/training/runs/${_resolved_training_name}``.

    Behavior:
      - Missing names get a random human-readable name from ``names_generator``.
      - Explicit names fail if their run directory already exists.
      - Explicit names auto-increment to ``name_1``, ``name_2``, etc. when
        ``training_name_auto_increment`` is true.
      - Repeated resolutions in one process return the same value, matching the
        directory Hydra already created.
    """
    # Hydra CLI/config values can arrive as real bools or strings.
    auto_increment = (
        str(training_name_auto_increment).lower() == "true"
        if not isinstance(training_name_auto_increment, bool)
        else training_name_auto_increment
    )

    # Treat YAML null, CLI-style "null"/"None", and blank strings as "no name".
    requested_name = "" if training_name is None else str(training_name).strip()

    # Reuse the first answer for this run. This is what prevents duplicate
    # folders when Hydra resolves once before main and our code resolves again.
    cache_key = (str(DEFAULT_CACHE_DIR), requested_name, auto_increment)
    if cache_key in _RESOLVED_TRAINING_RUN_NAMES:
        return _RESOLVED_TRAINING_RUN_NAMES[cache_key]

    # No explicit name: generate a readable random one, converting underscores
    # to hyphens to match the existing run-name style.
    if (
        training_name is None
        or requested_name.lower() in {"none", "null"}
        or requested_name == ""
    ):
        resolved_name = generate_name(style="underscore").replace("_", "-")
        _RESOLVED_TRAINING_RUN_NAMES[cache_key] = resolved_name
        return resolved_name

    run_name = requested_name
    runs_dir = DEFAULT_CACHE_DIR / "runs"
    run_dir = runs_dir / run_name

    # The requested name is free, so keep it exactly as supplied.
    if not run_dir.exists():
        _RESOLVED_TRAINING_RUN_NAMES[cache_key] = run_name
        return run_name

    # The requested name is taken and the caller asked for strict behavior.
    if not auto_increment:
        raise FileExistsError(
            f"A training named {run_name!r} already exists at {run_dir}. "
            "Either use a different training_name, or set "
            "training_name_auto_increment=true to use an incremented name."
        )

    # Find the first available suffix using the conventional ``name_N`` format.
    suffix = 1
    while (runs_dir / f"{run_name}_{suffix}").exists():
        suffix += 1
    resolved_name = f"{run_name}_{suffix}"
    _RESOLVED_TRAINING_RUN_NAMES[cache_key] = resolved_name
    return resolved_name


OmegaConf.register_new_resolver(
    "resolve_training_run_name",
    _resolve_training_run_name,
    use_cache=True,
    replace=True,
)

STRUCTURED_DEFAULTS_CONFIG_NAME = "structured_defaults"

SYNCHRONIZATION_FIELDS_SET_BY_EMBODIMENT = frozenset({"cross_embodiment_union"})

# Pipelines that an override replaces as a whole, so their defaults come from
# TrainingConfig at validation instead of a Hydra node an override merges into.
TRAINING_FIELDS_REPLACED_AS_A_WHOLE = frozenset({"preprocessing"})


def build_structured_defaults() -> dict[str, Any]:
    """Return the training and synchronization defaults as Hydra sections."""
    return {
        "training": TrainingConfig().model_dump(
            mode="json", exclude=set(TRAINING_FIELDS_REPLACED_AS_A_WHOLE)
        ),
        "synchronization": SynchronizationConfig().model_dump(
            mode="json", exclude=set(SYNCHRONIZATION_FIELDS_SET_BY_EMBODIMENT)
        ),
    }


def register_structured_defaults() -> None:
    """Store the training and synchronization defaults in the Hydra ConfigStore."""
    ConfigStore.instance().store(
        name=STRUCTURED_DEFAULTS_CONFIG_NAME, node=build_structured_defaults()
    )


register_structured_defaults()


def load_training_and_synchronization_config(
    cfg: DictConfig,
) -> tuple[TrainingConfig, SynchronizationConfig]:
    """Validate the training and synchronization sections of a config.

    Args:
        cfg: Hydra configuration holding training and synchronization sections.

    Returns:
        The validated training and synchronization settings.
    """
    training_values = OmegaConf.to_container(
        cfg.get("training") or OmegaConf.create({}), resolve=True
    )
    synchronization_values = OmegaConf.to_container(
        cfg.get("synchronization") or OmegaConf.create({}), resolve=True
    )
    return (
        TrainingConfig.model_validate(training_values),
        SynchronizationConfig.model_validate(synchronization_values),
    )


def packaged_algorithm_class_path(cfg: DictConfig) -> str | None:
    """Return the class path of the packaged algorithm the config selects.

    Args:
        cfg: Hydra configuration.

    Returns:
        The class path for the algorithm_name, or None for a custom algorithm
        that algorithm_id selects.
    """
    algorithm_name = cfg.get("algorithm_name")
    if cfg.get("algorithm_id") is not None or not isinstance(algorithm_name, str):
        return None
    return PACKAGED_ALGORITHM_CLASS_PATHS[_packaged_algorithm_stem(algorithm_name)]


def load_algorithm_arguments(cfg: DictConfig) -> dict[str, Any]:
    """Validate the algorithm section of a config as constructor arguments.

    Args:
        cfg: Hydra configuration holding an optional algorithm section.

    Returns:
        The algorithm section as constructor arguments.
    """
    algorithm = cfg.get("algorithm")
    algorithm_values: dict[str, Any] = (
        OmegaConf.to_container(algorithm, resolve=True)
        if isinstance(algorithm, DictConfig)
        else {}
    )
    if "_target_" in algorithm_values:
        raise ValueError(
            "Choose the algorithm with algorithm_name or algorithm_id. The "
            "algorithm section holds only algorithm arguments."
        )
    return AlgorithmConfig.model_validate(algorithm_values).model_dump()


def _reject_top_level_section_keys(cfg: DictConfig) -> None:
    """Raise when a training or synchronization setting sits at the top level."""
    misplaced_section_by_key = {
        **{key: "training" for key in TrainingConfig.model_fields},
        **{key: "synchronization" for key in SynchronizationConfig.model_fields},
    }
    misplaced_keys = sorted(key for key in cfg if key in misplaced_section_by_key)
    if misplaced_keys:
        moves = ", ".join(
            f"{key} to {misplaced_section_by_key[key]}.{key}" for key in misplaced_keys
        )
        raise ValueError(f"Move these settings into their section: {moves}.")
    if "algorithm_params" in cfg:
        raise ValueError(
            "Move algorithm_params into the algorithm section: algorithm.<key>."
        )


def _normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm names for config lookup."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _packaged_algorithm_stem(algorithm_name: str) -> str:
    """Return the config file stem of a packaged algorithm.

    Args:
        algorithm_name: Config file stem or class name of the algorithm.

    Returns:
        The config file stem.
    """
    requested_name = _normalize_algorithm_name(algorithm_name)
    for stem, class_path in PACKAGED_ALGORITHM_CLASS_PATHS.items():
        class_name = class_path.rsplit(".", 1)[-1]
        if requested_name in {
            _normalize_algorithm_name(stem),
            _normalize_algorithm_name(class_name),
        }:
            return stem
    available = ", ".join(sorted(PACKAGED_ALGORITHM_CLASS_PATHS))
    raise ValueError(
        f"Unknown algorithm {algorithm_name!r}. Expected one of: {available}."
    )


def _load_algorithm_config_from_name(algorithm_name: str) -> DictConfig:
    """Load a packaged algorithm config from a file stem or class name."""
    stem = _packaged_algorithm_stem(algorithm_name)
    return OmegaConf.load(ALGORITHM_CONFIG_DIR / f"{stem}.yaml")


def _resolve_algorithm_name_config(cfg: DictConfig) -> DictConfig:
    """Resolve algorithm_name shorthand into a full Hydra algorithm config."""
    if isinstance(cfg.get("algorithm"), str):
        raise ValueError(
            "'algorithm' as a string is not supported. Use 'algorithm_name' "
            "for packaged algorithms or 'algorithm_id' for custom algorithms."
        )

    if not isinstance(cfg.get("algorithm_name"), str):
        return cfg

    if cfg.get("algorithm_id") is not None:
        raise ValueError(
            "Both 'algorithm_name' and 'algorithm_id' are provided. "
            "Please specify only one."
        )

    algorithm_cfg = _load_algorithm_config_from_name(cfg.algorithm_name)
    cfg_with_overrides = copy.deepcopy(cfg)
    if "algorithm" in cfg_with_overrides and cfg_with_overrides.algorithm is None:
        del cfg_with_overrides.algorithm
    return OmegaConf.merge(algorithm_cfg, cfg_with_overrides)


def resolve_to_merged_config(cfg: DictConfig) -> DictConfig:
    """Resolve a potentially incomplete training config to a complete one."""
    # Merge with the default config to ensure all expected keys are present,
    # even if the user provided a custom config that may not include all the
    # default values.
    _reject_top_level_section_keys(cfg)
    default_cfg = OmegaConf.merge(
        OmegaConf.create(build_structured_defaults()),
        OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "config.yaml"),
    )
    cfg = OmegaConf.merge(default_cfg, cfg)

    return _resolve_algorithm_name_config(cfg)


def _is_provided(value: Any) -> bool:
    """Return whether a config value is present and non-empty."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def validate_complete_config(cfg: DictConfig) -> None:
    """Validate mutually exclusive and required top-level training config fields."""
    if cfg.get("algorithm_name") is None and cfg.get("algorithm_id") is None:
        raise ValueError(
            "Neither 'algorithm_name' nor 'algorithm_id' is provided. "
            "Please specify one."
        )

    dataset_id = cfg.get("dataset_id")
    dataset_name = cfg.get("dataset_name")
    if dataset_id is None and dataset_name is None:
        raise ValueError("Either 'dataset_id' or 'dataset_name' must be provided.")
    if dataset_id is not None and dataset_name is not None:
        raise ValueError(
            "Both 'dataset_id' and 'dataset_name' are provided. "
            "Please specify only one."
        )

    input_data_types = cfg.get("input_data_types")
    input_cross_embodiment_description = cfg.get("input_cross_embodiment_description")
    if _is_provided(input_data_types) and _is_provided(
        input_cross_embodiment_description
    ):
        raise ValueError(
            "Both 'input_data_types' and 'input_cross_embodiment_description' "
            "are provided. Please specify only one."
        )
    if not _is_provided(input_data_types) and not _is_provided(
        input_cross_embodiment_description
    ):
        raise ValueError(
            "Neither 'input_data_types' nor 'input_cross_embodiment_description' "
            "is provided. Please specify one."
        )

    output_data_types = cfg.get("output_data_types")
    output_cross_embodiment_description = cfg.get("output_cross_embodiment_description")
    if _is_provided(output_data_types) and _is_provided(
        output_cross_embodiment_description
    ):
        raise ValueError(
            "Both 'output_data_types' and 'output_cross_embodiment_description' "
            "are provided. Please specify only one."
        )
    if not _is_provided(output_data_types) and not _is_provided(
        output_cross_embodiment_description
    ):
        raise ValueError(
            "Neither 'output_data_types' nor 'output_cross_embodiment_description' "
            "is provided. Please specify one."
        )


def _resolve_local_output_dir(cfg: DictConfig) -> None:
    """Resolve the generated output directory without resolving Hydra internals."""
    if "local_output_dir" in cfg:
        cfg.local_output_dir = str(cfg.local_output_dir)


def resolve_user_input_config(cfg: DictConfig) -> DictConfig:
    """Resolve user-facing shorthand while preserving runtime-only settings."""
    cfg = resolve_to_merged_config(cfg)
    _resolve_local_output_dir(cfg)
    return cfg


def _resolve_algorithm_name_and_supported_data_types(
    cfg: DictConfig, algorithms_jsons: list[dict]
) -> tuple[str, set[DataType], set[DataType]]:
    """Resolve algorithm name and supported input and output data types.

    If ``algorithm_id`` is provided (cloud case), use the algorithm ID to get
    the algorithm name and supported data types. If ``algorithm_id`` is not
    provided (local case), use the algorithm class to get the supported data
    types.

    Args:
        cfg: Hydra configuration.
        algorithms_jsons: List of algorithm metadata dictionaries.

    Returns:
        A tuple containing:
          - Algorithm name.
          - Supported input data types.
          - Supported output data types.

    Raises:
        ValueError: If the algorithm does not have supported input or output data types.
    """
    if cfg.algorithm_id is not None:
        if algorithms_jsons:
            algorithm_name = get_algorithm_name(
                algorithm_id=cfg.algorithm_id,
                algorithm_jsons=algorithms_jsons,
            )
            (
                supported_input_data_types,
                supported_output_data_types,
            ) = _get_data_types_for_algorithms(
                algorithm_name=algorithm_name,
                algorithm_jsons=algorithms_jsons,
            )
        else:
            algorithm_json = get_algorithm(cfg.algorithm_id)
            algorithm_name = algorithm_json["name"]
            (
                supported_input_data_types,
                supported_output_data_types,
            ) = _get_data_types_for_algorithms(
                algorithm_name=algorithm_name,
                algorithm_jsons=[algorithm_json],
            )
        return (
            algorithm_name,
            supported_input_data_types,
            supported_output_data_types,
        )

    # Local case: use the algorithm class to get the supported data types.
    algorithm_class_path = packaged_algorithm_class_path(cfg)
    if algorithm_class_path is None:
        raise ValueError("Either 'algorithm_name' or 'algorithm_id' must be provided.")
    algorithm_name = algorithm_class_path.rsplit(".", 1)[-1]
    algorithm_cls = hydra.utils.get_object(algorithm_class_path)
    supported_input_data_types = algorithm_cls.get_supported_input_data_types()
    supported_output_data_types = algorithm_cls.get_supported_output_data_types()
    if (
        supported_input_data_types is not None
        and supported_output_data_types is not None
    ):
        return algorithm_name, supported_input_data_types, supported_output_data_types

    raise ValueError(
        f"Algorithm {algorithm_name} does not have supported input or output "
        "data types, please check the algorithm class."
    )


def build_cross_embodiment_description_from_data_types(
    data_types_cfg: list[str],
    dataset: Dataset,
) -> CrossEmbodimentDescription:
    """Construct a cross-embodiment description from data types and dataset specs."""
    data_types = [DataType(data_type) for data_type in data_types_cfg]
    robot_ids = dataset.robot_ids
    robot_names = (
        dataset.get_robot_names()
        if any(is_robot_id(robot_id) for robot_id in robot_ids)
        else {}
    )

    cross_embodiment_description: CrossEmbodimentDescription = {}
    for robot_id in robot_ids:
        robot_full_embodiment_description = dataset.get_full_embodiment_description(
            robot_id
        )
        robot_name = robot_names[robot_id] if is_robot_id(robot_id) else robot_id
        cross_embodiment_description[robot_name] = {
            data_type: dict(robot_full_embodiment_description.get(data_type, {}))
            for data_type in data_types
        }

    return cross_embodiment_description


def normalize_cross_embodiment_description(
    cross_embodiment_cfg: Any,
) -> CrossEmbodimentDescription:
    """Convert config data type keys and item indexes to runtime types."""
    result: CrossEmbodimentDescription = {}
    for embodiment, embodiment_values in cross_embodiment_cfg.items():
        result[embodiment] = {}
        for data_type, item_names in embodiment_values.items():
            try:
                data_type_enum = (
                    data_type
                    if isinstance(data_type, DataType)
                    else DataType(data_type)
                )
            except ValueError:
                raise ValueError(
                    f"Invalid data type '{data_type}' for robot '{embodiment}'. "
                    f"Expected one of {[item.value for item in DataType]}."
                )
            result[embodiment][data_type_enum] = {
                int(index): name for index, name in item_names.items()
            }
    return result


def _resolve_cross_embodiment_description(
    cross_embodiment_description_cfg: Any,
    data_types_cfg: list[str],
    dataset: Dataset,
    field_name: str,
) -> CrossEmbodimentDescription:
    """Resolve an explicit cross-embodiment config or build one from data types."""
    if _is_provided(cross_embodiment_description_cfg):
        return normalize_cross_embodiment_description(cross_embodiment_description_cfg)
    if not _is_provided(data_types_cfg):
        raise ValueError(f"Either '{field_name}' or data types must be provided.")
    return build_cross_embodiment_description_from_data_types(
        data_types_cfg=data_types_cfg,
        dataset=dataset,
    )


def _primitive_attr(value: Any, attr: str, fallback: str | None) -> str | None:
    """Return a primitive string attribute value, ignoring mock attributes."""
    resolved = getattr(value, attr, None)
    return resolved if isinstance(resolved, str) else fallback


def resolve_to_complete_config(
    cfg: DictConfig, dataset: Dataset | None = None
) -> DictConfig:
    """Resolve selectors and data-type shorthand to a complete training config."""
    cfg = resolve_to_merged_config(cfg)
    _resolve_local_output_dir(cfg)

    # Populate dataset name/id from whichever selector the caller supplied.
    if dataset is not None:
        cfg.dataset_id = _primitive_attr(dataset, "id", cfg.get("dataset_id"))
        cfg.dataset_name = _primitive_attr(dataset, "name", cfg.get("dataset_name"))
    elif cfg.dataset_name is not None:
        dataset = nc.get_dataset(name=cfg.dataset_name)
        cfg.dataset_id = _primitive_attr(dataset, "id", cfg.get("dataset_id"))
    else:
        dataset = nc.get_dataset(id=cfg.dataset_id)
        cfg.dataset_name = _primitive_attr(dataset, "name", cfg.get("dataset_name"))

    # Populate algorithm name/id from whichever selector the caller supplied.
    if cfg.algorithm_id is not None:
        algorithm_name, supported_input_data_types, supported_output_data_types = (
            _resolve_algorithm_name_and_supported_data_types(cfg, [])
        )
        cfg.algorithm_name = algorithm_name
    elif cfg.get("algorithm_name") is not None:
        algorithm_name, supported_input_data_types, supported_output_data_types = (
            _resolve_algorithm_name_and_supported_data_types(cfg, [])
        )
    else:
        algorithms_jsons = _get_algorithms()
        algorithm_id = get_algorithm_id(cfg.algorithm_name, algorithms_jsons)
        if algorithm_id is None:
            raise ValueError(
                f"Algorithm name {cfg.algorithm_name} not found in available "
                "algorithms. Please check the training job requirements."
            )
        cfg.algorithm_id = algorithm_id
        algorithm_name, supported_input_data_types, supported_output_data_types = (
            _resolve_algorithm_name_and_supported_data_types(cfg, algorithms_jsons)
        )

    cfg.input_cross_embodiment_description = _resolve_cross_embodiment_description(
        cross_embodiment_description_cfg=cfg.input_cross_embodiment_description,
        data_types_cfg=cfg.input_data_types,
        dataset=dataset,
        field_name="input_cross_embodiment_description",
    )
    cfg.output_cross_embodiment_description = _resolve_cross_embodiment_description(
        cross_embodiment_description_cfg=cfg.output_cross_embodiment_description,
        data_types_cfg=cfg.output_data_types,
        dataset=dataset,
        field_name="output_cross_embodiment_description",
    )

    input_cross_embodiment_description = (
        convert_cross_embodiment_description_names_to_ids(
            cfg.input_cross_embodiment_description
        )
    )
    output_cross_embodiment_description = (
        convert_cross_embodiment_description_names_to_ids(
            cfg.output_cross_embodiment_description
        )
    )
    cfg.input_cross_embodiment_description = input_cross_embodiment_description
    cfg.output_cross_embodiment_description = output_cross_embodiment_description

    validate_training_params(
        dataset,
        dataset_name=cfg.dataset_name if cfg.dataset_name is not None else "",
        algorithm_name=algorithm_name,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        supported_input_data_types=supported_input_data_types,
        supported_output_data_types=supported_output_data_types,
    )

    return cfg
