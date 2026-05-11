"""Utilities for normalizing Hydra training configuration."""

import concurrent
import copy
import re
from pathlib import Path
from typing import Any

import hydra
from neuracore_types import CrossEmbodimentDescription, DataType
from omegaconf import DictConfig, OmegaConf

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.robot import get_robot_name_from_id
from neuracore.core.utils.http_session import get_session
from neuracore.core.utils.robot_data_spec_utils import (
    convert_cross_embodiment_description_names_to_ids,
    convert_omegaconf_to_cross_embodiment_description,
)
from neuracore.core.utils.training_input_args_validation import (
    _get_data_types_for_algorithms,
    get_algorithm_id,
    get_algorithm_name,
    validate_training_params,
)

ALGORITHM_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "algorithm"


def _normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm names for config lookup."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _load_algorithm_config_from_name(algorithm_name: str) -> DictConfig:
    """Load a packaged algorithm config from a file stem or target class name."""
    requested_name = _normalize_algorithm_name(algorithm_name)
    available_names: list[str] = []

    for config_path in sorted(ALGORITHM_CONFIG_DIR.glob("*.yaml")):
        algorithm_cfg = OmegaConf.load(config_path)
        cfg_algorithm = algorithm_cfg.get("algorithm")
        if cfg_algorithm is None or "_target_" not in cfg_algorithm:
            continue

        target_name = str(cfg_algorithm._target_).rsplit(".", 1)[-1]
        candidate_names = {
            _normalize_algorithm_name(config_path.stem),
            _normalize_algorithm_name(target_name),
        }
        available_names.extend(sorted(candidate_names))

        if requested_name in candidate_names:
            return algorithm_cfg

    available = ", ".join(sorted(set(available_names)))
    raise ValueError(
        f"Unknown algorithm {algorithm_name!r}. Expected one of: {available}."
    )


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
    cfg_without_algorithm_name = copy.deepcopy(cfg)
    del cfg_without_algorithm_name.algorithm_name
    if (
        "algorithm" in cfg_without_algorithm_name
        and cfg_without_algorithm_name.algorithm is None
    ):
        del cfg_without_algorithm_name.algorithm
    return OmegaConf.merge(algorithm_cfg, cfg_without_algorithm_name)


def resolve_to_merged_config(cfg: DictConfig) -> DictConfig:
    """Resolve a potentially incomplete training config to a complete one."""
    # Merge with the default config to ensure all expected keys are present,
    # even if the user provided a custom config that may not include all the
    # default values.
    default_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "config" / "config.yaml"
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
    algorithm = cfg.get("algorithm")
    algorithm_id = cfg.get("algorithm_id")
    if algorithm is not None and algorithm_id is not None:
        raise ValueError(
            "Both 'algorithm' and 'algorithm_id' are provided. "
            "Please specify only one."
        )
    if algorithm is None and algorithm_id is None:
        raise ValueError(
            "Neither 'algorithm' nor 'algorithm_id' is provided. " "Please specify one."
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


def _get_algorithms() -> list[dict]:
    """Retrieve all available algorithms from the API.

    Fetches both organization-specific and shared algorithms concurrently.

    Returns:
        list[dict]: List of algorithm dictionaries containing algorithm metadata

    Raises:
        requests.exceptions.HTTPError: If the API request fails
        requests.exceptions.RequestException: If there is a network problem
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        org_alg_req = executor.submit(
            get_session().get,
            f"{API_URL}/org/{org_id}/algorithms",
            headers=auth.get_headers(),
            params={"shared": False},
        )
        shared_alg_req = executor.submit(
            get_session().get,
            f"{API_URL}/org/{org_id}/algorithms",
            headers=auth.get_headers(),
            params={"shared": True},
        )
        org_alg, shared_alg = org_alg_req.result(), shared_alg_req.result()
    org_alg.raise_for_status()
    shared_alg.raise_for_status()
    return org_alg.json() + shared_alg.json()


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
        return (
            algorithm_name,
            supported_input_data_types,
            supported_output_data_types,
        )

    # Local case: use the algorithm class to get the supported data types.
    algorithm_name = cfg.algorithm._target_.rsplit(".", 1)[-1]
    algorithm_cls = hydra.utils.get_object(cfg.algorithm._target_)
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

    cross_embodiment_description: CrossEmbodimentDescription = {}
    for robot_id in dataset.robot_ids:
        robot_full_spec = dataset.get_full_embodiment_description(robot_id)
        robot_name = get_robot_name_from_id(robot_id)
        cross_embodiment_description[robot_name] = {
            data_type: dict(robot_full_spec.get(data_type, {}))
            for data_type in data_types
        }

    return cross_embodiment_description


def _resolve_cross_embodiment_description(
    cross_embodiment_description_cfg: DictConfig | None,
    data_types_cfg: list[str],
    dataset: Dataset,
    field_name: str,
) -> CrossEmbodimentDescription:
    """Resolve an explicit cross-embodiment config or data-type shorthand."""
    if cross_embodiment_description_cfg is not None and len(
        cross_embodiment_description_cfg
    ):
        if not isinstance(cross_embodiment_description_cfg, DictConfig):
            raise ValueError(
                f"'{field_name}' must either be None or a dictionary mapping robot "
                "names to dictionaries of data types to lists of data names."
            )
        return convert_omegaconf_to_cross_embodiment_description(
            cross_embodiment_description_cfg
        )

    if data_types_cfg is None:
        raise ValueError(
            f"Either '{field_name}' or the corresponding data_types "
            "configuration must be provided."
        )

    cross_embodiment_description: CrossEmbodimentDescription = {}
    data_types = [DataType(data_type) for data_type in data_types_cfg]
    for robot_id in dataset.robot_ids:
        robot_full_spec = dataset.get_full_embodiment_description(robot_id)
        cross_embodiment_description[robot_id] = {
            data_type: dict(robot_full_spec.get(data_type, {}))
            for data_type in data_types
        }
    return cross_embodiment_description


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

    algorithms_jsons = _get_algorithms()
    # Populate algorithm name/id from whichever selector the caller supplied.
    if cfg.algorithm_id is not None:
        algorithm_name = get_algorithm_name(
            algorithm_id=cfg.algorithm_id,
            algorithm_jsons=algorithms_jsons,
        )
        cfg.algorithm_name = algorithm_name
    elif cfg.get("algorithm") is not None:
        pass
    else:
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

    # handle inputs
    if _is_provided(cfg.input_data_types):
        cfg.input_cross_embodiment_description = (
            build_cross_embodiment_description_from_data_types(
                data_types_cfg=cfg.input_data_types,
                dataset=dataset,
            )
        )

    if _is_provided(cfg.output_data_types):
        # Calculate output cross-embodiment description from output data types
        cfg.output_cross_embodiment_description = (
            build_cross_embodiment_description_from_data_types(
                data_types_cfg=cfg.output_data_types,
                dataset=dataset,
            )
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

    # TODO: validate resolved descriptions against specs and algorithm requirements.
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
