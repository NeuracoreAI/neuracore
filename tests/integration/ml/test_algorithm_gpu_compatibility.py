"""Integration tests for algorithm GPU compatibility metadata."""

import pytest
from neuracore_types import GPUType

import neuracore as nc
from neuracore.api.training import _get_algorithms, _validate_gpu_to_algorithm


def test_algorithm_gpu_compatibility_accepts_and_rejects() -> None:
    """Live algorithm metadata accepts a supported GPU and rejects a mismatch."""
    nc.login()
    algorithms = _get_algorithms()
    all_gpus = set(GPUType)

    algorithm = next(
        (
            candidate
            for candidate in algorithms
            if candidate.get("supported_gpus")
            and all_gpus - {GPUType(value) for value in candidate["supported_gpus"]}
        ),
        None,
    )
    assert (
        algorithm is not None
    ), "Expected at least one algorithm that does not support every GPU type"

    algorithm_name = algorithm["name"]
    supported_gpus = {GPUType(value) for value in algorithm["supported_gpus"]}
    supported_gpu = next(iter(supported_gpus))
    unsupported_gpu = next(iter(all_gpus - supported_gpus))

    _validate_gpu_to_algorithm(
        supported_gpu,
        algorithm_name,
        algorithms,
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"GPU {unsupported_gpu.value} is not supported by algorithm "
            rf"{algorithm_name}\."
        ),
    ):
        _validate_gpu_to_algorithm(
            unsupported_gpu,
            algorithm_name,
            algorithms,
        )
