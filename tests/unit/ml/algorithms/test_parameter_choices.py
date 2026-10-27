"""Fixed-choice algorithm parameters are string enums with the old defaults.

The backend reads these enums from each algorithm's source to offer the
parameters as dropdowns, and training configs pass their values as plain
strings, so the enums must stay string enums with string values.
"""

import importlib
import inspect
from enum import Enum

import pytest

from neuracore.ml.utils.algorithm_loader import install_packaged_algorithm_requirements

ALGORITHMS_PACKAGE = "neuracore.ml.algorithms"
GEMMA_VARIANTS = ["gemma_300m", "gemma_2b", "gemma_tiny"]
COMPILE_MODES = [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

FIXED_CHOICE_PARAMETERS = [
    pytest.param(
        "diffusion_policy.diffusion_policy",
        "DiffusionPolicy",
        "process_type",
        "ProcessType",
        "diffusion",
        ["diffusion", "flow_matching"],
        id="diffusion_policy-process_type",
    ),
    pytest.param(
        "diffusion_policy.diffusion_policy",
        "DiffusionPolicy",
        "noise_scheduler_type",
        "NoiseSchedulerType",
        "DDPM",
        ["DDPM", "DDIM"],
        id="diffusion_policy-noise_scheduler_type",
    ),
    pytest.param(
        "diffusion_policy.diffusion_policy",
        "DiffusionPolicy",
        "prediction_type",
        "PredictionType",
        "epsilon",
        ["epsilon", "sample"],
        id="diffusion_policy-prediction_type",
    ),
    pytest.param(
        "diffusion_policy.diffusion_policy",
        "DiffusionPolicy",
        "beta_schedule",
        "BetaSchedule",
        "squaredcos_cap_v2",
        ["linear", "scaled_linear", "squaredcos_cap_v2"],
        id="diffusion_policy-beta_schedule",
    ),
    pytest.param(
        "diffusion_policy.diffusion_policy",
        "DiffusionPolicy",
        "lr_scheduler_type",
        "LRSchedulerType",
        "cosine",
        [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        id="diffusion_policy-lr_scheduler_type",
    ),
    pytest.param(
        "cnnmlp.cnnmlp",
        "CNNMLP",
        "image_backbone",
        "ImageBackbone",
        "resnet18",
        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        id="cnnmlp-image_backbone",
    ),
    *[
        pytest.param(
            f"{name}.{name}",
            class_name,
            parameter,
            enum_name,
            default,
            choices,
            id=f"{name}-{parameter}",
        )
        for name, class_name in [("pi0", "Pi0"), ("pi05", "Pi05")]
        for parameter, enum_name, default, choices in [
            ("paligemma_variant", "GemmaVariant", "gemma_2b", GEMMA_VARIANTS),
            ("action_expert_variant", "GemmaVariant", "gemma_300m", GEMMA_VARIANTS),
            ("compile_mode", "CompileMode", "max-autotune", COMPILE_MODES),
        ]
    ],
    pytest.param(
        "groot.groot",
        "Groot",
        "torch_compile_mode",
        "CompileMode",
        "default",
        COMPILE_MODES,
        id="groot-torch_compile_mode",
    ),
]


@pytest.mark.parametrize(
    ("module_name", "class_name", "parameter", "enum_name", "default", "choices"),
    FIXED_CHOICE_PARAMETERS,
)
def test_fixed_choice_parameter_is_a_string_enum(
    module_name: str,
    class_name: str,
    parameter: str,
    enum_name: str,
    default: str,
    choices: list[str],
) -> None:
    target = f"{ALGORITHMS_PACKAGE}.{module_name}.{class_name}"
    install_packaged_algorithm_requirements(target)
    module = importlib.import_module(f"{ALGORITHMS_PACKAGE}.{module_name}")
    enum_cls = getattr(module, enum_name)
    signature_param = inspect.signature(
        getattr(module, class_name).__init__
    ).parameters[parameter]

    assert issubclass(enum_cls, str)
    assert issubclass(enum_cls, Enum)
    assert [member.value for member in enum_cls] == choices
    assert signature_param.default is enum_cls(default)
    assert signature_param.default == default
    # Pi0, Pi05 and Groot postpone annotation evaluation, so theirs are strings.
    assert signature_param.annotation in (enum_cls, enum_name)


@pytest.mark.parametrize(
    ("module_name", "enum_name"),
    [
        ("diffusion_policy.diffusion_policy", "ProcessType"),
        ("pi0.pi0", "GemmaVariant"),
    ],
)
def test_plain_strings_act_like_enum_members(module_name: str, enum_name: str):
    """Training configs carry plain strings; they must compare and hash like
    the members the defaults use."""
    install_packaged_algorithm_requirements(
        f"{ALGORITHMS_PACKAGE}.{module_name}.Unused"
    )
    enum_cls = getattr(
        importlib.import_module(f"{ALGORITHMS_PACKAGE}.{module_name}"), enum_name
    )

    for member in enum_cls:
        assert member == member.value
        assert enum_cls(member.value) is member
        assert {member.value: True}[member]
