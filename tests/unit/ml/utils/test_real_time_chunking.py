"""Tests for the real-time chunking guided sampler."""

import math

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn

from neuracore.ml.utils.real_time_chunking import (
    DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT,
    DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT,
    RTCConfig,
    align_previous_chunk,
    as_ddim_scheduler,
    build_global_conditioning,
    guided_diffusion_sample,
    guided_flow_matching_sample,
    missing_rtc_attributes,
    rtc_soft_mask,
    supports_real_time_chunking,
)

H = 12
ACTION_DIM = 2
BATCH = 64
DELAY = 3
EXECUTION_HORIZON = 4
STEPS = 10

# Data prior used by the analytic denoisers below.
DATA_MEAN = 1.0
DATA_STD = 0.7


def _scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
    )


class _GaussianDiffusionDenoiser(nn.Module):
    """Exact optimal epsilon-denoiser for ``x0 ~ N(DATA_MEAN, DATA_STD^2)``.

    A real trained policy is not usable here: an untrained network has an
    arbitrary Jacobian, and one overfit to a handful of samples has a
    *vanishing* clean-prediction Jacobian, which makes guidance provably
    inert. This posterior is the smallest denoiser that is simultaneously
    analytic and non-degenerate.
    """

    def __init__(self, scheduler: DDIMScheduler) -> None:
        super().__init__()
        self.noise_scheduler = scheduler
        self.process_type = "diffusion"
        self.prediction_type = "epsilon"
        self.num_inference_steps = STEPS
        self.clip_sample = False
        self.clip_sample_range = 1.0
        self.unet = self._forward

    def _forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(x.device)[timestep[0].long()]
        sqrt_alpha_bar = alpha_bar.sqrt()
        posterior = alpha_bar * DATA_STD**2 + (1.0 - alpha_bar)
        clean = DATA_STD**2 * sqrt_alpha_bar * x + (1.0 - alpha_bar) * DATA_MEAN
        clean = clean / posterior
        return (x - sqrt_alpha_bar * clean) / (1.0 - alpha_bar).sqrt()


class _GaussianFlowDenoiser(nn.Module):
    """Exact optimal velocity field for ``x0 ~ N(DATA_MEAN, DATA_STD^2)``."""

    def __init__(self) -> None:
        super().__init__()
        self.noise_scheduler = None
        self.process_type = "flow_matching"
        self.prediction_type = "epsilon"
        self.num_inference_steps = STEPS
        self.clip_sample = False
        self.clip_sample_range = 1.0
        self.unet = self._forward

    def _forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tau = float(timestep[0])
        posterior = tau**2 * DATA_STD**2 + (1.0 - tau) ** 2
        clean = (DATA_STD**2 * tau * x + (1.0 - tau) ** 2 * DATA_MEAN) / posterior
        return (clean - x) / max(1.0 - tau, 1e-6)


def _sample(model: nn.Module, target: torch.Tensor, mask: torch.Tensor, seed: int = 4):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((BATCH, H, ACTION_DIM), generator=generator)
    if model.process_type == "flow_matching":
        return guided_flow_matching_sample(
            model, noise, target, mask, num_inference_steps=STEPS
        )
    return guided_diffusion_sample(
        model, noise, target, mask, num_inference_steps=STEPS
    )


@pytest.fixture(params=["diffusion", "flow_matching"])
def denoiser(request) -> nn.Module:
    if request.param == "diffusion":
        return _GaussianDiffusionDenoiser(_scheduler())
    return _GaussianFlowDenoiser()


# --------------------------------------------------------------------------- #
# Soft mask
# --------------------------------------------------------------------------- #


def test_soft_mask_regions():
    mask = rtc_soft_mask(H, DELAY, EXECUTION_HORIZON)
    assert mask.shape == (H,)
    assert torch.all(mask[:DELAY] == 1.0), "frozen prefix must be fully weighted"
    assert torch.all(mask[H - EXECUTION_HORIZON :] == 0.0), "tail must be free"
    assert torch.all(mask[1:] <= mask[:-1] + 1e-6), "mask must be non-increasing"
    assert torch.all((mask >= 0.0) & (mask <= 1.0))


def test_soft_mask_boundary_is_continuous():
    """The exponential schedule must meet both flat regions without a step."""
    mask = rtc_soft_mask(H, DELAY, EXECUTION_HORIZON)
    # c_i == 1 at i == d - 1, so the first soft entry sits just below 1.
    assert mask[DELAY] < 1.0
    assert mask[H - EXECUTION_HORIZON - 1] < 0.1


@pytest.mark.parametrize(
    "schedule,expected",
    [("ones", 1.0), ("zeros", 0.0)],
)
def test_soft_mask_degenerate_schedules(schedule, expected):
    mask = rtc_soft_mask(H, DELAY, EXECUTION_HORIZON, schedule=schedule)
    overlap = mask[DELAY : H - EXECUTION_HORIZON]
    assert torch.all(overlap == expected)


def test_soft_mask_exp_matches_formula():
    mask = rtc_soft_mask(H, DELAY, EXECUTION_HORIZON, schedule="exp")
    index = DELAY + 1
    c = (H - EXECUTION_HORIZON - index) / (H - EXECUTION_HORIZON - DELAY + 1)
    expected = c * (math.exp(c) - 1.0) / (math.e - 1.0)
    assert mask[index].item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "horizon,delay,execution_horizon",
    [
        (H, H - EXECUTION_HORIZON + 1, EXECUTION_HORIZON),  # d > H - s
        (H, -1, EXECUTION_HORIZON),  # negative delay
        (H, 0, 0),  # zero execution horizon
        (H, 0, H + 1),  # execution horizon beyond the chunk
    ],
)
def test_soft_mask_rejects_invalid_horizons(horizon, delay, execution_horizon):
    with pytest.raises(ValueError):
        rtc_soft_mask(horizon, delay, execution_horizon)


def test_soft_mask_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="prefix_attention_schedule"):
        rtc_soft_mask(H, DELAY, EXECUTION_HORIZON, schedule="sigmoid")


# --------------------------------------------------------------------------- #
# Chunk alignment
# --------------------------------------------------------------------------- #


def test_align_previous_chunk_shifts_and_pads():
    chunk = torch.arange(H, dtype=torch.float32).view(1, H, 1).repeat(1, 1, ACTION_DIM)
    aligned = align_previous_chunk(chunk, EXECUTION_HORIZON, H)

    assert aligned.shape == (1, H, ACTION_DIM)
    overlap = H - EXECUTION_HORIZON
    expected = torch.arange(EXECUTION_HORIZON, H, dtype=torch.float32)
    assert torch.equal(aligned[0, :overlap, 0], expected)
    assert torch.all(aligned[0, overlap:, :] == 0.0)


def test_align_previous_chunk_padding_lands_where_mask_is_zero():
    """The zero padding must never be read as a guidance target."""
    chunk = torch.rand(1, H, ACTION_DIM)
    aligned = align_previous_chunk(chunk, EXECUTION_HORIZON, H)
    mask = rtc_soft_mask(H, DELAY, EXECUTION_HORIZON)
    padded = aligned[0].abs().sum(dim=-1) == 0.0
    assert torch.all(mask[padded] == 0.0)


def test_align_previous_chunk_rejects_negative_consumption():
    with pytest.raises(ValueError):
        align_previous_chunk(torch.zeros(1, H, ACTION_DIM), -1, H)


# --------------------------------------------------------------------------- #
# Guided sampling
# --------------------------------------------------------------------------- #


def test_zero_mask_reproduces_unguided_sampling(denoiser):
    """RTC must be inert when nothing is masked.

    This is the regression guard for every caller that does not opt in.
    """
    target = torch.rand(BATCH, H, ACTION_DIM)
    guided = _sample(denoiser, target, torch.zeros(H))

    generator = torch.Generator().manual_seed(4)
    noise = torch.randn((BATCH, H, ACTION_DIM), generator=generator)
    with torch.no_grad():
        if denoiser.process_type == "flow_matching":
            dt = 1.0 / STEPS
            unguided = noise
            for step in range(STEPS):
                timestep = torch.full(unguided.shape[:1], step * dt)
                unguided = unguided + dt * denoiser.unet(unguided, timestep)
        else:
            scheduler = denoiser.noise_scheduler
            scheduler.set_timesteps(STEPS)
            unguided = noise
            for timestep in scheduler.timesteps:
                model_output = denoiser.unet(
                    unguided,
                    torch.full(unguided.shape[:1], timestep, dtype=torch.long),
                )
                unguided = scheduler.step(model_output, timestep, unguided).prev_sample

    assert torch.allclose(guided, unguided, atol=1e-4)


def test_guidance_pulls_frozen_prefix_toward_previous_chunk(denoiser):
    target = DATA_MEAN + DATA_STD * torch.randn(
        BATCH, H, ACTION_DIM, generator=torch.Generator().manual_seed(0)
    )
    guided = _sample(denoiser, target, rtc_soft_mask(H, DELAY, EXECUTION_HORIZON))
    unguided = _sample(denoiser, target, torch.zeros(H))

    def error(chunk, low, high):
        return (chunk[:, low:high] - target[:, low:high]).abs().mean().item()

    prefix_guided = error(guided, 0, DELAY)
    prefix_unguided = error(unguided, 0, DELAY)
    assert (
        prefix_guided < 0.4 * prefix_unguided
    ), f"frozen prefix barely moved: {prefix_unguided:.3f} -> {prefix_guided:.3f}"


def test_guidance_is_graded_across_the_chunk(denoiser):
    """Prefix is pinned hardest, overlap partially, tail not at all."""
    target = DATA_MEAN + DATA_STD * torch.randn(
        BATCH, H, ACTION_DIM, generator=torch.Generator().manual_seed(0)
    )
    guided = _sample(denoiser, target, rtc_soft_mask(H, DELAY, EXECUTION_HORIZON))
    unguided = _sample(denoiser, target, torch.zeros(H))

    def error(chunk, low, high):
        return (chunk[:, low:high] - target[:, low:high]).abs().mean().item()

    prefix = error(guided, 0, DELAY)
    overlap = error(guided, DELAY, H - EXECUTION_HORIZON)
    tail_guided = error(guided, H - EXECUTION_HORIZON, H)
    tail_unguided = error(unguided, H - EXECUTION_HORIZON, H)

    assert prefix < overlap, "overlap should be pulled less than the frozen prefix"
    assert overlap < error(unguided, DELAY, H - EXECUTION_HORIZON)
    assert tail_guided == pytest.approx(
        tail_unguided, rel=1e-3
    ), "the freely generated tail must be untouched"


def test_guidance_preserves_tail_diversity(denoiser):
    """Guidance must not collapse the free region onto the previous chunk."""
    target = DATA_MEAN + DATA_STD * torch.randn(
        BATCH, H, ACTION_DIM, generator=torch.Generator().manual_seed(0)
    )
    guided = _sample(denoiser, target, rtc_soft_mask(H, DELAY, EXECUTION_HORIZON))
    tail_std = guided[:, H - EXECUTION_HORIZON :].std().item()
    assert tail_std > 0.5 * DATA_STD, f"tail collapsed (std={tail_std:.3f})"


def test_guided_output_is_finite(denoiser):
    target = torch.rand(BATCH, H, ACTION_DIM) * 10.0  # far outside the prior
    guided = _sample(denoiser, target, rtc_soft_mask(H, DELAY, EXECUTION_HORIZON))
    assert torch.isfinite(guided).all()


def test_guided_diffusion_rejects_unsupported_prediction_type():
    model = _GaussianDiffusionDenoiser(_scheduler())
    model.prediction_type = "v_prediction"
    with pytest.raises(ValueError, match="prediction_type"):
        guided_diffusion_sample(
            model,
            torch.randn(1, H, ACTION_DIM),
            torch.zeros(1, H, ACTION_DIM),
            rtc_soft_mask(H, DELAY, EXECUTION_HORIZON),
        )


def test_guided_diffusion_requires_a_scheduler():
    model = _GaussianFlowDenoiser()
    model.process_type = "diffusion"
    with pytest.raises(ValueError, match="flow matching"):
        guided_diffusion_sample(
            model,
            torch.randn(1, H, ACTION_DIM),
            torch.zeros(1, H, ACTION_DIM),
            rtc_soft_mask(H, DELAY, EXECUTION_HORIZON),
        )


# --------------------------------------------------------------------------- #
# Scheduler swap
# --------------------------------------------------------------------------- #


def test_as_ddim_scheduler_preserves_the_noise_schedule():
    ddpm = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    ddim = as_ddim_scheduler(ddpm)

    assert isinstance(ddim, DDIMScheduler)
    assert torch.allclose(ddim.alphas_cumprod, ddpm.alphas_cumprod)
    assert ddim.config.prediction_type == ddpm.config.prediction_type
    assert ddim.config.steps_offset == 0


def test_as_ddim_scheduler_is_identity_for_ddim():
    ddim = _scheduler()
    assert as_ddim_scheduler(ddim) is ddim


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_resolve_guidance_weight_defaults_per_process_type():
    """Diffusion needs a far larger beta than flow matching; see the module docs."""
    config = RTCConfig(inference_delay=1, execution_horizon=EXECUTION_HORIZON)

    assert config.max_guidance_weight is None
    assert config.resolve_guidance_weight("flow_matching") == (
        DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT
    )
    assert config.resolve_guidance_weight("diffusion") == (
        DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT
    )
    assert DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT > DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT


def test_resolve_guidance_weight_honours_an_explicit_value():
    config = RTCConfig(
        inference_delay=1, execution_horizon=EXECUTION_HORIZON, max_guidance_weight=2.5
    )
    assert config.resolve_guidance_weight("diffusion") == 2.5
    assert config.resolve_guidance_weight("flow_matching") == 2.5


def test_with_inference_delay_preserves_an_unset_guidance_weight():
    config = RTCConfig(inference_delay=1, execution_horizon=EXECUTION_HORIZON)
    assert config.with_inference_delay(4).max_guidance_weight is None


def test_with_horizons_updates_both_fields():
    config = RTCConfig(
        inference_delay=1,
        execution_horizon=EXECUTION_HORIZON,
        max_guidance_weight=3.0,
        num_inference_steps=7,
    )
    updated = config.with_horizons(4, 9)
    assert updated.inference_delay == 4
    assert updated.execution_horizon == 9
    assert updated.max_guidance_weight == 3.0
    assert updated.num_inference_steps == 7
    assert config.execution_horizon == EXECUTION_HORIZON


@pytest.mark.parametrize(
    "horizon,s_min,expected",
    [
        (40, 20, 20),
        (16, 8, 8),
        (16, 3, 8),
        (40, 30, 10),
    ],
)
def test_max_feasible_inference_delay(horizon, s_min, expected):
    from neuracore.ml.utils.real_time_chunking import max_feasible_inference_delay

    assert max_feasible_inference_delay(horizon, s_min) == expected
    # Feasibility: s = max(s_min, d) and s + d <= H.
    d = expected
    s = max(s_min, d)
    assert s + d <= horizon
    if expected + 1 <= horizon:
        s_next = max(s_min, expected + 1)
        assert s_next + (expected + 1) > horizon


def test_align_and_mask_agree_when_s_equals_consumed():
    """P0 regression: padding must only sit where the mask is zero."""
    from neuracore.ml.utils.real_time_chunking import (
        align_previous_chunk,
        rtc_soft_mask,
    )

    for consumed in (EXECUTION_HORIZON, EXECUTION_HORIZON + 3, H - DELAY - 1):
        chunk = torch.rand(1, H, ACTION_DIM)
        aligned = align_previous_chunk(chunk, consumed, H)
        mask = rtc_soft_mask(H, min(DELAY, H - consumed), consumed)
        padded = aligned[0].abs().sum(dim=-1) == 0.0
        assert torch.all(mask[padded] == 0.0), consumed


def test_large_beta_saturates_rather_than_destabilising():
    """zeta is hard-capped, so raising beta past saturation must be inert."""
    model = _GaussianDiffusionDenoiser(_scheduler())
    target = DATA_MEAN + DATA_STD * torch.randn(
        BATCH, H, ACTION_DIM, generator=torch.Generator().manual_seed(0)
    )
    mask = rtc_soft_mask(H, DELAY, EXECUTION_HORIZON)

    def run(beta):
        generator = torch.Generator().manual_seed(4)
        noise = torch.randn((BATCH, H, ACTION_DIM), generator=generator)
        return guided_diffusion_sample(
            model,
            noise,
            target,
            mask,
            max_guidance_weight=beta,
            num_inference_steps=STEPS,
        )

    saturated = run(DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT)
    huge = run(DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT * 100)
    assert torch.allclose(saturated, huge, atol=1e-5)
    assert torch.isfinite(huge).all()


# --------------------------------------------------------------------------- #
# Model-archive compatibility
# --------------------------------------------------------------------------- #


class _LegacyDenoiser(_GaussianDiffusionDenoiser):
    """A model as loaded from an archive predating ``_build_global_cond``.

    Archives bundle their algorithm source and ``AlgorithmLoader`` imports it as
    a fresh module, so the loaded class is neither the installed one nor a
    subclass of it.
    """

    def __init__(self, scheduler, global_cond):
        super().__init__(scheduler)
        self.output_prediction_horizon = H
        self.max_output_size = ACTION_DIM
        self._predict_action = lambda batch, horizon: None
        self.input_data_types = set()
        self.global_cond_dim = global_cond.shape[-1]
        self._global_cond = global_cond
        self.action_normalizer = object()

    def _combine_proprio(self, batch):
        return self._global_cond

    def _prepare_global_conditioning(self, joint_states, rgb, mask):
        raise AssertionError("not reached for a proprio-only model")


def test_supports_real_time_chunking_accepts_a_legacy_archive_model():
    """isinstance would reject this model; the structural check must not."""
    from neuracore.ml.algorithms.diffusion_policy.diffusion_policy import (
        DiffusionPolicy,
    )

    model = _LegacyDenoiser(_scheduler(), torch.zeros(1, 4))

    assert not isinstance(model, DiffusionPolicy)
    assert not hasattr(model, "_build_global_cond")
    assert missing_rtc_attributes(model) == []
    assert supports_real_time_chunking(model) is True


def test_missing_rtc_attributes_names_what_is_absent():
    class NotAPolicy:
        pass

    missing = missing_rtc_attributes(NotAPolicy())
    assert "unet" in missing
    assert "_build_global_cond" in missing
    assert supports_real_time_chunking(NotAPolicy()) is False


def test_build_global_conditioning_falls_back_for_legacy_models():
    expected = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    model = _LegacyDenoiser(_scheduler(), expected)

    assert torch.equal(build_global_conditioning(model, _FakeBatch()), expected)


def test_build_global_conditioning_prefers_the_modern_hook():
    model = _LegacyDenoiser(_scheduler(), torch.zeros(1, 4))
    sentinel = torch.ones(1, 4)
    model._build_global_cond = lambda batch: sentinel

    assert torch.equal(build_global_conditioning(model, _FakeBatch()), sentinel)


class _FakeBatch:
    """Minimal stand-in for BatchedInferenceInputs."""

    inputs: dict = {}
    inputs_mask: dict = {}

    def __len__(self) -> int:
        return 1


def test_config_with_inference_delay_copies_every_other_field():
    config = RTCConfig(
        inference_delay=1,
        execution_horizon=EXECUTION_HORIZON,
        max_guidance_weight=3.0,
        prefix_attention_schedule="linear",
        force_ddim=False,
        num_inference_steps=7,
        guidance_start_step=2,
    )
    updated = config.with_inference_delay(5)

    assert updated.inference_delay == 5
    assert config.inference_delay == 1, "the original must not be mutated"
    for field in (
        "execution_horizon",
        "max_guidance_weight",
        "prefix_attention_schedule",
        "force_ddim",
        "num_inference_steps",
        "guidance_start_step",
    ):
        assert getattr(updated, field) == getattr(config, field)
