"""Real-Time Chunking (RTC) guided sampling for diffusion policies.

Implements the inference-time method from "Real-Time Execution of Action
Chunking Flow Policies" (Black, Galliker, Levine; arXiv:2506.07339).

An action-chunking policy predicts ``H`` actions from one observation but only
executes ``s`` of them before replanning. If inference takes ``d`` controller
ticks, the first ``d`` actions of the new chunk are already in the past by the
time it lands, and a naively resampled chunk can be wholly incompatible with the
trajectory the robot is already following.

RTC reframes replanning as inpainting: the first ``d`` actions are *frozen* to
the previous chunk, the overlap region is guided toward it with exponentially
decaying weight, and the tail is generated freely. The guidance is applied to
the denoiser's clean-sample prediction through a vector-Jacobian product, so it
needs no training-time changes and works on existing checkpoints.

Nothing here mutates the model. ``DiffusionPolicy._conditional_sample`` and
``_flow_matching_sample`` are untouched, so behaviour is unchanged for every
caller that does not opt into RTC.
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from neuracore_types import DataType

from neuracore.ml import BatchedInferenceInputs

if TYPE_CHECKING:
    from neuracore.ml.algorithms.diffusion_policy.diffusion_policy import (
        DiffusionPolicy,
    )

logger = logging.getLogger(__name__)

# The paper's beta, calibrated for flow matching. There the guidance weight has
# no other bound, so beta is what keeps few-step sampling stable.
DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT = 5.0
# The diffusion path needs a much larger beta. Its weight carries a second,
# stricter cap (zeta <= 1 / k_t, i.e. "pull no further than onto Y"), so beta
# only binds near the ends of the noise schedule -- and the clean end is exactly
# where a few-step DDIM sampler produces the output. At beta = 5 the final step
# applies about a tenth of the intended correction. Measured on a trained UR5e
# policy, the boundary discontinuity falls monotonically until beta ~= 50 and is
# then completely flat: the hard cap has taken over, so larger values are inert
# rather than unstable.
DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT = 50.0
DEFAULT_RTC_INFERENCE_STEPS = 10

PREFIX_ATTENTION_SCHEDULES = ("exp", "linear", "ones", "zeros")

# Guards against division by zero at the endpoints of the noise schedule, where
# the analytic guidance weight diverges and the beta clip takes over anyway.
_EPS = 1e-8


@dataclass(frozen=True)
class RTCConfig:
    """Configuration for real-time chunking.

    Attributes:
        inference_delay: ``d``, the inference latency measured in controller
            ticks. Actions ``[0, d)`` of a new chunk are frozen to the previous
            chunk because they will already have executed when it lands. Must
            satisfy ``0 <= d <= H - s``.
        execution_horizon: ``s``, the number of actions executed from each chunk
            before replanning. The final ``s`` actions of a chunk are generated
            without any guidance.
        max_guidance_weight: ``beta``, the clip applied to the guidance weight.
            ``None`` picks the right default for the model's process type -
            :data:`DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT` for flow matching and the
            much larger :data:`DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT` for
            diffusion, which needs it because its weight carries a second cap.
            Set a number to override.
        prefix_attention_schedule: Soft-mask family over the overlap region.
            ``"exp"`` is the paper's default; ``"ones"`` reproduces hard
            inpainting over the whole overlap and ``"zeros"`` reduces to
            freezing only the ``d`` guaranteed actions.
        force_ddim: Sample with a deterministic DDIM scheduler even when the
            model was configured with DDPM. DDPM injects fresh noise at every
            reverse step, so the frozen prefix never converges exactly and
            successive replans stay stochastic - which is the discontinuity RTC
            exists to remove. Ignored for flow-matching models.
        num_inference_steps: Overrides ``model.num_inference_steps`` for the RTC
            path only, leaving offline inference untouched. Must divide
            ``num_train_timesteps`` for the diffusion path.
        guidance_start_step: Number of leading denoising steps to run unguided.
            The guidance weight is beta-clipped near the pure-noise end anyway,
            so skipping a step or two trades little quality for latency.
    """

    inference_delay: int
    execution_horizon: int
    max_guidance_weight: float | None = None
    prefix_attention_schedule: str = "exp"
    force_ddim: bool = True
    num_inference_steps: int | None = DEFAULT_RTC_INFERENCE_STEPS
    guidance_start_step: int = 0

    def resolve_guidance_weight(self, process_type: str) -> float:
        """Return ``beta``, filling in the per-process default when unset.

        Args:
            process_type: The model's ``process_type``.

        Returns:
            float: The guidance weight clip to use.
        """
        if self.max_guidance_weight is not None:
            return self.max_guidance_weight
        if process_type == "flow_matching":
            return DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT
        return DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT

    def with_inference_delay(self, inference_delay: int) -> "RTCConfig":
        """Return a copy with a new measured inference delay.

        Args:
            inference_delay: The new value for ``d``, in controller ticks.

        Returns:
            RTCConfig: A copy of this config with ``inference_delay`` replaced.
        """
        return RTCConfig(
            inference_delay=inference_delay,
            execution_horizon=self.execution_horizon,
            max_guidance_weight=self.max_guidance_weight,
            prefix_attention_schedule=self.prefix_attention_schedule,
            force_ddim=self.force_ddim,
            num_inference_steps=self.num_inference_steps,
            guidance_start_step=self.guidance_start_step,
        )


def rtc_soft_mask(
    prediction_horizon: int,
    inference_delay: int,
    execution_horizon: int,
    *,
    schedule: str = "exp",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the RTC soft mask ``W`` over chunk indices.

    With ``H`` the prediction horizon, ``d`` the inference delay and ``s`` the
    execution horizon, and ``c_i = (H - s - i) / (H - s - d + 1)``:

    * ``i < d``: ``W_i = 1`` - frozen, guaranteed to have executed.
    * ``d <= i < H - s``: ``W_i = c_i (e^{c_i} - 1) / (e - 1)`` - soft overlap.
    * ``i >= H - s``: ``W_i = 0`` - generated freely.

    Args:
        prediction_horizon: ``H``, the length of a chunk.
        inference_delay: ``d``, in controller ticks.
        execution_horizon: ``s``, actions executed per chunk.
        schedule: One of ``PREFIX_ATTENTION_SCHEDULES``.
        device: Device for the returned tensor.
        dtype: Dtype for the returned tensor.

    Returns:
        torch.Tensor: The mask ``W`` with shape ``(prediction_horizon,)``.

    Raises:
        ValueError: If the horizons are inconsistent, the real-time constraint
            ``d <= H - s`` is violated, or ``schedule`` is unknown.
    """
    h, d, s = prediction_horizon, inference_delay, execution_horizon
    if h < 1:
        raise ValueError(f"prediction_horizon must be >= 1, got {h}")
    if s < 1 or s > h:
        raise ValueError(f"execution_horizon must be in [1, {h}], got {s}")
    if d < 0:
        raise ValueError(f"inference_delay must be >= 0, got {d}")
    if d > h - s:
        raise ValueError(
            f"Real-time constraint violated: inference_delay d={d} exceeds "
            f"H - s = {h - s} (H={h}, s={s}). Reduce inference latency, reduce "
            "the execution horizon, or train with a longer prediction horizon."
        )
    if schedule not in PREFIX_ATTENTION_SCHEDULES:
        raise ValueError(
            f"Unknown prefix_attention_schedule {schedule!r}; "
            f"expected one of {PREFIX_ATTENTION_SCHEDULES}."
        )

    index = torch.arange(h, device=device, dtype=dtype)
    c = (float(h - s) - index) / float(h - s - d + 1)

    if schedule == "exp":
        weights = c * (torch.exp(c) - 1.0) / (math.e - 1.0)
    elif schedule == "linear":
        weights = c
    elif schedule == "ones":
        weights = torch.ones_like(c)
    else:  # "zeros"
        weights = torch.zeros_like(c)

    weights = torch.where(index < d, torch.ones_like(weights), weights)
    weights = torch.where(index >= h - s, torch.zeros_like(weights), weights)
    return weights


def align_previous_chunk(
    prev_chunk: torch.Tensor,
    ticks_consumed: int,
    prediction_horizon: int,
) -> torch.Tensor:
    """Shift a previous chunk so its indices line up with a new chunk's.

    Index ``i`` of the result is the action the previous chunk had scheduled for
    the same controller tick as index ``i`` of the chunk about to be generated.
    The tail is zero-padded; when ``ticks_consumed == s`` the padding lands
    exactly where the mask is zero, so it is never read by the guidance.

    Args:
        prev_chunk: Previous chunk with shape ``(B, H_prev, A)``.
        ticks_consumed: Actions already executed from ``prev_chunk``.
        prediction_horizon: ``H``, the length of the new chunk.

    Returns:
        torch.Tensor: Aligned chunk with shape ``(B, prediction_horizon, A)``.

    Raises:
        ValueError: If ``ticks_consumed`` is negative.
    """
    if ticks_consumed < 0:
        raise ValueError(f"ticks_consumed must be >= 0, got {ticks_consumed}")

    batch_size, _, action_dim = prev_chunk.shape
    tail = prev_chunk[:, ticks_consumed:, :]
    aligned = prev_chunk.new_zeros((batch_size, prediction_horizon, action_dim))
    overlap = min(tail.shape[1], prediction_horizon)
    if overlap > 0:
        aligned[:, :overlap, :] = tail[:, :overlap, :]
    return aligned


def as_ddim_scheduler(
    scheduler: DDPMScheduler | DDIMScheduler,
) -> DDIMScheduler:
    """Build a deterministic DDIM sampler for the same learned noise network.

    Training only ever uses ``add_noise`` and a random timestep, so the network
    learns ``eps_theta(x_t, t)`` for the beta schedule and never the reverse
    kernel. DDIM is a different reverse sampler for that same quantity, so this
    swap is valid on a DDPM-trained checkpoint provided the schedule config is
    carried over verbatim.

    Args:
        scheduler: The model's configured scheduler.

    Returns:
        DDIMScheduler: ``scheduler`` itself if it is already DDIM, otherwise a
        new DDIM instance sharing its noise schedule.
    """
    if isinstance(scheduler, DDIMScheduler):
        return scheduler
    config = scheduler.config
    return DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        clip_sample_range=config.clip_sample_range,
        prediction_type=config.prediction_type,
        set_alpha_to_one=True,
        steps_offset=0,
    )


def missing_rtc_attributes(model: object) -> list[str]:
    """List the attributes ``model`` lacks for real-time chunking.

    Checked structurally rather than with ``isinstance``: model archives bundle
    their algorithm source and ``AlgorithmLoader`` imports it as a fresh module,
    so an archived ``DiffusionPolicy`` is a different class object from the
    installed one and would fail an identity check despite being fully usable.

    Args:
        model: The loaded model.

    Returns:
        list[str]: Missing attribute names; empty if the model is usable.
    """
    missing = [
        name
        for name in (
            "unet",
            "action_normalizer",
            "process_type",
            "prediction_type",
            "num_inference_steps",
            "max_output_size",
            "clip_sample",
            "clip_sample_range",
            "output_prediction_horizon",
            "_predict_action",
        )
        if not hasattr(model, name)
    ]
    if not hasattr(model, "_build_global_cond") and not all(
        hasattr(model, name)
        for name in ("_combine_proprio", "_prepare_global_conditioning")
    ):
        missing.append("_build_global_cond")
    return missing


def supports_real_time_chunking(model: object) -> bool:
    """Whether ``model`` exposes everything real-time chunking needs.

    Args:
        model: The loaded model.

    Returns:
        bool: True if guided sampling can run against this model.
    """
    return not missing_rtc_attributes(model)


def build_global_conditioning(
    model: "DiffusionPolicy", batch: BatchedInferenceInputs
) -> torch.Tensor:
    """Encode observations into the UNet's global conditioning vector.

    Model archives bundle the algorithm source they were trained with, and
    ``AlgorithmLoader`` runs *that* copy rather than the installed package. A
    model trained before ``DiffusionPolicy._build_global_cond`` existed
    therefore will not have it, so fall back to the equivalent inline block from
    ``_predict_action``. Without this, real-time chunking would only work on
    models trained after that refactor, which defeats the point of a method that
    needs no retraining.

    Args:
        model: The diffusion policy, from the installed package or an archive.
        batch: Input observations.

    Returns:
        torch.Tensor: Global conditioning with shape (B, global_cond_dim).
    """
    build = getattr(model, "_build_global_cond", None)
    if build is not None:
        return build(batch)

    batch_size = len(batch)
    joint_states = model._combine_proprio(batch)
    if (
        DataType.RGB_IMAGES in model.input_data_types
        and DataType.RGB_IMAGES in batch.inputs
    ):
        return model._prepare_global_conditioning(
            joint_states,
            batch.inputs[DataType.RGB_IMAGES],
            batch.inputs_mask[DataType.RGB_IMAGES],
        )
    if DataType.RGB_IMAGES in model.input_data_types:
        # RGB configured but absent in this batch: zero-pad to full cond dim
        global_cond = torch.zeros(
            batch_size, model.global_cond_dim, device=model.device
        )
        if joint_states is not None:
            global_cond[:, : joint_states.shape[-1]] = joint_states
        return global_cond
    return joint_states  # proprio-only model, dims already match


def _vjp_through_denoiser(
    x: torch.Tensor,
    clean_prediction: torch.Tensor,
    weighted_mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute ``(d clean_prediction / dx)^T [W * (target - clean_prediction)]``.

    A single reverse-mode sweep; the Jacobian is never materialised. Only ``x``
    is passed as an autograd input, so weight-gradient branches are pruned.

    Args:
        x: The sample the denoiser was evaluated at; must require grad.
        clean_prediction: The denoiser's clean-sample estimate, in the graph.
        weighted_mask: Mask broadcastable to ``clean_prediction``.
        target: The inpainting target, detached.

    Returns:
        torch.Tensor: The vector-Jacobian product, same shape as ``x``.
    """
    error = (weighted_mask * (target - clean_prediction)).detach()
    (grad_x,) = torch.autograd.grad(
        outputs=clean_prediction,
        inputs=x,
        grad_outputs=error,
        retain_graph=False,
        create_graph=False,
    )
    return grad_x


def guided_flow_matching_sample(
    model: "DiffusionPolicy",
    sample: torch.Tensor,
    target_normalized: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_guidance_weight: float = DEFAULT_FLOW_MAX_GUIDANCE_WEIGHT,
    num_inference_steps: int | None = None,
    guidance_start_step: int = 0,
    global_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the velocity field with RTC inpainting guidance.

    Mirrors ``DiffusionPolicy._flow_matching_sample`` and reduces to it exactly
    when ``mask`` is all zeros. Flow-matching training uses
    ``x_t = (1 - t) * noise + t * action``, so the sampler's time variable is
    the flow time ``tau`` directly and the clean-sample estimate
    ``x_hat = x_tau + (1 - tau) v`` is exact.

    Args:
        model: The diffusion policy providing the velocity network.
        sample: Gaussian prior with shape ``(B, H, A)``, in normalized space.
        target_normalized: Aligned previous chunk ``Y``, normalized, ``(B, H, A)``.
        mask: Soft mask ``W`` with shape ``(H,)``.
        max_guidance_weight: ``beta``, the guidance weight clip.
        num_inference_steps: Euler steps; defaults to the model's setting.
        guidance_start_step: Leading steps to run unguided.
        global_cond: Conditioning with shape ``(B, global_cond_dim)``.

    Returns:
        torch.Tensor: Sampled chunk with shape ``(B, H, A)``, still normalized.
    """
    steps = num_inference_steps or model.num_inference_steps
    dt = 1.0 / steps
    weighted_mask = mask.view(1, -1, 1).to(sample.dtype)
    target_normalized = target_normalized.detach()

    for step in range(steps):
        # Continuous time shared across the batch, ascending from 0 to 1.
        tau = step * dt

        if step < guidance_start_step:
            with torch.no_grad():
                t_full = torch.full(
                    sample.shape[:1], tau, dtype=torch.float32, device=sample.device
                )
                sample = sample + dt * model.unet(
                    sample, t_full, global_cond=global_cond
                )
        else:
            # A fresh leaf each step; the previous step's output was detached.
            x = sample.detach().requires_grad_(True)
            with torch.enable_grad():
                t_full = torch.full(
                    x.shape[:1], tau, dtype=torch.float32, device=x.device
                )
                velocity = model.unet(x, t_full, global_cond=global_cond)
                # Do not clamp inside the graph: saturated clamp has zero
                # gradient and would kill the guidance exactly where it bites.
                clean_prediction = x + (1.0 - tau) * velocity
                grad_x = _vjp_through_denoiser(
                    x, clean_prediction, weighted_mask, target_normalized
                )

            # min(beta, (1 - tau) / (tau * r_tau^2)) with
            # r_tau^2 = (1 - tau)^2 / (tau^2 + (1 - tau)^2), which simplifies to
            # (tau^2 + (1 - tau)^2) / (tau (1 - tau)). Singular at both ends;
            # beta clips it and the floor keeps tau == 0 finite.
            denominator = max(tau * (1.0 - tau), _EPS)
            guidance_weight = min(
                max_guidance_weight,
                (tau * tau + (1.0 - tau) ** 2) / denominator,
            )
            sample = x.detach() + dt * (velocity.detach() + guidance_weight * grad_x)

        if model.clip_sample:
            sample = torch.clamp(
                sample, -model.clip_sample_range, model.clip_sample_range
            )

    return sample


def guided_diffusion_sample(
    model: "DiffusionPolicy",
    sample: torch.Tensor,
    target_normalized: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_guidance_weight: float = DEFAULT_DIFFUSION_MAX_GUIDANCE_WEIGHT,
    num_inference_steps: int | None = None,
    guidance_start_step: int = 0,
    global_cond: torch.Tensor | None = None,
    scheduler: DDPMScheduler | DDIMScheduler | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise with RTC inpainting guidance (DDPM / DDIM).

    Mirrors ``DiffusionPolicy._conditional_sample`` and reduces to it exactly
    when ``mask`` is all zeros.

    Matching the signal-to-noise ratio between the variance-preserving path
    ``x_t = sqrt(a_bar) x_0 + sqrt(1 - a_bar) eps`` and the rectified-flow path
    ``x_tau = tau x_0 + (1 - tau) eps`` gives ``tau = sqrt(a_bar) / k`` with
    ``k = sqrt(a_bar) + sqrt(1 - a_bar)``, and hence ``r_tau^2 = 1 - a_bar``.
    Substituting into the paper's guidance coefficient, the pull applied to the
    clean-sample estimate is ``min(beta (1 - tau), (tau^2 + (1-tau)^2) / tau)``.

    That coefficient is calibrated for a denoiser whose clean-sample Jacobian is
    the identity. An epsilon-parameterised diffusion model instead has
    ``d x0_hat / d x_t = (I - sqrt(1 - a_bar) d eps / d x) / sqrt(a_bar)``, whose
    identity component grows like ``1 / sqrt(a_bar)``. Cancelling that factor so
    the two processes apply the same effective pull leaves

        x0_guided = x0_hat + zeta_t (d x0_hat / d x_t)^T [W * (Y - x0_hat)]
        zeta_t    = min(beta * sqrt(a_bar_t * (1 - a_bar_t)), 1) / k_t

    which vanishes at both ends of the schedule and is bounded in between.
    Skipping the ``sqrt(a_bar)`` cancellation makes the first, highest-noise step
    overshoot by roughly ``1 / sqrt(a_bar)`` and destabilises the whole
    trajectory. The correction is folded back into the model output before
    ``scheduler.step`` so the scheduler's own coefficients and sample clipping
    still apply.

    Args:
        model: The diffusion policy providing the noise network.
        sample: Gaussian prior with shape ``(B, H, A)``, in normalized space.
        target_normalized: Aligned previous chunk ``Y``, normalized, ``(B, H, A)``.
        mask: Soft mask ``W`` with shape ``(H,)``.
        max_guidance_weight: ``beta``, the guidance weight clip.
        num_inference_steps: Denoising steps; defaults to the model's setting.
        guidance_start_step: Leading steps to run unguided.
        global_cond: Conditioning with shape ``(B, global_cond_dim)``.
        scheduler: Sampler to use; defaults to the model's own scheduler.
        generator: Random number generator for stochastic samplers.

    Returns:
        torch.Tensor: Sampled chunk with shape ``(B, H, A)``, still normalized.

    Raises:
        ValueError: If the model has no noise scheduler, or its
            ``prediction_type`` is not ``"epsilon"`` or ``"sample"``.
    """
    scheduler = scheduler if scheduler is not None else model.noise_scheduler
    if scheduler is None:
        raise ValueError(
            "guided_diffusion_sample requires a noise scheduler; this model was "
            "built for flow matching. Use guided_flow_matching_sample instead."
        )
    if model.prediction_type not in ("epsilon", "sample"):
        raise ValueError(
            f"RTC does not support prediction_type {model.prediction_type!r}; "
            "expected 'epsilon' or 'sample'."
        )

    steps = num_inference_steps or model.num_inference_steps
    scheduler.set_timesteps(steps, device=sample.device)
    # Indexed by raw training timestep, which is what scheduler.timesteps holds.
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=sample.device, dtype=sample.dtype
    )

    weighted_mask = mask.view(1, -1, 1).to(sample.dtype)
    target_normalized = target_normalized.detach()

    for step, t in enumerate(scheduler.timesteps):
        if step < guidance_start_step:
            with torch.no_grad():
                t_full = torch.full(
                    sample.shape[:1], t, dtype=torch.long, device=sample.device
                )
                model_output = model.unet(sample, t_full, global_cond=global_cond)
            sample = scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample
            continue

        alpha_bar = alphas_cumprod[t]
        sqrt_alpha_bar = alpha_bar.clamp_min(_EPS).sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).clamp_min(_EPS).sqrt()

        x = sample.detach().requires_grad_(True)
        with torch.enable_grad():
            t_full = torch.full(x.shape[:1], t, dtype=torch.long, device=x.device)
            model_output = model.unet(x, t_full, global_cond=global_cond)
            if model.prediction_type == "epsilon":
                clean_prediction = (x - sqrt_one_minus * model_output) / sqrt_alpha_bar
            else:
                clean_prediction = model_output
            grad_x = _vjp_through_denoiser(
                x, clean_prediction, weighted_mask, target_normalized
            )

        zeta = torch.clamp(
            max_guidance_weight * sqrt_alpha_bar * sqrt_one_minus, max=1.0
        ) / (sqrt_alpha_bar + sqrt_one_minus)
        guided_clean = clean_prediction.detach() + zeta * grad_x

        # Re-express in the parameterisation the scheduler expects, so it
        # recomputes exactly this clean prediction internally.
        if model.prediction_type == "epsilon":
            guided_output = (
                x.detach() - sqrt_alpha_bar * guided_clean
            ) / sqrt_one_minus
        else:
            guided_output = guided_clean

        sample = scheduler.step(
            guided_output, t, x.detach(), generator=generator
        ).prev_sample

    return sample


def rtc_predict_actions(
    model: "DiffusionPolicy",
    batch: BatchedInferenceInputs,
    prev_chunk: torch.Tensor | None,
    config: RTCConfig | None,
    *,
    prediction_horizon: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one RTC-guided action chunk.

    Falls back to the model's ordinary unguided sampler when ``prev_chunk`` is
    ``None``, which is the case for the very first chunk of a rollout.

    Args:
        model: The diffusion policy to sample from.
        batch: The observation to condition on.
        prev_chunk: Previous chunk, **unnormalized** and already aligned by
            :func:`align_previous_chunk`, with shape ``(B, H, A)``. ``None`` on
            the first chunk.
        config: Real-time chunking configuration. Only read when ``prev_chunk``
            is given, so it may be ``None`` for the first chunk.
        prediction_horizon: Chunk length; defaults to the model's trained value.
        generator: Random number generator for reproducible sampling.

    Returns:
        torch.Tensor: Unnormalized actions with shape ``(B, H, A)``.

    Raises:
        ValueError: If ``prev_chunk`` is given without a ``config``.
    """
    horizon = prediction_horizon or model.output_prediction_horizon

    if prev_chunk is not None and config is None:
        raise ValueError("config is required when prev_chunk is provided.")

    if prev_chunk is None and (config is None or config.num_inference_steps is None):
        # Nothing to guide against and no sampler override to apply: this is
        # exactly the model's own unguided sampler.
        with torch.no_grad():
            return model.action_normalizer.unnormalize(
                model._predict_action(batch, horizon)
            )

    if config is None:
        raise ValueError("config is required to sample an action chunk.")

    if prev_chunk is None:
        # The bootstrap chunk has nothing to guide against, but it must still
        # honour the step-count and scheduler overrides. Otherwise it falls back
        # to the model's offline default (often 100 steps) and takes an order of
        # magnitude longer than every chunk after it, which both skews the
        # startup latency measurement and stalls the first replan.
        prev_chunk = torch.zeros(
            (len(batch), horizon, model.max_output_size),
            dtype=torch.float32,
            device=model.device,
        )
        config = RTCConfig(
            inference_delay=0,
            execution_horizon=horizon,  # mask is all zeros: nothing is guided
            max_guidance_weight=config.max_guidance_weight,
            prefix_attention_schedule=config.prefix_attention_schedule,
            force_ddim=config.force_ddim,
            num_inference_steps=config.num_inference_steps,
            # Skipping every step's guidance also skips every backward pass.
            guidance_start_step=config.num_inference_steps or 0,
        )

    if model.process_type != "flow_matching" and model.prediction_type == "sample":
        # With x0-prediction the VJP is purely the network Jacobian: it loses
        # the explicit x_t / sqrt(a_bar) term that makes epsilon-prediction's
        # guidance well conditioned, so it can collapse toward zero.
        logger.warning(
            "RTC guidance is validated for prediction_type='epsilon'; "
            "'sample' may produce weak guidance."
        )

    # Guidance needs autograd even though callers such as PolicyInference run
    # under torch.no_grad(); inference_mode(False) also covers inference_mode.
    with torch.inference_mode(False):
        with torch.no_grad():
            # Images are encoded once, outside the denoising loop, and detached
            # so the vision encoders never enter the autograd graph.
            global_cond = build_global_conditioning(model, batch)
            if global_cond is not None:
                global_cond = global_cond.detach().clone()
            target_normalized = model.action_normalizer.normalize(prev_chunk).clone()

        mask = rtc_soft_mask(
            horizon,
            config.inference_delay,
            config.execution_horizon,
            schedule=config.prefix_attention_schedule,
            device=model.device,
        )
        sample = torch.randn(
            (len(batch), horizon, model.max_output_size),
            dtype=torch.float32,
            device=model.device,
            generator=generator,
        )

        if model.process_type == "flow_matching":
            sampled = guided_flow_matching_sample(
                model,
                sample,
                target_normalized,
                mask,
                max_guidance_weight=config.resolve_guidance_weight(model.process_type),
                num_inference_steps=config.num_inference_steps,
                guidance_start_step=config.guidance_start_step,
                global_cond=global_cond,
            )
        else:
            scheduler = model.noise_scheduler
            if config.force_ddim and scheduler is not None:
                # Cached so the schedule is not rebuilt on every replan.
                cached = getattr(model, "_rtc_ddim_scheduler", None)
                if cached is None:
                    cached = as_ddim_scheduler(scheduler)
                    model._rtc_ddim_scheduler = cached
                scheduler = cached
            sampled = guided_diffusion_sample(
                model,
                sample,
                target_normalized,
                mask,
                max_guidance_weight=config.resolve_guidance_weight(model.process_type),
                num_inference_steps=config.num_inference_steps,
                guidance_start_step=config.guidance_start_step,
                global_cond=global_cond,
                scheduler=scheduler,
                generator=generator,
            )

        with torch.no_grad():
            return model.action_normalizer.unnormalize(sampled.detach())
