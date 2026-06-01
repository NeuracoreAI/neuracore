<!-- cspell:disable -->
# Training-only Color Jitter for DiffusionPolicy

**Date:** 2026-06-01
**Status:** Approved
**File touched:** `neuracore/ml/algorithms/diffusion_policy/diffusion_policy.py`,
`neuracore/ml/config/algorithm/diffusion_policy.yaml`,
`tests/unit/ml/algorithms/test_diffusion_policy.py`

## Goal

Add configurable RGB color-jitter data augmentation to `DiffusionPolicy`,
applied **only during training** and never during inference. This follows the
same "config flag on the existing class" pattern used to add `flow_matching`
via `process_type`, rather than introducing a new algorithm class.

## Requirements

- Color jitter parameters are configurable.
- Jitter is applied during training only.
- Jitter does not run at all during inference.
- Default behavior is unchanged (augmentation off by default).
- Use `torchvision.transforms.ColorJitter`.

## Design

### Constructor parameters

Add three keyword arguments to `DiffusionPolicy.__init__`, all defaulting to
`0.0` (off):

- `color_jitter_brightness: float = 0.0`
- `color_jitter_contrast: float = 0.0`
- `color_jitter_saturation: float = 0.0`

Hue is intentionally not exposed (kept at `0`).

If all three values are `0.0`, no jitter transform is constructed
(`self.color_jitter = None`) and behavior is byte-for-byte identical to today.
Otherwise build:

```python
self.color_jitter = T.ColorJitter(
    brightness=color_jitter_brightness,
    contrast=color_jitter_contrast,
    saturation=color_jitter_saturation,
    hue=0,
)
```

`ColorJitter` holds no learnable parameters, so it is stored as a plain
attribute (not registered as a submodule / not in an `nn.ModuleList`).

### Where jitter runs

`_prepare_global_conditioning` is shared by both the training and inference
paths. Add an explicit boolean parameter:

```python
def _prepare_global_conditioning(
    self,
    joint_states,
    batched_nc_data,
    camera_images_mask,
    apply_color_jitter: bool = False,
) -> torch.FloatTensor:
```

- `training_step` calls it with `apply_color_jitter=True`.
- `_predict_action` (inference) calls it with `apply_color_jitter=False`
  (the default).

This makes the training-only guarantee explicit and independent of whether the
model is in `train()` or `eval()` mode.

Inside the per-camera loop, jitter is applied to the raw frame **before**
normalization:

```python
last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W), values in [0, 1]
if apply_color_jitter and self.color_jitter is not None:
    last_frame = self.color_jitter(last_frame)
transformed = normalizer(last_frame)
features = encoder(transformed)
```

Order matters: jitter operates on `[0, 1]` intensities, so it must run before
`T.Normalize` (ResNet mean/std), not after.

### Config

Add a new section to `neuracore/ml/config/algorithm/diffusion_policy.yaml`:

```yaml
  # Vision augmentation (training only; 0.0 disables each)
  color_jitter_brightness: 0.0
  color_jitter_contrast: 0.0
  color_jitter_saturation: 0.0
```

## Trade-offs / decisions

- **Batch-level jitter:** `torchvision.transforms.ColorJitter` samples one set
  of random factors per call, so all images in a batch receive the same jitter
  factors on a given step. This is standard, cheap, and GPU-friendly. Per-image
  jitter would require a Python loop and is out of scope.
- **Explicit flag vs. `self.training`:** an explicit `apply_color_jitter`
  argument is used instead of relying on `self.training`, guaranteeing inference
  never augments regardless of the caller's mode.
- **Default off:** all three params default to `0.0`, preserving current
  behavior and keeping existing configs/tests valid.

## Testing

In `tests/unit/ml/algorithms/test_diffusion_policy.py`:

1. Construct `DiffusionPolicy` with jitter enabled (non-zero
   brightness/contrast/saturation) on an RGB-input config; run a training
   forward + backward and assert finite loss and finite gradients (mirroring
   `test_process_type_forward_backward`).
2. Assert inference output is unaffected by jitter: with a fixed seed, the
   `forward` output is identical whether jitter params are zero or non-zero
   (because inference passes `apply_color_jitter=False`).
