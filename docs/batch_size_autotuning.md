# Automatic Batch-Size Tuning

When you set `batch_size: auto` in the training config, Neuracore automatically
picks the **largest batch size that fits on your GPU** before training starts.
This page explains how it works, in plain language.

> Code: `neuracore/ml/trainers/batch_autotuner.py` — entry point
> `find_optimal_batch_size(...)`.

## Contents

1. [Summary](#summary)
2. [Key terms](#key-terms)
3. [The memory model (the core idea)](#the-memory-model-the-core-idea)
4. [The algorithm, step by step](#the-algorithm-step-by-step)
5. [How the batch size shrinks on OOM](#how-the-batch-size-shrinks-on-oom)
6. [Why the safety margin](#why-the-safety-margin)
7. [Parameters](#parameters)
8. [Edge cases](#edge-cases)
9. [Worked example](#worked-example)

---

## Summary

GPU memory used during training grows **almost linearly** with the batch
size. So we:

1. **measure** memory at a few batch sizes (called *probes*),
2. **fit a line** through the measurements,
3. **read off** the batch size that would fill the GPU, and
4. **shave off a safety margin** so long training runs don't crash.

Typical cost: ~3–5 short trial runs, even for very large models.

---

## Key terms

| Term | Meaning |
|---|---|
| **Batch size** | How many samples are processed together in one training step. Bigger = faster training, but more memory. |
| **VRAM** | The GPU's own memory (e.g. 40 GB on an A100). All training tensors must fit here. |
| **OOM** | "Out Of Memory" — the step needed more memory than available, so it crashes. |
| **Peak memory** | The highest memory used at any moment during a step. This is what must stay under the limit. |
| **Reserved memory** | Memory the GPU allocator has claimed (≥ what's actively in use). OOM happens against *reserved* memory, so that's what we measure. |
| **Probe** | A short trial run at one batch size to (a) check if it fits and (b) measure its peak memory. Each probe takes seconds–minutes. |
| **Intercept** | The *fixed* memory that doesn't depend on batch size — model weights, optimizer state, CUDA context. Can dominate for big models. |
| **Slope** | The *extra* memory per added sample (the activations for one more item). |
| **Convex** | A curve that bends upward — here, each extra sample costs a bit *more* than the previous one. |
| **Budget** | The memory ceiling we aim to fill: `90% of VRAM`, leaving a little headroom. |
| **Safety factor** | A final shrink (×0.7) applied to the result, so real training has spare room. |
| **Anchor** | A probe point used to fit the line. We use three: a small one, a medium one, and a large one. |
| **Geometric mean** | `sqrt(a · b)` — the "halfway point on a log scale" between two numbers; used to shrink quickly. |

---


## The memory model (the core idea)

For a fixed model, peak GPU memory at batch size `b` is roughly a straight line:

```
peak(b) ≈ intercept + slope · b
```

- **`intercept`** = fixed cost (weights, optimizer, CUDA context). For a big model
  this can be most of the memory — e.g. PI0 uses ~18 GB before the batch matters.
- **`slope`** = memory per extra sample (its activations).

We want the largest `b` whose `peak(b)` stays under the **budget**:

```
budget = total_VRAM · max_gpu_utilization        (max_gpu_utilization = 0.9)
```

Setting `peak(b) = budget` and solving for `b`:

```
b_max = (budget − intercept) / slope
```

Then we apply the safety factor (see [Why the safety margin](#why-the-safety-margin)):

```
chosen = floor(b_max · safety_factor)             (safety_factor = 0.7)
```

Memory is **convex**: each extra sample costs slightly more than the last
(memory fragmentation + attention/workspace growth). Two *tiny* batches use almost
the same memory, so a line through them has an unreliable slope — that's why we
also probe a **larger** point and measure the slope between the 2nd and 3rd probes.

![Picking a batch size from a few memory probes: probe1 and probe2 sit on the
flat fixed-cost floor, probe3 is a larger fitting batch, and the line through
probe2 and probe3 is extrapolated to where memory hits the 0.9·VRAM budget
(b_max); the chosen batch is 0.7·b_max.](assets/gpu_batch_size.png)

- `probe1`, `probe2` (batch **1** and **8**) sit in the flat region — fixed cost
  dominates, so a line through them *under*-estimates the true slope.
- `probe3` is a larger batch whose slope reflects real training. The line through
  `probe2` and `probe3` predicts where memory hits the budget (`b_max`).
- Because the curve bends upward, the prediction can slightly overshoot; if it
  OOMs we [shrink](#how-the-batch-size-shrinks-on-oom) until it fits.

---

## The algorithm, step by step

Every probe runs in a **separate process** (so a GPU crash can't kill training)
and measures steady-state **reserved** memory after a short warm-up.

1. **Probe the smallest batch**, `low = min_batch_size` (default **1**).
   If even this OOMs → raise `OutOfMemoryError`: the model can't train on this GPU
   at all.

2. **Probe the second anchor**, `high = min(min · 8, max)` (default **8**).
   If it OOMs → [shrink](#how-the-batch-size-shrinks-on-oom) down from `high` and stop.

3. **Place a large third anchor.** Fit a rough line through the first two probes,
   solve it for `rough = b_max` (no safety yet), and start the anchor halfway down:

   > `third = rough / 2`

4. **Make the third anchor fit.** Probe it; while it OOMs, pull it toward `high`
   with the [geometric mean](#how-the-batch-size-shrinks-on-oom) and re-probe:

   > `third ← floor( sqrt(high · third) )`

5. **Predict the max batch** using the slope between the **2nd and 3rd** probes
   (only this segment — the two tiny probes are nearly flat and would flatten the
   slope and overshoot):

   > `slope = (peak₃ − peak₂) / (third − high)`
   >
   > `predicted = third + (budget − peak₃) / slope`

6. **Confirm the prediction.** Probe it; if it OOMs,
   [shrink](#how-the-batch-size-shrinks-on-oom) it 10% at a time until it fits.

7. **Apply the safety margin** to the largest batch that actually fit:

   > `chosen = floor( largest_fitting_batch · 0.7 )`

---

## How the batch size shrinks on OOM

When something OOMs we shrink and re-probe —
but the **step size depends on what OOM'd**, because the two cases overshoot by
very different amounts:

**1. The third anchor OOMs → big, fast jumps (geometric mean).**
The anchor was placed using the unreliable tiny-batch slope, so it can be a *huge*
overshoot. We shrink toward `high` with the geometric mean — the log-scale
midpoint, roughly the square root of the ratio:

```
third ← floor( sqrt(high · third) )
```

Example: `high = 8`, `third = 162` (OOM) → `sqrt(8 · 162) = 36`. **One step,
162 → 36** (a 4.5× drop). The bigger the overshoot, the bigger the jump — and it
always stays above `high`, so one or two probes usually land a fitting anchor.

**2. The prediction OOMs → small, gentle steps (×0.9).**
After the good 2nd–3rd slope, the prediction is only ever a *little* over, so we
step down 10% at a time until it fits:

```
candidate ← floor(candidate · 0.9)      # _DOWNSCALE_FACTOR
```

| What OOM'd | Shrink rule | Why this step size |
|---|---|---|
| **3rd anchor** | `third ← √(high · third)` | overshoot can be huge → big multiplicative jumps |
| **Prediction** | `candidate ← candidate · 0.9` | overshoot is small → gentle 10% steps |

Both stop at the largest batch already known to fit (so they always terminate),
and the safety factor is applied to whatever fit.

---

## Why the safety margin

A probe finds the batch that *just barely* fits during a short, 2-iteration test.
Real training uses **more** memory over a long run, so we leave headroom:

- **batch-content variance** — batches differ in size (variable episode lengths,
  camera counts, padding); the probe only sees the first few.
- **memory fragmentation** builds up over thousands of steps.
- **other consumers** not exercised by the probe (full validation, EMA weights,
  checkpoint buffers, logging).

So we train at `0.7 × (the batch that fit)` to avoid an OOM crash hours into
training.

---

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_batch_size` | `1` | Smallest allowed batch / lower clamp. Also the first probe. |
| `max_batch_size` | `len(train_dataset)` | Upper clamp — you can't batch more than the dataset. |
| `max_gpu_utilization` | `0.9` | Fraction of VRAM used as the budget. |
| `safety_factor` | `0.7` | Final batch = `0.7 × largest fitting batch`. |
| `high` anchor | `min · 8` | The second probe (calibration point). |
| `_DOWNSCALE_FACTOR` | `0.9` | Step size when shrinking an OOM *prediction*. |
| warm-up / measure iters | `1` / `2` | Iterations before / during memory measurement. |

Override `min_batch_size` / `max_batch_size` in the training config, or set a fixed
`batch_size: <N>` to skip autotuning entirely.

---

## Edge cases

- **Batch 1 OOMs** → raise `OutOfMemoryError` (model too big for this GPU).
- **`high` OOMs** → shrink from `high` toward `low`.
- **Flat or noisy memory** (fitted slope ≤ 0) → fall back to the largest probe that
  fit, with the safety factor applied.
- **Not GPU-bound** — the probe also checks system RAM. If RAM is the real limit
  (e.g. many dataloader workers), the shrink steps still find a safe size that the
  GPU-only model wouldn't have predicted.

---

## Worked example

PI0 (a large vision-language model) on a 40 GB A100:

```
probe 1   → 17.92 GB        fixed cost dominates
probe 8   → 18.30 GB        8x the batch, only +2% memory → slope unreliable
rough b_max ≈ 324           huge overshoot from the noisy small-batch slope
third = 162 → OOM           shrink: sqrt(8 · 162) ≈ 36
probe 36  → ~25.5 GB → fits
slope(8, 36) ≈ 245 MB/sample
predicted ≈ 73  → fits
chosen = floor(73 · 0.7) ≈ 51

Result: Using 30GB out of 40GB vram
```

About **5 probes** total, instead of the dozen a naive search would need — because
the third anchor is forced to land on a real, fitting, large batch.
