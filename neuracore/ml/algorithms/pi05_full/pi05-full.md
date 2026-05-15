# pi05_full — Subtask Generation and Knowledge Insulation for Pi0.5

**Status:** Approved design
**Date:** 2026-05-06
**Owner:** Ke Wang
**Implements:** `neuracore/ml/algorithms/pi05_full/`

## Goal

Add subtask generation and Knowledge Insulation (KI) training to Pi0.5 in Neuracore, alongside the existing `pi05` algorithm. The new variant lives in `neuracore/ml/algorithms/pi05_full/` and is selectable as a sibling algorithm to `pi05`.

## References

- JAX implementation: https://github.com/Ke-Wang1017/openpi_subtask
- Subtask training guide: https://github.com/Ke-Wang1017/openpi_subtask/blob/main/README_subtask.md
- Blog post: https://kewang1017.substack.com/p/implementing-the-full-capability
- PyTorch port (lerobot): https://github.com/cijerezg/lerobot/tree/main/src/lerobot/policies/pi05_full

## 1. Summary of differences vs. vanilla pi05

| Aspect | pi05 | pi05_full |
|---|---|---|
| Training losses | flow-matching MSE only | flow MSE + subtask CE + FAST CE |
| Prefix segments | `[images, language+state]` | `[images, language+state, subtask, fast_action_tokens]` |
| LM head usage | unused at training | supervises subtask + FAST tokens |
| VLM-to-action gradient flow | unrestricted | blocked via attention-level `detach()` (Knowledge Insulation) |
| Inference | flow-matching denoise from noise | autoregressive subtask generation → flow-matching conditioned on subtask |
| Required input data types | RGB, language, proprio (optional) | adds `SUBTASK_LANGUAGE` (required) |
| Output data types | joints, gripper | joints, gripper, **`SUBTASK_LANGUAGE`** |
| Two-stage curriculum | n/a | achievable via existing `finetune_action_expert_only` + loss-weight knobs |

The mechanical changes are:

1. **Three-loss training.** Each forward pass produces `flow_mse_loss`, `subtask_ce_loss`, `fast_token_ce_loss`, weighted-sum into `loss`. Subtask CE and FAST CE go through PaliGemma's existing tied LM head (no new parameters).

2. **FAST tokenization on the fly.** Target action chunks run through the `physical-intelligence/fast` HuggingFace tokenizer on CPU per training step to produce integer token IDs that map into the last 128 slots of the PaliGemma vocab.

3. **Prefix grows two segments.** `_embed_prefix` embeds, in order: image patches → language+state tokens → subtask tokens → FAST tokens. Custom 2D attention mask: subtask attends to images/language causally to itself; FAST attends to images/language/subtask causally to itself; the flow-matching suffix attends to images/language/**subtask** but **not** to FAST.

4. **Knowledge Insulation in attention.** When KI is enabled, action queries attend to `K_vlm.detach()` / `V_vlm.detach()`. Forward values unchanged; backward gradient from action losses cannot reach VLM parameters. Subtask CE and FAST CE losses still reach VLM normally via the VLM stream.

5. **Inference adds a subtask generation loop.** Before flow-matching denoise, the model autoregressively decodes subtask tokens from a BOS prefix (greedy by default, temperature configurable). KV cache from prefix is reused.

6. **`finetune_action_expert_only` tightened** to set `requires_grad=False` on VLM parameters (small fix to inherited code so Stage 2 saves backward compute and memory).


What does **not** change: pretrained weights initialization (`lerobot/pi05_base`), `transformers_replace` patches (adaRMS, gated residuals, SigLIP), image preprocessing, state discretization in prompt, time sampling, sinusoidal time embedding, Euler denoiser, optimizer/scheduler, gradient checkpointing, torch.compile path, output type plumbing for joints/gripper.

## 2. Architecture

### File map

`neuracore/ml/algorithms/pi05_full/` (currently a copy of pi05; will diverge):

```
__init__.py                  SAME (transformers patching)
gemma_pytorch.py             EDIT — add knowledge_insulation path in attention
modules.py                   EDIT — extend _embed_prefix; add subtask generation loop;
                                    add LM-head loss path; expose detailed losses
pi05.py                      EDIT — add SUBTASK_LANGUAGE to supported types;
                                    integrate FAST tokenizer; build subtask tokens;
                                    compute three losses in training_step;
                                    decode subtask text in forward()
utils.py                     EDIT — extend PI05Config (loss weights, KI flag, FAST cfg);
                                    add FAST tokenizer loader; subtask prompt helpers
requirements.txt             SAME — FAST tokenizer is a HF tokenizer, no extra dep
transformers_replace/        SAME
```

### Class hierarchy

```
NeuracoreModel (existing abstract)
└── Pi05Full (was Pi05; renamed only inside this folder)
    ├── owns: prompt_tokenizer (PaliGemma), language_decode_tokenizer,
    │         fast_tokenizer (NEW), proprio_normalizer, action_normalizer
    ├── owns: PI05FullPolicy (the inner nn.Module)
    └── implements: forward, training_step, configure_optimizers, configure_schedulers

PI05FullPolicy (was PI05Policy)
├── paligemma_with_expert: PaliGemmaWithExpertModel  ← KI-aware (modified)
├── action_in_proj, action_out_proj                  (unchanged)
├── time_mlp_in, time_mlp_out                        (unchanged)
├── methods:
│   ├── forward(...)                  extended to take subtask_tokens, fast_tokens
│   │                                  and return three-loss dict
│   ├── sample_actions(...)            unchanged signature; receives generated subtask
│   ├── generate_subtask_tokens(...)   NEW — autoregressive decode loop
│   └── _embed_prefix(...)             extended to handle subtask + FAST segments

PaliGemmaWithExpertModel (in gemma_pytorch.py)
└── attention layer gains a knowledge_insulation flag selecting between:
    - vanilla attention (current)
    - KI attention: action queries see K_vlm.detach(), V_vlm.detach()
```

### Module ownership boundaries

| Module | Owns |
|---|---|
| `pi05.py` | Data marshalling, normalization, prompt construction, FAST tokenization on host, loss aggregation, optimizer/scheduler config, decoded subtask output construction |
| `modules.py` | Embedding the prefix (4 segments), embedding the suffix (noisy actions), running fused VLM+expert forward, computing per-loss tensors, autoregressive subtask decode loop |
| `gemma_pytorch.py` | The KI branch in attention computation (one new code path; the surrounding wrapper stays the same) |
| `utils.py` | `PI05FullConfig` dataclass (extends pi05's PI05Config), FAST tokenizer loader, prompt-template helpers, attention mask helpers |

### Independence from `neuracore.ml.algorithms.pi05`

pi05_full is a self-contained sibling and **does not import** from `pi05`. The currently-shared imports (`pi05.gemma_pytorch`, `pi05.utils`) are rewritten to local `pi05_full` files because we modify `gemma_pytorch.py` and extend the config. Vanilla `pi05` stays unchanged.

### Three-loss return contract

`PI05FullPolicy.forward(...)` returns:

```python
{
    "flow_mse_loss":   Tensor,   # scalar
    "subtask_ce_loss": Tensor,   # scalar
    "fast_ce_loss":    Tensor,   # scalar
    "loss":            Tensor,   # weighted sum, used for backward
}
```

`Pi05Full.training_step` consumes this dict and returns `BatchedTrainingOutputs(losses=..., metrics=...)` with all four scalars surfaced in both fields so they're logged independently.

### New config fields (`PI05FullConfig`)

Beyond inherited `PI05Config` fields:

```python
# Loss weights (defaults match the JAX paper)
subtask_loss_weight:        float = 10.0
fast_token_loss_weight:     float = 1.0
flow_matching_loss_weight:  float = 1.0

# Knowledge insulation (training only)
knowledge_insulation: bool = True

# Subtask + FAST tokenization
max_subtask_tokens:    int = 64
max_fast_tokens:       int = 128
fast_tokenizer_name:   str = "physical-intelligence/fast"
fast_skip_tokens:      int = 128   # last 128 PaliGemma vocab slots reserved for FAST

# Subtask generation at inference
max_decoding_steps:    int = 200
subtask_temperature:   float = 0.0   # 0 = greedy
```

The `Pi05Full` model class also exposes these as `__init__` kwargs so they appear in the algorithm config surfaced by Neuracore's training UI.

## 3. Data flow

### Training step (per batch)

```
BatchedTrainingSamples
   inputs:  {RGB_IMAGES, LANGUAGE, SUBTASK_LANGUAGE, joint/gripper proprio}
   outputs: {joint targets / gripper targets}
        │
        ▼
Pi05Full.training_step(batch):
  1. _build_inputs_from_batch(batch)
     → images, image_masks, lang_tokens, lang_masks
  2. _process_subtask_tokens(batch)                       NEW
     → subtask_tokens [B, max_subtask_tokens=64], subtask_masks
       (last-timestep SUBTASK_LANGUAGE input_ids; BOS prepended; right-padded)
  3. _build_action_targets_and_fast_tokens(batch)         NEW
     - concat outputs → action_data [B, T, total_action_dim]
     - normalize → target_actions [B, T, max_action_dim=32]
     - fast_tokenize(target_actions)  (CPU)
       → fast_tokens [B, max_fast_tokens=128], fast_masks
  4. PI05FullPolicy.forward(images, image_masks,
                            lang_tokens, lang_masks,
                            subtask_tokens, subtask_masks,
                            fast_tokens, fast_masks,
                            target_actions)
     → {flow_mse_loss, subtask_ce_loss, fast_ce_loss, loss}
  5. Return BatchedTrainingOutputs(losses=..., metrics=...)
```

### Inside `PI05FullPolicy.forward(...)` at training time

```
_embed_prefix:
  images           → [B, N_img,    D]   bidirectional segment (att=0)
  language+state   → [B, L_lang,   D]   bidirectional segment (att=0)
  subtask_tokens   → [B, L_st,     D]   causal segment        (att=1)
  fast_tokens      → [B, L_fast,   D]   causal segment        (att=1)
  pad/att masks combined →
     - subtask attends to {images, language} bidir + itself causally
     - fast    attends to {images, language, subtask} bidir + itself causally
     - flow suffix attends to {images, language, subtask}, NOT fast

_embed_suffix:
  x_t = t·noise + (1-t)·actions  [B, chunk_size, 32]
  time-conditioning → adarms_cond
  action_in_proj(x_t)            → [B, chunk_size, D_expert]

PaliGemmaWithExpertModel.forward(..., knowledge_insulation=cfg.ki):
  Each transformer layer:
     - VLM stream computes attention as today
     - action stream attends to K_vlm.detach(), V_vlm.detach()
       (forward values unchanged; backward gradient never reaches VLM K/V proj)

LOSSES:
  flow_mse_loss   = MSE( action_out_proj(suffix_out), noise - actions )
                    masked to real action_dim, then mean

  subtask_logits  = lm_head( prefix_out[:, subtask_slice, :] )
  subtask_ce_loss = CE( subtask_logits[:, :-1], subtask_tokens[:, 1:] )
                    weighted by subtask_masks[:, 1:]; sum / mask.sum().clamp(min=1)

  fast_logits     = lm_head( prefix_out[:, fast_slice, :] )
  fast_ce_loss    = CE( fast_logits[:, :-1], fast_tokens[:, 1:] )
                    weighted by fast_masks[:, 1:]; sum / mask.sum().clamp(min=1)

  loss = w_flow·flow_mse_loss + w_subtask·subtask_ce_loss + w_fast·fast_ce_loss
```

### Inference (per call to `forward(batch)`)

```
1. _build_inputs_from_batch(batch)
       → images, image_masks, lang_tokens, lang_masks
   (no SUBTASK_LANGUAGE in inputs — model generates it)

2. PI05FullPolicy.generate_subtask_tokens(...)               NEW
   Phase A (prefill):
     embed_prefix with subtask_tokens=[BOS], no FAST
     forward with use_cache=True
     → past_key_values populated
   Phase B (autoregressive decode, t = 1 .. max_decoding_steps):
     embed single new token
     forward with cache extension
     logits = lm_head(out[:, -1, :])
     mask out PaliGemma <loc####> tokens
     next_token = greedy or sample(temperature)
     break when EOS reached for all batch items

3. PI05FullPolicy.sample_actions(..., subtask_tokens=generated)
   Embed prefix WITH subtask, WITHOUT FAST
   Prefill cache for [images, language, subtask]
   Euler denoise for num_inference_steps:
     embed_suffix(x_t, time)
     forward with past_key_values (cache cropped each step)
     v_t = action_out_proj(suffix_out[:, -chunk_size:])
     x_t = x_t + dt · v_t

4. Build outputs dict:
   - JOINT_TARGET_POSITIONS / GRIPPER predictions (same as pi05)
   - SUBTASK_LANGUAGE prediction:
       decode(generated_subtask_tokens) via prompt_tokenizer
       wrap in BatchedSubtaskLanguageData(input_ids, attention_mask)
```

### Subtask caching at inference (deferred)

Lerobot caches generated subtasks for ~2 seconds to amortize decode cost. Neuracore's `forward()` is per-invocation; caching would require either model-internal state (un-Neuracore-like) or upstream caching in `policy_inference.py`. **Out of scope for v1.**

## 4. Knowledge Insulation implementation

Standard attention computes `out = softmax(Q · K^T / √d) · V` with `Q, K, V = [Q_vlm | Q_action], [K_vlm | K_action], [V_vlm | V_action]`.

KI splits the call into two:

```
out_vlm    = attn(Q_vlm,    [K_vlm,         K_action], [V_vlm,         V_action])
out_action = attn(Q_action, [K_vlm.detach(),K_action], [V_vlm.detach(),V_action])
out        = cat([out_vlm, out_action], dim=seq)
```

Forward values are identical (`detach` is no-op forward). Backward: gradient from any loss flowing into `Q_action` cannot reach VLM K/V projections. Subtask CE and FAST CE losses still reach VLM normally because their gradient path goes through the VLM stream.

### Code touchpoint

Inside `PaliGemmaWithExpertModel.forward` (`gemma_pytorch.py`), the per-layer attention step gets a knowledge-insulation branch keyed on the `knowledge_insulation` flag. The branch:

1. Computes Q, K, V from per-stream weights as today (VLM weights produce VLM Q/K/V; expert weights produce action Q/K/V; concat along sequence dim).
2. Splits along `seq_dim` at `vlm_len = prefix_pad_masks.shape[1]` — already known from the embed step and plumbed through the forward as an explicit `prefix_len: int` argument.
3. Runs two attention calls as shown above.
4. Concatenates outputs along seq dim.
5. Resumes the standard layer flow (output projection, MLP, residual).

The existing 4D attention mask is sliced the same way at `vlm_len` so each call sees its correct mask slice.

### Per call-site behavior

| Call site | Stream lengths | KI applied? |
|---|---|---|
| Training: full forward (prefix + flow suffix) | VLM > 0, action = chunk_size | **Yes** when `cfg.knowledge_insulation=True` |
| Inference: subtask decode (autoregressive) | VLM > 0, action = 0 | No (collapses to vanilla) |
| Inference: flow denoise | VLM > 0, action = chunk_size | No (no gradient at inference) |

KI is genuinely training-only behavior. We short-circuit with `if not self.training: return vanilla_attention(...)` so inference is never penalized.

### Why custom attention rather than `transformers_replace` patches

1. The KI split needs awareness of `prefix_len`, a pi05_full-specific concept. Pushing it into transformers' Gemma attention would couple a generic library patch to a single algorithm's geometry.
2. transformers' Gemma attention has multiple kernels (eager / SDPA / FlashAttention). Patching all of them is more surface than wrapping at the layer-loop level.

### Stage 2 with KI redundant

When `finetune_action_expert_only=True`, VLM parameters get `requires_grad=False`. KI then has no effect because there's no gradient to block. We document this and ignore the `knowledge_insulation` flag during Stage 2 (a non-fatal warning surfaces if the user sets both).

## 5. Two-stage training

The JAX paper's two-stage Knowledge Insulation curriculum is achievable with no new code paths, only configuration:

### Stage 1 — VLM finetuning (~20k steps)

```yaml
flow_matching_loss_weight: 0.0       # disable flow MSE
subtask_loss_weight:       10.0
fast_token_loss_weight:    1.0
finetune_action_expert_only: false
knowledge_insulation:      false     # irrelevant since flow is off
```

Action expert receives no gradients (its only supervision is flow loss). Pretrained action expert from `lerobot/pi05_base` is preserved unchanged. VLM specializes in subtask + FAST prediction.

### Stage 2 — Action expert finetuning (~8k steps), resume from Stage 1 checkpoint

```yaml
flow_matching_loss_weight: 1.0
subtask_loss_weight:       0.0
fast_token_loss_weight:    0.0
finetune_action_expert_only: true   # also sets requires_grad=False on VLM
# knowledge_insulation: omitted — irrelevant when VLM is fully frozen
```

VLM frozen at autograd level. Action expert + projections train against flow MSE only.

### Joint training (default, ~40k steps)

```yaml
flow_matching_loss_weight: 1.0
subtask_loss_weight:       10.0
fast_token_loss_weight:    1.0
finetune_action_expert_only: false
knowledge_insulation:      true     # blocks flow gradients into VLM
```

All three losses; KI prevents flow gradients from corrupting VLM; subtask + FAST losses keep VLM grounded.

### A small Stage-1 inefficiency, accepted for v1

With `flow_matching_loss_weight=0` the action expert still does its forward pass on the noisy-action suffix. Gradients are zero so nothing updates, but we pay the FLOPs. We can short-circuit this later (~10 lines), saving ~20-30% Stage 1 wall-clock; not worth doing for v1.

## 6. Error handling and validation

### Construction-time

`Pi05Full.__init__` runs after `_validate_input_output_types`:

1. `DataType.SUBTASK_LANGUAGE` in `input_data_types` — error: "Pi05Full requires SUBTASK_LANGUAGE in inputs. Use the Pi05 algorithm if your dataset has no subtask annotations."
2. `DataType.SUBTASK_LANGUAGE` in `output_data_types` — same message, mentions outputs.
3. FAST tokenizer loads successfully — error mentions HF auth/network and the pinned model id explicitly.
4. Loss weights are non-negative floats — `ValueError`.
5. Stage-misconfiguration warnings (non-fatal):
   - `finetune_action_expert_only=True` with `subtask_loss_weight > 0` → warn.
   - Same for `fast_token_loss_weight > 0`.
   - All three loss weights == 0 → error.
6. `knowledge_insulation=True` with `finetune_action_expert_only=True` → warn (KI is no-op when VLM frozen).

### Training-time (per batch, kept minimal)

1. Subtask channel present in batch (defensive).
2. Subtask token length: truncate from the right at `max_subtask_tokens` with a single rate-limited warning if exceeded.
3. FAST output length: truncate at `max_fast_tokens` with at most one warning per training run; if length 0, zero the FAST mask for that sample and warn.
4. Action chunk shape: `RuntimeError` if not `[B, chunk_size, max_action_dim]` (programmer error).

### Inference-time

1. SUBTASK_LANGUAGE in `batch.inputs` at inference: log a warning, ignore (model generates its own).
2. Generation runaway: hard cap at `max_decoding_steps`, terminate.
3. Empty subtask decode: acceptable, flow matching falls back to image+language conditioning.

### Numerical safety

- CE divisor never zero: `mask.sum().clamp(min=1)`.
- bfloat16 mode: CE losses computed in float32.
- Mixed-batch FAST lengths handled by padding.

### Explicitly NOT validated

- Subtask quality (we trust user annotations).
- Subtask presence per timestep (only last timestep's subtask is used; documented).
- FAST tokenizer version (pinning deferred to implementation phase).

## 7. Testing strategy

Tests live under `tests/unit/ml/algorithms/pi05_full/` and cover only behavior new vs. pi05.

| Test | What it pins down |
|---|---|
| `test_construction_validation.py` | Errors on missing SUBTASK_LANGUAGE in/out, negative loss weights, all-zero weights. Warnings for misconfiguration combos. |
| `test_fast_tokenization.py` | Action chunk → FAST tokens in correct vocab range; mask shape; truncation behavior. |
| `test_subtask_token_pipeline.py` | Mixed-length BatchedSubtaskLanguageData → padded subtask_tokens with correct masks; BOS at position 0. |
| `test_three_loss_shapes.py` | Forward+backward on synthetic batch produces three finite scalar losses with `requires_grad=True`; weighted sum matches manual computation. |
| `test_knowledge_insulation_gradient_flow.py` | With KI on and only flow loss, VLM `k_proj.weight.grad == 0`. With KI off, same param has non-zero grad. With KI on and only subtask loss, same param has non-zero grad. |
| `test_finetune_action_expert_only.py` | Optimizer contains only expert+projection params; VLM `requires_grad=False`; VLM weights unchanged after a step. |
| `test_inference_subtask_generation.py` | `forward(batch)` returns SUBTASK_LANGUAGE; generation halts at EOS or max steps; output is decodable. |
| `test_inference_output_shapes.py` | Joint/gripper outputs match pi05 shapes; SUBTASK_LANGUAGE matches BatchedSubtaskLanguageData. |
| `test_stage_configurations.py` | Stage 1 and Stage 2 configs both build, both run a training step. |
| `test_dtype_handling.py` | bfloat16 mode: no NaN, embeddings cast correctly. |
| `test_training_step_smoke.py` | Synthetic batch → full training_step → finite loss with grad; four scalars in losses+metrics. |
| `test_pretrained_load.py` (slow) | `lerobot/pi05_base` loads cleanly; LM head ties; no missing keys for action expert. |

Out of scope for the suite: convergence/quality, subtask generation quality, full multi-stage training in CI.

### Fixtures

Existing pi05 fixtures reused. New fixtures: `subtask_language_stats`, `synthetic_subtask_batch`, `tiny_pi05_full_config` (gemma_300m variant for fast tests).

## 8. Defaults summary

| Setting | Default | Rationale |
|---|---|---|
| `subtask_loss_weight` | 10.0 | Matches JAX paper |
| `fast_token_loss_weight` | 1.0 | Matches JAX paper |
| `flow_matching_loss_weight` | 1.0 | Matches JAX paper |
| `knowledge_insulation` | True | KI is the headline feature |
| `max_subtask_tokens` | 64 | Empirical: subtasks are short |
| `max_fast_tokens` | 128 | Matches lerobot port |
| `fast_tokenizer_name` | `physical-intelligence/fast` | Standard FAST tokenizer |
| `fast_skip_tokens` | 128 | Last 128 PaliGemma vocab slots |
| `max_decoding_steps` | 200 | Matches lerobot port |
| `subtask_temperature` | 0.0 | Greedy (deterministic) |
| Pretrained checkpoint | `lerobot/pi05_base` | Same as pi05 |

## 9. Out of scope

- **Subtask annotation tooling** — users supply per-timestep subtask labels in their dataset. A separate spec can later add an LLM-based annotation script under `pi05_full/annotate.py`.
- **Subtask caching at inference** — would need policy-pipeline-level support; can be added as a wrapper later.
- **2-stage trainer with explicit stage flag** — the Stage 1 / Stage 2 / Joint configs above suffice; no `training_stage` enum needed.
- **Dataset-format pre-tokenization of FAST tokens** — on-the-fly tokenization is fast enough for v1.
- **Convergence/quality evaluation** — empirical validation lives outside this spec.
- **Subtask generation quality benchmarking** — separate concern.

## 10. Open questions deferred to implementation

- Pinning `physical-intelligence/fast` to a specific HF revision (chose default-resolved for v1; revisit if behavior drift surfaces).
- Whether to add a `skip_action_expert_forward_if_no_flow` perf shortcut for Stage 1 (defer; add only if profiling shows it matters).
- Optimal subtask + FAST max-length defaults; first iteration uses lerobot's values, may be tuned after first training run.
