# Realtime overlapping-chunk execution

Action-chunking policies predict a horizon of `H` actions from one observation,
but a robot usually executes only `s` of them before replanning. If inference
takes non-negligible time, a naive replan can jump discontinuously relative to
the trajectory the robot is already following.

Neuracore’s realtime API runs the next chunk **in the background** while the
control loop keeps streaming actions from the current chunk. Two mutually
exclusive strategies share the same controller:

| Mode | What happens at replan | Typical models |
|------|------------------------|----------------|
| `rtc` | Guided denoising toward the previous chunk (arXiv:2506.07339) | Diffusion / flow-matching |
| `temporal_ensemble` | Unguided predict fused with ACT Algorithm 2 (`ACTTemporalEnsembler`; favors older) | Any chunking policy that can emit an action chunk |

They are **not** stacked. Pick one mode at construction.

## Prerequisites

```bash
pip install "neuracore[ml]"
```

RTC requires an in-process diffusion/flow model (autograd through the denoiser).
There is no remote-endpoint equivalent of `policy_realtime`.

## Minimal control loop

```python
import time
import neuracore as nc
from neuracore.ml.utils.real_time_chunking import RTCConfig
# from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

CONTROL_HZ = 50.0

policy = nc.policy_realtime(
    model_file="/path/to/model.nc.zip",  # or train_run_name=...
    mode="rtc",
    config=RTCConfig(inference_delay=4, execution_horizon=16),
    # mode="temporal_ensemble",
    # config=TemporalEnsembleConfig(execution_horizon=1, m=0.01),  # ACT every step
    control_hz=CONTROL_HZ,
    adapt_inference_delay=True,  # RTC only; ignored for temporal_ensemble
)

# On hardware: size d from measured latency, then replace_config (see below).

policy.start()
# Seed an observation so the inference thread can produce the first chunk.
# Hardware with Neuracore streams logged: policy.get_action()  # no arg
# Sim / offline: pass an explicit SynchronizedPoint.
policy.get_action(observation)  # or policy.get_action()
assert policy.wait_for_first_chunk(timeout=30.0)

try:
    while running:
        # Prefer logging sensors *before* get_action when using the fallback.
        action = policy.get_action()          # hardware
        # action = policy.get_action(observation)  # sim
        send_to_robot(action)
        time.sleep(1.0 / CONTROL_HZ)
finally:
    policy.request_stop()  # non-blocking; use stop() at process shutdown
```

### What `get_action` does (and does not do)

- Returns the **next action from the current chunk** (non-blocking; no inference).
- Stores the observation for the **next** replan (explicit arg, or
  `get_latest_sync_point()` when omitted).
- When `_index >= execution_horizon` (`s`), wakes the background thread to replan.

Most actions you receive were planned at an **earlier** replan. That is
intentional: the robot never pauses for inference.

```text
Tick:     0 … s-1 | s … s+d | …
Action:   chunk 0 | still 0 | chunk 1 …
Replan:           obs@s → inference → swap (~s+d for RTC; ~immediate for TE)
```

## Configuration

### RTC (`RTCConfig`)

- `inference_delay` (`d`): ticks to freeze/align with what already executed while
  the new chunk was being generated.
- `execution_horizon` (`s`): minimum actions consumed before replan; may grow
  with measured latency when `adapt_inference_delay=True`.
- Must satisfy `d <= s <= H - d` (equivalently `d <= s` and `d <= H - s`).
- `num_inference_steps`: denoising steps for the RTC path (default `10`). Keep
  this small enough that guided inference fits in the `H - s` buffer.
- `force_ddim` (default `True`): use deterministic DDIM even if the model was
  trained with DDPM. The frozen prefix must converge; a stochastic sampler
  can break the RTC alignment.
- `max_guidance_weight`: clip on guidance weight (`beta`; paper default `5`).

Pass `adapt_inference_delay=True` to `policy_realtime` (the default) so the
controller can grow `d` and `s` when measured latency exceeds the configured
delay. Set it `False` only when you want a fixed budget.

### Sizing `d` / `s` before hardware

Do not guess `d` on a real robot. Time guided inference on a representative
observation, set `d` from the **worst** sample, then validate the real-time
constraint before starting the controller:

```python
import math

H = policy.prediction_horizon
s = 16  # or int(H * 0.25), etc.
tick = 1.0 / CONTROL_HZ

# Provisional config so benchmark uses the intended denoise settings.
policy.replace_config(
    RTCConfig(inference_delay=1, execution_horizon=s),
    control_hz=CONTROL_HZ,
    adapt_inference_delay=True,
)

durations = policy.benchmark(observation, iterations=10)  # ascending seconds
d = max(1, math.ceil(durations[-1] / tick))  # worst-case ticks
assert d <= s and d <= H - s, (
    f"Real-time constraint violated: d={d}, s={s}, H-s={H - s}. "
    "Lower num_inference_steps, raise s, or lower CONTROL_HZ."
)

policy.replace_config(
    RTCConfig(inference_delay=d, execution_horizon=s),
    control_hz=CONTROL_HZ,
)
```

`replace_config` updates the session and drops any existing controller, so call
it **before** `start()` (or stop first if you are mid-session).

### Temporal ensemble (`TemporalEnsembleConfig`)

Async TE drives the same `ACTTemporalEnsembler` used by classic ACT (positive
`m` favors **older** predictions). Because inference runs in the background,
several control ticks can pass between updates; the controller tells the
ensembler how many, so the rows it fuses stay aligned to the same wall-clock
ticks rather than to the update count.

- `execution_horizon` (`s`): actions per chunk before replan. Default **`1`**
  (replan as often as possible). Larger `s` only updates the ensembler every
  `s` ticks, so fewer predictions are averaged per tick.
- `m`: ACT exponential decay (default `0.01`).
- `blend_steps`: optional continuity lerp of the chunk head (default `0`;
  leave at 0 for ACT-matched behaviour).
- `num_inference_steps`: sampler steps for models that have a step count
  (default `10`). Ignored by ACT and CNNMLP. **Do not leave this at the model
  default for a diffusion policy** — `DiffusionPolicy` ships with the offline
  default of 100 denoising steps, which cannot finish inside a control tick.
  `None` keeps whatever the model was built with.

With `s = 1` and inference that completes within one control tick, this is
exactly the synchronous predict-every-step ACT loop. As latency grows the
ensemble simply averages fewer, staler predictions per tick — it stays
correctly aligned, but the benefit shrinks.

Standalone use without the async controller:

```python
from neuracore.ml.utils.temporal_ensemble import ACTTemporalEnsembler

ens = ACTTemporalEnsembler(m=0.01, chunk_size=policy.prediction_horizon)
while running:
    chunk = predict_chunk(observation)  # (H, A), your own inference call
    # `advance` is the number of control ticks since the previous update; it is
    # always 1 in a synchronous predict-every-step loop like this one.
    action = ens.update(chunk, advance=1)  # (A,)
    send_to_robot(action)
```

## Monitoring (`stats()`)

After (or during) a run, inspect chunking health:

```python
stats = policy.stats()
print(
    f"chunks={stats.chunks} d={stats.inference_delay} "
    f"s={stats.execution_horizon} "
    f"deadline_misses={stats.deadline_misses} "
    f"stalled_ticks={stats.stalled_ticks} "
    f"median_latency_ms={stats.median_latency_s * 1e3:.1f}"
)
```

| Field | Meaning |
|-------|---------|
| `inference_delay` / `execution_horizon` | Current `d` / `s` (may have adapted) |
| `median_latency_s` | Typical replan time |
| `deadline_misses` | RTC chunks that landed after the planned `d` ticks |
| `stalled_ticks` | Ticks that repeated the last action because the chunk was exhausted |

Rising `deadline_misses` or `stalled_ticks` usually means inference is too slow
for the configured budget — raise `s`, lower `num_inference_steps`, or lower
`control_hz`.

## Example

See [`examples/example_realtime_chunking_vx300s.py`](../examples/example_realtime_chunking_vx300s.py)
for a MuJoCo Transfer Cube rollout that exercises both modes.

```bash
cd examples
python example_realtime_chunking_vx300s.py --train-run-name MyTrainingJob --mode rtc
python example_realtime_chunking_vx300s.py --model-file /path/to/model.nc.zip --mode temporal_ensemble
```

## Further reading

- Black, Galliker, Levine — *Real-Time Execution of Action Chunking Flow Policies*
  ([arXiv:2506.07339](https://arxiv.org/abs/2506.07339))
- Implementation: `neuracore/ml/utils/rtc_controller.py`,
  `neuracore/ml/utils/real_time_chunking.py`,
  `neuracore/ml/utils/temporal_ensemble.py`
