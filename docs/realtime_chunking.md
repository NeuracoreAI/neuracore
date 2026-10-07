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
)

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
- Must satisfy `d <= s <= H - d`.
- Optional: `num_inference_steps`, `force_ddim`, `max_guidance_weight`.

Use `policy.benchmark(sync_point)` to time guided inference and size `d` / `s`
before going on hardware.

### Temporal ensemble (`TemporalEnsembleConfig`)

Async TE now drives the same :class:`~neuracore.ml.utils.temporal_ensemble.ACTTemporalEnsembler`
used by classic ACT (positive ``m`` favors **older** predictions):

- `execution_horizon` (`s`): actions per chunk before replan. Default **`1`**
  (predict every control tick). Larger ``s`` only updates the ensembler every
  ``s`` ticks.
- `m`: ACT exponential decay (default ``0.01``).
- `blend_steps`: optional continuity lerp of the chunk head (default ``0``;
  leave at 0 for ACT-matched behaviour).

Standalone use without the async controller:

```python
from neuracore.ml.utils.temporal_ensemble import ACTTemporalEnsembler

ens = ACTTemporalEnsembler(m=0.01, chunk_size=policy.prediction_horizon)
while running:
    chunk = policy._policy.predict_action_chunk(observation)  # (H, A)
    action = ens.update(chunk)  # (A,)
    send_to_robot(action)
```

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
