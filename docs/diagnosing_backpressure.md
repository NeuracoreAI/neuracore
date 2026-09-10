# Diagnosing Backpressure in the Data Daemon

This guide explains why the data daemon applies **backpressure** to video
logging, how to recognize it, and the concrete steps to fix it. It's aimed at
anyone running `log_rgb` / `log_depth` on a robot and hitting stalls, slow
`log_*` calls, or `LoggingStalledError`.

It does not explain internal implementation details. For general daemon setup and configuration, see
[`data_daemon.md`](data_daemon.md).

---

## Contents

- [Diagnosing Backpressure in the Data Daemon](#diagnosing-backpressure-in-the-data-daemon)
  - [Contents](#contents)
  - [What backpressure is](#what-backpressure-is)
  - [How to recognize it](#how-to-recognize-it)
  - [Why it happens](#why-it-happens)
  - [How to fix it](#how-to-fix-it)
    - [1) Increase the spool size](#1-increase-the-spool-size)
    - [2) Free up memory elsewhere on the machine](#2-free-up-memory-elsewhere-on-the-machine)
    - [3) Reduce camera / joint count, or resolution](#3-reduce-camera--joint-count-or-resolution)
  - [Baseline hardware recommendations](#baseline-hardware-recommendations)
  - [Last resort: wipe the spool and reset](#last-resort-wipe-the-spool-and-reset)
  - [Quick checklist](#quick-checklist)

---

## What backpressure is

Video frames logged with `log_rgb` / `log_depth` don't go straight to the
daemon over IPC, pixel data is too large for that. Instead, each producer
process writes raw frames to a small on-disk staging area called the
**spool** (`.rgb_spool/` under your recordings root). The daemon reads from
the spool, transcodes chunks to MP4 with `ffmpeg`, and uploads the result.

The spool is bounded (2 GiB by default) so a stalled daemon can never fill
your disk. When the daemon can't drain the spool as fast as frames are being
written into it, the un-encoded backlog grows toward that limit. Once it's
full, `log_rgb`/`log_depth` **blocks** for up to 1 second waiting for room,
and if the daemon still hasn't caught up, the call raises
`LoggingStalledError` instead of silently dropping the frame.

That blocking/rejecting behavior is backpressure: it's the system protecting
itself (and your disk) by refusing new video data faster than it can be
processed, rather than buffering it unboundedly in memory or on disk.

Backpressure is a **producer-vs-consumer rate problem**: it happens whenever
frames arrive faster than the daemon can encode + upload them, for long
enough to exhaust the spool. Fixing it means either giving the system more
runway (bigger spool, more headroom) or slowing the producer side down
(fewer streams, less resolution, less contention).

---

## How to recognize it

Look for any of the following:

- **`LoggingStalledError`** raised from `log_rgb` / `log_depth` in your
  script, with a message like:
  > `video logging stalled: the data daemon is not draining the spool
  > backlog (frame rejected after 1s of backpressure)`
- Your logging calls (`log_rgb`, `log_depth`) becoming noticeably slower or
  irregular, even though nothing else changed.
- Daemon logs containing `dropping frame`, `dropping item`, or `dropping
  sample`, these indicate the daemon is shedding load elsewhere (storage
  budget exceeded, publisher thread failures), which is a symptom of the same
  underlying resource pressure even when it isn't the spool itself.
- The `.rgb_spool/` directory under your recordings root (see
  `path_to_store_record` in [`data_daemon.md`](data_daemon.md)) growing
  continuously during a recording instead of staying roughly flat.

If you're seeing these, work through the steps below in order.

---

## Why it happens

Backpressure almost always comes down to one of three root causes:

1. **The spool is too small for your workload's burst size.** A short CPU
   spike (e.g. an OS scheduling hiccup, a background job) can cause a
   momentary encode backlog. If the spool is small, that backlog fills it
   before the daemon catches back up.
2. **The daemon (and `ffmpeg`) don't have enough CPU/memory headroom to keep
   up with encoding**, because something else on the machine, a logging
   script, a dashboard, another training/monitoring process, is competing
   for the same resources.
3. **You're asking the daemon to encode more data than the machine can
   sustain**, too many cameras, too high a resolution, or (less commonly)
   too many joint-logging threads.

The sections below map each of these to a specific fix.

---

## How to fix it

### 1) Increase the spool size

**What it does:** raises `spool_limit`, the cap on how much un-encoded video
the producer is allowed to buffer on disk before it applies backpressure.
Default is 2 GiB.

**Why this helps:** the spool is disk-backed, not memory-backed. Increasing
it gives the daemon more time to ride out a transient encode backlog
(a CPU spike, a slow disk write, a brief `ffmpeg` contention window) without
ever holding that backlog in RAM. Because it's disk space rather than
memory, you can afford a much larger buffer than you could ever justify
keeping in process memory, disk is cheap and plentiful compared to RAM, and
a bigger on-disk cushion doesn't compete with everything else running on the
machine for the same memory pool. In short: **use disk headroom to absorb
bursts instead of asking memory to do it.**

This is the right fix when backpressure is **occasional / bursty** rather
than sustained, if the daemon is *permanently* behind, a bigger spool just
delays the same failure (see steps 2 and 3 instead).

**How:**

```bash
# Update the default profile
neuracore data-daemon profile update --spool-limit 8589934592   # 8 GiB

# Or a named profile
neuracore data-daemon profile update my-profile --spool-limit 8589934592
```

Or via environment variable (bytes):

```bash
export NCD_SPOOL_LIMIT=8589934592   # 8 GiB
```

`0` disables the bound entirely (not recommended, you lose the disk-fill
safety net). Restart the daemon after changing this.

> Precedence: built-in default → profile YAML → environment variable → CLI
> flag. See [`data_daemon.md`](data_daemon.md) for the full table.

---

### 2) Free up memory elsewhere on the machine

**Why this matters:** the daemon shares the machine's CPU and memory with
everything else you're running. If another process, a heavy logging
script, a monitoring dashboard, a Jupyter notebook holding large arrays, a
second training job, is consuming most of the available memory or CPU, the
daemon and its `ffmpeg` encode workers get starved and fall behind, which
produces exactly the same symptoms as an undersized spool.

**This is the first thing to check if increasing the spool size didn't
help, or only delayed the problem.**

**How to check:**

```bash
# System-wide memory pressure
free -h

# Top consumers by memory/CPU
top -o %MEM
# or
htop
```

Look specifically for:
- Ad-hoc logging or plotting scripts running alongside the robot script
  (these are outside the daemon's scope, the daemon has no visibility into
  them and can't apply backpressure to protect itself from them).
- Multiple robot/recording processes running concurrently on the same
  machine, each maintaining their own spool and encode load.
- Swap usage (`free -h` showing non-zero `Swap` in use), this is a strong
  sign the machine is memory-constrained and everything, including `ffmpeg`,
  is running slower than it should.

**Fix:** stop or move unrelated intensive processes off the machine
running the data daemon, or reduce their footprint (e.g. don't hold full
camera frames in memory in a debug/plotting script when you don't need to;
downsample or subsample if you're just visualizing).

---

### 3) Reduce camera / joint count, or resolution

**Why this matters:** the daemon's encode capacity (roughly `cores / 2`
concurrent `ffmpeg` transcodes) and the producer's compression worker pool
are **shared across every camera stream you log**, not allocated per
camera. Adding more cameras, or higher-resolution/higher-framerate ones,
doesn't give you more capacity; it divides the same fixed capacity further.
Past a certain point, the aggregate data rate exceeds what the machine can
transcode in real time, and the spool fills no matter how big you make it.

Joint (and other scalar/JSON) logging is comparatively cheap and rarely the
cause of video backpressure, but a very large number of concurrent
logging threads across many joints/robots can add its own pressure, if
you're logging from an unusually high number of independent threads/robots
on one machine, consolidate them.

**How to reduce load:**

- Log only the cameras you actually need for this recording session.
- Lower resolution and/or frame rate on cameras where full fidelity isn't
  required.
- If you must log many high-resolution streams, spread them across multiple
  machines instead of one, or reduce concurrent robots-per-machine.
- Check `video_codec`, faster presets encode more cheaply at some quality
  cost; see [`data_daemon.md`](data_daemon.md) for supported values.

If reducing cameras/resolution isn't an option, this is generally a signal
you need more capable hardware, see baseline specs below, since the
system is being asked to sustain more encode throughput than any
configuration change alone can fix.

---

## Baseline hardware recommendations

These are starting points, not hard requirements, actual needs scale with
camera count/resolution/frame rate. If you're consistently hitting
backpressure even after applying the steps above, compare your machine
against this baseline before assuming it's a configuration problem.

| Resource | Minimum | Recommended for multi-camera (3+) |
|---|---|---|
| CPU cores | 4 | 8+ (encode concurrency defaults to `cores / 2`) |
| RAM | 8 GB | 16 GB+ |
| Disk | SSD, 20 GB free | SSD, 50 GB+ free (headroom for spool + recordings) |
| `ffmpeg` | Installed and on `PATH` | Same, hardware-accelerated encoder if available |

Notes:
- Disk **must** be reasonably fast (SSD, not a network mount or a slow USB
  drive), the spool is written and read continuously during recording, and
  a slow disk directly limits how fast the backlog can drain.
- Leave real headroom in each dimension. Running at 90%+ steady-state CPU or
  memory utilization on a "clean" machine means there's no slack left to
  absorb the bursts backpressure is designed to protect against.

---

## Last resort: wipe the spool and reset

If you don't care about losing recordings that are currently stuck (e.g.
you're mid-debugging and just want a clean slate), you can discard
everything and start fresh:

```bash
neuracore data-daemon reset --yes
```

This stops the daemon, then **permanently deletes**:
- The entire recordings root (including all spool chunks in `.rgb_spool/`
  and any recordings not yet uploaded)
- The daemon's local database
- Daemon process/IPC state files

> ⚠️ **This is destructive and unrecoverable.** Anything not already
> uploaded is lost. Only use this when you've decided any pending local
> recordings are not worth keeping, for example, after repeated
> `LoggingStalledError`s have already corrupted or interrupted whatever was
> being captured.

Use this to clear a wedged spool and confirm (by relaunching and re-running
a short test recording) that the daemon itself is healthy, before
re-attempting a real recording with the fixes above applied.

---

## Quick checklist

1. Are you seeing `LoggingStalledError` or a growing `.rgb_spool/`
   directory? → You have backpressure, keep going.
2. Is it occasional/bursty? → [Increase the spool size](#1-increase-the-spool-size).
3. Still stalling, or sustained rather than bursty? → [Check for other
   memory/CPU-hungry processes on the machine](#2-free-up-memory-elsewhere-on-the-machine).
4. Machine looks otherwise idle but still can't keep up? → [Reduce cameras,
   resolution, or frame rate](#3-reduce-camera--joint-count-or-resolution),
   and compare your hardware against the [baseline](#baseline-hardware-recommendations).
5. Need a clean slate right now and don't care about pending recordings?
   → [`neuracore data-daemon reset --yes`](#last-resort-wipe-the-spool-and-reset).
