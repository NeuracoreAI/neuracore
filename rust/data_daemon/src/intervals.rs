//! Centralised poll / tick cadences for the daemon's recurring background loops.
//!
//! Every cloud coordinator, the watchers, and the connection monitor drive their
//! work from a `tokio::time::interval` on one of these constants. Keeping them in
//! one place makes the daemon's timing budget legible at a glance and avoids the
//! per-module `POLL_INTERVAL` name collisions that arise when each loop defines
//! its own.
//!
//! Each loop still picks its own [`tokio::time::MissedTickBehavior`] at the call
//! site: `Delay` for the steady-state pollers (a missed tick simply slips), and
//! `Skip` for the flush / rescan safety-nets (only the next deadline matters).
//! Cadences internal to a single component — the dispatcher's interleaved
//! housekeeping, the trace-DB write-behind flush, the IPC drain decay, and the
//! CLI stop-wait — stay local to those modules rather than living here.

use std::time::Duration;

/// Org-id config poll: re-reads `config.json` so every coordinator observes org
/// changes (`login`, `set_organization`) within a second. The file is tiny and
/// the read is async, so a coarse re-parse each tick is cheaper to reason about
/// than mtime gating.
pub const ORG_CONFIG_POLL: Duration = Duration::from_secs(1);

/// Daemon-profile config poll: re-resolves the effective `DaemonConfig` (profile
/// YAML + env) so the trace actors and registration coordinator observe profile
/// changes — chiefly the video codec — from an in-memory copy rather than
/// re-reading the YAML per trace. Matches [`ORG_CONFIG_POLL`]; a `RefreshConfig`
/// command additionally forces an immediate re-resolve for the SDK path.
pub const CONFIG_POLL: Duration = Duration::from_secs(1);

/// Registration drain fallback: the coordinator is event-driven off the bus and
/// only falls back to this poll when the bus is quiet.
pub const REGISTRATION_POLL: Duration = Duration::from_millis(500);

/// Progress-report sweep: kept short so a freshly-uploaded recording reports
/// promptly; the sweep is cheap because settled recordings are filtered out
/// server-side.
pub const PROGRESS_TICK: Duration = Duration::from_secs(2);

/// Status-update flush: coalesces upload progress / complete updates into
/// batched backend writes, firing regardless of inbox load.
pub const STATUS_FLUSH: Duration = Duration::from_millis(100);

/// Uploader safety-net rescan: catches traces skipped while the upload semaphore
/// was full during a drain, without relying on bus events.
pub const UPLOAD_RESCAN: Duration = Duration::from_secs(5);

/// Recording-reaper sweep: reclamation only frees space already replicated to
/// the cloud, so a relaxed cadence keeps the scan off the hot path.
pub const RECORDING_RECLAIM: Duration = Duration::from_secs(60);

/// Connection health probe — matches the Python `connection_manager.py` cadence.
pub const CONNECTION_HEALTH_CHECK: Duration = Duration::from_secs(10);
