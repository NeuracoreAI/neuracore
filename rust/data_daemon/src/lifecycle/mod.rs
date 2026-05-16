//! Daemon lifecycle: PID file management, daemonization, signal handling, and
//! startup recovery from a previous unclean exit.
//!
//! Phase 2 of the rewrite (see `docs/data-daemon-rewrite.md`). The Phase 4 IPC
//! bring-up will plug into [`recovery::cleanup_stale_ipc`] and the readiness
//! gate used by `crate::cli::launch::run`.

pub mod daemonize;
pub mod pidfile;
pub mod recovery;
pub mod signals;
