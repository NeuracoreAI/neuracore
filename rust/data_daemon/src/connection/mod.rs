//! Connection-state monitoring for the daemon.
//!
//! Phase 6b — runs a 10 s `HEAD /status/health` tick and publishes
//! [`DaemonEvent::ConnectionState`] transitions to the broadcast bus so the
//! upload coordinator can pause / resume on persistent network failures.

pub mod monitor;

#[allow(unused_imports)]
pub use monitor::{spawn_connection_monitor, ConnectionState, MonitorHandle};
