//! Connection-state monitoring for the daemon.
//!
//! Runs a 10 s `HEAD /status/health` tick and publishes
//! [`DaemonEvent::ConnectionState`] transitions to the broadcast bus so the
//! upload coordinator can pause / resume on persistent network failures.

pub mod monitor;
pub mod wakelock;

#[allow(unused_imports)]
pub use monitor::{spawn_connection_monitor, ConnectionState, MonitorHandle};
#[allow(unused_imports)]
pub use wakelock::{spawn_wakelock, WakelockHandle};
