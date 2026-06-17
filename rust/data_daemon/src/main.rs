//! Neuracore data daemon.
//!
//! Entry point: parses the CLI synchronously and dispatches to a command
//! handler. The handler decides whether to spin up the Tokio runtime — most
//! subcommands (`status`, `stop`, `profile *`) do not need it, and the
//! `launch --background` path must `fork` *before* a multi-threaded runtime is
//! created (forking after spawning worker threads is undefined behaviour).

// The cloud subsystem and HTTP client expose a wide surface that the daemon
// binary consumes via the launch routine — `cargo check` would otherwise
// flag the not-yet-reachable items as dead code.
#[allow(dead_code)]
mod api;
mod cli;
#[allow(dead_code)]
mod cloud;
mod config;
#[allow(dead_code)]
mod connection;
mod encoding;
mod ipc;
mod lifecycle;
mod pipeline;
mod state;
mod storage;

use anyhow::Result;

fn main() -> Result<()> {
    cli::run()
}
