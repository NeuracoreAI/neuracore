//! Neuracore data daemon — Rust rewrite.
//!
//! Entry point: parses the CLI synchronously and dispatches to a command
//! handler. The handler decides whether to spin up the Tokio runtime — most
//! subcommands (`status`, `stop`, `profile *`) do not need it, and the
//! `launch --background` path must `fork` *before* a multi-threaded runtime is
//! created (forking after spawning worker threads is undefined behaviour).
//!
//! See `docs/data-daemon-rewrite.md` for the full architecture and the
//! phased implementation plan. This is the Phase 2 entry point.

// The cloud subsystem and HTTP client expose a wide surface that the daemon
// binary consumes via the launch routine — `cargo check` would otherwise
// flag each unused item as dead code while phase 6 stabilises.
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
