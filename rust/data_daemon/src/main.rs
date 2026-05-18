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

mod cli;
mod config;
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
