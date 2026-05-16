//! Neuracore data daemon — Rust rewrite.
//!
//! Entry point: bootstraps the Tokio runtime and dispatches the CLI. The
//! command tree mirrors the Python `typer` CLI in `neuracore/data_daemon`
//! exactly so that `python -m neuracore.data_daemon <cmd>` keeps working once
//! the shim in `__main__.py` hands off to this binary.
//!
//! See `docs/data-daemon-rewrite.md` for the full architecture and the
//! phased implementation plan. This is Phase 1: scaffolding, CLI parity, and
//! profile/environment configuration.

mod cli;
mod config;

use anyhow::Result;

fn main() -> Result<()> {
    // One multi-threaded Tokio runtime backs the whole daemon (see the
    // rewrite plan's "Threading & async model" section). Phase 1 command
    // handlers are synchronous, but later phases run the daemon loop here.
    let runtime = tokio::runtime::Builder::new_multi_thread().build()?;
    runtime.block_on(cli::run())
}
