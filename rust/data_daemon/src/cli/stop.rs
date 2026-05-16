//! `stop` subcommand handler.
//!
//! The real stop sequence (SIGTERM, wait, escalate to SIGKILL, clean up the
//! PID file and IPC artefacts) is implemented in Phase 2 — daemon lifecycle.
//! See `docs/data-daemon-rewrite.md`.

/// Run the stop command.
pub fn run() {
    println!("Stop command is not implemented yet.");
}
