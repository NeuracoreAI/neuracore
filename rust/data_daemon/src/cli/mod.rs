//! Command-line interface.
//!
//! The `clap` command tree mirrors the Python `typer` CLI in
//! `neuracore/data_daemon/main.py` and `config_manager/args_handler.py`
//! exactly — same commands, flag names, aliases, and help strings — so that
//! `python -m neuracore.data_daemon <cmd>` behaves identically once the shim
//! hands off to this binary.

mod launch;
mod profile;
mod status;
mod stop;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::env::parse_bytes;

/// Neuracore Data Daemon CLI.
#[derive(Parser)]
#[command(
    name = "data-daemon",
    about = "Neuracore Data Daemon CLI.",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Launch the data daemon.
    Launch {
        /// Profile name to launch (from ~/.neuracore/data_daemon/profiles).
        #[arg(long)]
        profile: Option<String>,
        /// Run the daemon in the background without terminal output.
        #[arg(long)]
        background: bool,
        /// Enable debug mode.
        #[arg(long)]
        debug: bool,
    },
    /// Stop the data daemon.
    Stop,
    /// Show daemon status.
    Status,
    /// Install the data daemon as a system service.
    Install,
    /// Uninstall the data daemon system service.
    Uninstall,
    /// Manage daemon profiles.
    Profile {
        #[command(subcommand)]
        command: ProfileCommand,
    },
}

#[derive(Subcommand)]
enum ProfileCommand {
    /// Create a profile.
    Create {
        /// Profile name.
        name: String,
    },
    /// Update an existing profile.
    Update {
        /// Profile name to update.
        name: Option<String>,
        /// Storage limit in bytes.
        #[arg(long = "storage-limit", visible_alias = "storage_limit", value_parser = parse_bytes)]
        storage_limit: Option<i64>,
        /// Bandwidth limit in bytes per second.
        #[arg(long = "bandwidth-limit", visible_alias = "bandwidth_limit", value_parser = parse_bytes)]
        bandwidth_limit: Option<i64>,
        /// Path where records should be stored.
        #[arg(
            long = "storage-path",
            visible_aliases = ["storage_path", "path_to_store_record"]
        )]
        storage_path: Option<String>,
        /// Number of worker threads.
        #[arg(long = "num-threads", visible_alias = "num_threads")]
        num_threads: Option<i64>,
        /// Keep a wakelock while uploading.
        #[arg(long = "wakelock", overrides_with = "no_wakelock")]
        wakelock: bool,
        /// Do not keep a wakelock while uploading.
        #[arg(long = "no-wakelock", overrides_with = "wakelock")]
        no_wakelock: bool,
        /// Run in offline mode.
        #[arg(long = "offline", overrides_with = "online")]
        offline: bool,
        /// Run in online mode.
        #[arg(long = "online", overrides_with = "offline")]
        online: bool,
        /// API key used for authenticating the daemon.
        #[arg(long = "api-key", visible_alias = "api_key")]
        api_key: Option<String>,
        /// Active organisation ID for scoping daemon operations.
        #[arg(long = "current-org-id", visible_alias = "current_org_id")]
        current_org_id: Option<String>,
    },
    /// Get a profile's configuration.
    Get {
        /// Profile name to get.
        name: Option<String>,
    },
    /// Delete a profile.
    Delete {
        /// Profile name to delete.
        name: String,
    },
    /// List all configured daemon profiles.
    List,
}

/// Parse the process arguments and dispatch to the matching command handler.
///
/// Each handler is responsible for spinning up its own Tokio runtime when
/// needed; this keeps `launch --background` able to `fork` before the
/// multi-threaded runtime spawns worker threads (post-fork-with-threads is UB
/// on most libcs).
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Launch {
            profile,
            background,
            debug,
        } => launch::run(profile, background, debug),
        Command::Stop => stop::run(),
        Command::Status => status::run(),
        Command::Install => {
            // Preserves the Python stub behaviour for parity.
            println!("Install command is not implemented yet.");
            Ok(())
        }
        Command::Uninstall => {
            // Preserves the Python stub behaviour for parity.
            println!("Uninstall command is not implemented yet.");
            Ok(())
        }
        Command::Profile { command } => {
            profile::run(command);
            Ok(())
        }
    }
}
