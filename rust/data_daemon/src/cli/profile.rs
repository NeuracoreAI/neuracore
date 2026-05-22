//! `profile` subcommand handlers.
//!
//! Mirrors `config_manager/args_handler.py`: the `run_profile_*` functions,
//! including their exact stdout/stderr messages and exit codes.

use super::ProfileCommand;
use crate::config::profile::ProfileManager;
use crate::config::{DaemonConfig, DEFAULT_PROFILE_NAME};

/// Print `message` to stderr and exit with status 1, mirroring
/// `typer.echo(..., err=True)` followed by `raise typer.Exit(code=1)`.
fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// Dispatch a `profile` subcommand.
pub fn run(command: ProfileCommand) {
    let profiles = ProfileManager::new();
    match command {
        ProfileCommand::Create { name } => create(&profiles, &name),
        ProfileCommand::Update {
            name,
            storage_limit,
            bandwidth_limit,
            storage_path,
            num_threads,
            wakelock,
            no_wakelock,
            offline,
            online,
            api_key,
            current_org_id,
        } => {
            let updates = DaemonConfig {
                storage_limit,
                bandwidth_limit,
                path_to_store_record: storage_path,
                num_threads,
                keep_wakelock_while_upload: tristate(wakelock, no_wakelock),
                offline: tristate(offline, online),
                api_key,
                current_org_id,
            };
            update(&profiles, name.as_deref(), &updates);
        }
        ProfileCommand::Get { name } => get(&profiles, name.as_deref()),
        ProfileCommand::Delete { name } => delete(&profiles, &name),
        ProfileCommand::List => list(&profiles),
    }
}

/// Collapse a `--flag` / `--no-flag` pair into a tri-state `Option<bool>`:
/// unset stays `None`, otherwise the flag that `clap` left set (last one
/// wins) decides the value.
fn tristate(enabled: bool, disabled: bool) -> Option<bool> {
    if enabled {
        Some(true)
    } else if disabled {
        Some(false)
    } else {
        None
    }
}

fn create(profiles: &ProfileManager, name: &str) {
    match profiles.create_profile(name) {
        Ok(()) => println!("Created profile '{name}'."),
        Err(error) => fail(error),
    }
}

fn update(profiles: &ProfileManager, name: Option<&str>, updates: &DaemonConfig) {
    let name = name.unwrap_or(DEFAULT_PROFILE_NAME);
    match profiles.update_profile(name, updates) {
        Ok(_) => println!("Updated profile '{name}'."),
        Err(error) => fail(error),
    }
}

fn get(profiles: &ProfileManager, name: Option<&str>) {
    let name = name.unwrap_or(DEFAULT_PROFILE_NAME);
    match profiles.get_profile(Some(name)) {
        Ok(config) => match serde_json::to_string_pretty(&config) {
            Ok(json) => println!("{json}"),
            Err(error) => fail(error),
        },
        Err(error) => fail(error),
    }
}

fn delete(profiles: &ProfileManager, name: &str) {
    if name == DEFAULT_PROFILE_NAME {
        fail(format!(
            "Cannot delete default profile '{DEFAULT_PROFILE_NAME}'."
        ));
    }
    match profiles.delete_profile(name) {
        Ok(()) => println!("Deleted profile '{name}'."),
        Err(error) => fail(error),
    }
}

fn list(profiles: &ProfileManager) {
    let names = profiles.list_profiles();
    if names.is_empty() {
        println!("No profiles found.");
        return;
    }
    for name in names {
        println!("{name}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tristate_collapses_flag_pairs() {
        assert_eq!(tristate(false, false), None);
        assert_eq!(tristate(true, false), Some(true));
        assert_eq!(tristate(false, true), Some(false));
    }
}
