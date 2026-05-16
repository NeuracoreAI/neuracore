//! `launch` subcommand handler.
//!
//! Phase 1 implements the launch *preflight* only: it resolves the runtime
//! environment and the effective configuration (profile + `NCD_*` env
//! overrides), validating an explicitly named profile exactly as the Python
//! `run_launch` does, then reports what the daemon process *would* run with.
//! Daemonization, the PID file, IPC, and the per-trace pipeline arrive in
//! later phases — see `docs/data-daemon-rewrite.md`.

use crate::config::env::RuntimeEnv;
use crate::config::profile::{ProfileError, ProfileManager};
use crate::config::{resolve_effective_config, DaemonConfig, DEFAULT_PROFILE_NAME};

/// Run the launch preflight.
pub fn run(profile: Option<String>, background: bool, debug: bool) {
    let runtime_env = RuntimeEnv::from_env();
    let profiles = ProfileManager::new();

    // Mirrors `run_launch`: an explicitly named profile must exist on disk.
    if let Some(name) = &profile {
        if let Err(error) = profiles.get_profile(Some(name)) {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }

    // Ensure the on-disk `default_profile.yaml` exists before resolving
    // config, mirroring `runtime.py::_resolve_configuration`: the Python
    // daemon best-effort creates the default profile on every launch so a
    // fresh install always resolves against a real profile file rather than
    // the in-memory computed defaults from `build_default_daemon_config()`.
    if let Err(error) = ensure_default_profile_exists(&profiles) {
        eprintln!("Failed to create default profile '{DEFAULT_PROFILE_NAME}': {error}");
        std::process::exit(1);
    }

    // `--profile` wins over the `NEURACORE_DAEMON_PROFILE` environment value,
    // which in turn wins over `DEFAULT_PROFILE_NAME` — matching
    // `runtime.py::_resolve_configuration`'s
    // `os.environ.get("NEURACORE_DAEMON_PROFILE") or DEFAULT_PROFILE_NAME`.
    let selected_profile = profile
        .or_else(|| runtime_env.profile.clone())
        .unwrap_or_else(|| DEFAULT_PROFILE_NAME.to_string());

    let config = match resolve_effective_config(&profiles, Some(&selected_profile), None) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    print_preflight(&runtime_env, &config, &selected_profile, background);

    if debug || runtime_env.debug {
        eprintln!("effective configuration: {config:#?}");
    }

    println!("Note: the daemon runtime is not yet implemented in this build.");
}

/// Create `DEFAULT_PROFILE_NAME` on disk if it does not already exist.
///
/// Mirrors the `create_profile(DEFAULT_PROFILE_NAME)` step in
/// `runtime.py::_resolve_configuration`: existing files are left untouched and
/// any other I/O failure propagates to the caller.
fn ensure_default_profile_exists(profiles: &ProfileManager) -> Result<(), ProfileError> {
    match profiles.create_profile(DEFAULT_PROFILE_NAME) {
        Ok(()) | Err(ProfileError::AlreadyExists(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Report the resolved environment and configuration the daemon would use.
fn print_preflight(
    runtime_env: &RuntimeEnv,
    config: &DaemonConfig,
    selected_profile: &str,
    background: bool,
) {
    let offline = config.offline.unwrap_or(false);

    println!("Daemon launch prepared.");
    println!("  profile:            {selected_profile}");
    println!("  offline:            {offline}");
    println!("  background:         {background}");
    println!("  pid file:           {}", runtime_env.pid_path.display());
    println!("  database:           {}", runtime_env.db_path.display());
    println!(
        "  recordings root:    {}",
        runtime_env.recordings_root.display()
    );
    println!("  api url:            {}", runtime_env.api_url);
    println!("  manage pid:         {}", runtime_env.manage_pid);
    println!("  max spooled chunks: {}", runtime_env.max_spooled_chunks);
}
