//! Daemon configuration: the `DaemonConfig` model, profile storage, and the
//! profile + environment + CLI override merge.
//!
//! Mirrors `neuracore/data_daemon/config_manager/` — see that package for the
//! authoritative behaviour this Phase 1 port reproduces.

pub mod env;
pub mod profile;

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::profile::{ProfileError, ProfileManager};

/// Default profile name, matching `const.py::DEFAULT_PROFILE_NAME`.
pub const DEFAULT_PROFILE_NAME: &str = "default_profile";

// Defaults for a freshly built configuration, from `const.py`.
const DEFAULT_STORAGE_FREE_FRACTION: f64 = 0.5;
const DEFAULT_TARGET_DRAIN_HOURS: f64 = 12.0;
const DEFAULT_MIN_BANDWIDTH_MIB_S: f64 = 1.0;
const DEFAULT_MAX_BANDWIDTH_MIB_S: f64 = 20.0;
const SECONDS_PER_HOUR: f64 = 60.0 * 60.0;
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// Configuration options for a Neuracore data daemon instance.
///
/// Mirrors the Python `DaemonConfig` pydantic model: every field is optional
/// so partial profiles (e.g. a YAML file containing only `offline: true`) and
/// partial overrides round-trip cleanly. Field order matches the Python model
/// so that `profile get`'s JSON output is byte-compatible.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Maximum storage the daemon may use locally, in bytes.
    #[serde(default, deserialize_with = "env::deserialize_optional_bytes")]
    pub storage_limit: Option<i64>,
    /// Maximum upload bandwidth, in bytes per second.
    #[serde(default, deserialize_with = "env::deserialize_optional_bytes")]
    pub bandwidth_limit: Option<i64>,
    /// Directory where the daemon writes recording files.
    pub path_to_store_record: Option<String>,
    /// Number of worker threads used by the daemon.
    pub num_threads: Option<i64>,
    /// Whether to keep a wakelock while uploading data.
    pub keep_wakelock_while_upload: Option<bool>,
    /// When true, disable uploads and only store data locally.
    pub offline: Option<bool>,
    /// Neuracore API key for authentication.
    pub api_key: Option<String>,
    /// Organisation ID for the authenticated user.
    pub current_org_id: Option<String>,
}

impl DaemonConfig {
    /// Overlay `other`'s set fields on top of `self`.
    ///
    /// Mirrors pydantic's `model_copy(update=...)` as used by the Python
    /// `ConfigManager` and `ProfileManager`: a field is overwritten only when
    /// `other` provides a value for it, so `None` never clears an existing
    /// setting.
    pub fn overlay(&mut self, other: &DaemonConfig) {
        if other.storage_limit.is_some() {
            self.storage_limit = other.storage_limit;
        }
        if other.bandwidth_limit.is_some() {
            self.bandwidth_limit = other.bandwidth_limit;
        }
        if other.path_to_store_record.is_some() {
            self.path_to_store_record = other.path_to_store_record.clone();
        }
        if other.num_threads.is_some() {
            self.num_threads = other.num_threads;
        }
        if other.keep_wakelock_while_upload.is_some() {
            self.keep_wakelock_while_upload = other.keep_wakelock_while_upload;
        }
        if other.offline.is_some() {
            self.offline = other.offline;
        }
        if other.api_key.is_some() {
            self.api_key = other.api_key.clone();
        }
        if other.current_org_id.is_some() {
            self.current_org_id = other.current_org_id.clone();
        }
    }
}

/// Build a default daemon configuration based on local disk availability.
///
/// Mirrors `config_manager/helpers.py::build_default_daemon_config`: the
/// recordings directory is created if missing, the storage limit is set to a
/// fraction of free disk space, and the bandwidth limit is derived from it and
/// clamped to a sane range.
pub fn build_default_daemon_config() -> std::io::Result<DaemonConfig> {
    let record_dir = env::recordings_root_path();
    std::fs::create_dir_all(&record_dir)?;

    let free_bytes = free_disk_bytes(&record_dir)?;
    let storage_limit = (DEFAULT_STORAGE_FREE_FRACTION * free_bytes as f64) as i64;

    let raw_bandwidth = storage_limit as f64 / (DEFAULT_TARGET_DRAIN_HOURS * SECONDS_PER_HOUR);
    let min_bandwidth = (DEFAULT_MIN_BANDWIDTH_MIB_S * BYTES_PER_MIB) as i64;
    let max_bandwidth = (DEFAULT_MAX_BANDWIDTH_MIB_S * BYTES_PER_MIB) as i64;
    let bandwidth_limit = (raw_bandwidth as i64).clamp(min_bandwidth, max_bandwidth);

    Ok(DaemonConfig {
        storage_limit: Some(storage_limit),
        bandwidth_limit: Some(bandwidth_limit),
        path_to_store_record: Some(record_dir.to_string_lossy().into_owned()),
        num_threads: Some(1),
        keep_wakelock_while_upload: Some(false),
        offline: Some(false),
        api_key: None,
        current_org_id: None,
    })
}

/// Free bytes available to an unprivileged user on the filesystem holding
/// `path`, matching Python's `shutil.disk_usage(path).free`.
fn free_disk_bytes(path: &Path) -> std::io::Result<u64> {
    let stats = nix::sys::statvfs::statvfs(path).map_err(std::io::Error::from)?;
    let blocks_available: u64 = stats.blocks_available();
    let fragment_size: u64 = stats.fragment_size();
    Ok(blocks_available * fragment_size)
}

/// Resolve the effective daemon configuration from profile, environment, and
/// optional CLI overrides.
///
/// Mirrors `config_manager/config.py::ConfigManager.resolve_effective_config`:
/// the named profile (or the computed default when `profile` is `None`) is the
/// base, `NCD_*` environment variables are layered on top, and CLI overrides
/// win last.
pub fn resolve_effective_config(
    profiles: &ProfileManager,
    profile: Option<&str>,
    cli_overrides: Option<&DaemonConfig>,
) -> Result<DaemonConfig, ProfileError> {
    let mut config = profiles.get_profile(profile)?;
    config.overlay(&env::env_config_overrides());
    if let Some(cli) = cli_overrides {
        config.overlay(cli);
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_only_overwrites_set_fields() {
        let mut base = DaemonConfig {
            storage_limit: Some(100),
            offline: Some(false),
            api_key: Some("base-key".to_string()),
            ..DaemonConfig::default()
        };
        let overrides = DaemonConfig {
            storage_limit: Some(200),
            offline: None,
            api_key: None,
            num_threads: Some(4),
            ..DaemonConfig::default()
        };

        base.overlay(&overrides);

        assert_eq!(base.storage_limit, Some(200));
        assert_eq!(base.offline, Some(false));
        assert_eq!(base.api_key.as_deref(), Some("base-key"));
        assert_eq!(base.num_threads, Some(4));
    }

    #[test]
    fn json_output_keeps_python_field_order() {
        let config = DaemonConfig {
            storage_limit: Some(1),
            bandwidth_limit: Some(2),
            path_to_store_record: Some("/tmp/x".to_string()),
            num_threads: Some(1),
            keep_wakelock_while_upload: Some(false),
            offline: Some(false),
            api_key: None,
            current_org_id: None,
        };
        let json = serde_json::to_string_pretty(&config).unwrap();
        let keys: Vec<&str> = json
            .lines()
            .filter_map(|line| line.trim().strip_prefix('"'))
            .filter_map(|rest| rest.split('"').next())
            .collect();
        assert_eq!(
            keys,
            [
                "storage_limit",
                "bandwidth_limit",
                "path_to_store_record",
                "num_threads",
                "keep_wakelock_while_upload",
                "offline",
                "api_key",
                "current_org_id",
            ]
        );
    }
}
