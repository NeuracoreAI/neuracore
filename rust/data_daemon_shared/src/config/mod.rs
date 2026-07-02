//! Daemon configuration: the `DaemonConfig` model, profile storage, and the
//! profile + environment + CLI override merge.
//!
//! This module lives in the shared `data_daemon_shared` crate (rather than the
//! daemon binary) so the daemon **and** the PyO3 producer resolve the same
//! effective settings from the same profile/env inputs. The producer needs the
//! spool-backlog cap ([`resolve_spool_limit_bytes`]); resolving it here means
//! the two processes never drift on what the active profile says.

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

/// Default cap on the producer's on-disk video spool backlog, in bytes.
///
/// The producer spools raw-RGB NUT chunks to disk and the daemon transcodes
/// them; when the daemon can't keep up (small-CPU host, sustained 1080p video)
/// the un-encoded chunks would otherwise pile up unbounded — tens of GB — and
/// saturate a constrained disk, stalling `stop_recording`'s tail-chunk flush
/// for seconds. Bounding the spool backlog keeps the disk pressure flat. 2 GiB
/// is several 256 MiB chunks of headroom: large enough to absorb a transient
/// transcode stall without ever blocking the common case.
pub const DEFAULT_SPOOL_LIMIT_BYTES: i64 = 2 * 1024 * 1024 * 1024;

/// Configuration options for a Neuracore data daemon instance.
///
/// Every field is optional so partial profiles (e.g. a YAML file containing
/// only `offline: true`) and partial overrides round-trip cleanly. Field
/// order is fixed so that `profile get`'s JSON output is stable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Maximum storage the daemon may use locally, in bytes.
    #[serde(default, deserialize_with = "env::deserialize_optional_bytes")]
    pub storage_limit: Option<i64>,
    /// Maximum upload bandwidth, in bytes per second.
    #[serde(default, deserialize_with = "env::deserialize_optional_bytes")]
    pub bandwidth_limit: Option<i64>,
    /// Cap on the producer's on-disk video spool backlog, in bytes. When the
    /// un-encoded NUT backlog reaches this size the producer applies
    /// backpressure to video frame logging rather than letting the spool grow
    /// unbounded and fill the disk. See [`DEFAULT_SPOOL_LIMIT_BYTES`].
    #[serde(default, deserialize_with = "env::deserialize_optional_bytes")]
    pub spool_limit: Option<i64>,
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
    /// Global video codec selection (a Codec value, e.g. "h264_medium").
    pub video_codec: Option<String>,
}

impl DaemonConfig {
    /// Overlay `other`'s set fields on top of `self`.
    ///
    /// A field is overwritten only when `other` provides a value for it, so
    /// `None` never clears an existing setting.
    pub fn overlay(&mut self, other: &DaemonConfig) {
        if other.storage_limit.is_some() {
            self.storage_limit = other.storage_limit;
        }
        if other.bandwidth_limit.is_some() {
            self.bandwidth_limit = other.bandwidth_limit;
        }
        if other.spool_limit.is_some() {
            self.spool_limit = other.spool_limit;
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
        if other.video_codec.is_some() {
            self.video_codec = other.video_codec.clone();
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
        spool_limit: Some(DEFAULT_SPOOL_LIMIT_BYTES),
        path_to_store_record: Some(record_dir.to_string_lossy().into_owned()),
        num_threads: Some(1),
        keep_wakelock_while_upload: Some(false),
        offline: Some(false),
        api_key: None,
        current_org_id: None,
        video_codec: None,
    })
}

/// Free bytes available to an unprivileged user on the filesystem holding
/// `path`.
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

/// Resolve the effective spool-backlog cap (in bytes) for the **producer**,
/// which has no CLI args and must not trigger the directory-creating side
/// effects of the default-config build.
///
/// Precedence mirrors [`resolve_effective_config`] minus the CLI layer: the
/// `NCD_SPOOL_LIMIT` env override wins, then the active named profile's
/// `spool_limit` (`NEURACORE_DAEMON_PROFILE`), then [`DEFAULT_SPOOL_LIMIT_BYTES`].
/// A configured value of `0` is honoured verbatim and disables the bound. The
/// unnamed/default profile is deliberately *not* materialised here (that path
/// runs `build_default_daemon_config`, which creates the recordings dir and
/// stats the filesystem) — an unset profile simply falls through to the default.
pub fn resolve_spool_limit_bytes() -> i64 {
    if let Some(value) = env::env_config_overrides().spool_limit {
        return value;
    }
    if let Some(name) = env::active_profile_name() {
        if let Ok(config) = ProfileManager::new().get_profile(Some(&name)) {
            if let Some(value) = config.spool_limit {
                return value;
            }
        }
    }
    DEFAULT_SPOOL_LIMIT_BYTES
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
    fn overlay_sets_spool_limit_when_provided() {
        let mut base = DaemonConfig::default();
        base.overlay(&DaemonConfig {
            spool_limit: Some(1024),
            ..DaemonConfig::default()
        });
        assert_eq!(base.spool_limit, Some(1024));
        // A subsequent overlay without the field leaves it untouched.
        base.overlay(&DaemonConfig::default());
        assert_eq!(base.spool_limit, Some(1024));
    }

    #[test]
    fn json_output_keeps_python_field_order() {
        let config = DaemonConfig {
            storage_limit: Some(1),
            bandwidth_limit: Some(2),
            spool_limit: Some(3),
            path_to_store_record: Some("/tmp/x".to_string()),
            num_threads: Some(1),
            keep_wakelock_while_upload: Some(false),
            offline: Some(false),
            api_key: None,
            current_org_id: None,
            video_codec: None,
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
                "spool_limit",
                "path_to_store_record",
                "num_threads",
                "keep_wakelock_while_upload",
                "offline",
                "api_key",
                "current_org_id",
                "video_codec",
            ]
        );
    }
}
