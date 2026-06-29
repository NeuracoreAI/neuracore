//! Environment-variable parsing: `NCD_*` configuration overrides and the
//! `NEURACORE_DAEMON_*` / `NDD_*` / `NEURACORE_*` runtime settings.
//!
//! Mirrors `config_manager/config.py`, `config_manager/helpers.py`,
//! `data_daemon/helpers.py`, and `data_daemon/const.py`.

use std::path::PathBuf;

use serde::{Deserialize, Deserializer};

use crate::config::DaemonConfig;

/// Values treated as truthy for boolean environment variables, matching
/// `config.py::YES_CONFIRMATION`.
const YES_VALUES: [&str; 4] = ["1", "true", "yes", "y"];

/// Default backend API URL, from `const.py::API_URL`.
const DEFAULT_API_URL: &str = "https://api.neuracore.app/api";

/// Parse a byte quantity from an integer-or-unit-suffixed string.
///
/// Mirrors `config_manager/helpers.py::parse_bytes`. Supported units
/// (case-insensitive): `b`, `k`, `kb`, `m`, `mb`, `g`, `gb`. Also usable
/// directly as a `clap` value parser for the `--storage-limit` /
/// `--bandwidth-limit` / `--spool-limit` options.
pub fn parse_bytes(value: &str) -> Result<i64, String> {
    let normalized = value.trim().to_lowercase();

    if !normalized.is_empty()
        && normalized
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        return normalized
            .parse::<i64>()
            .map_err(|_| format!("Invalid byte value: '{value}'"));
    }

    let numeric_part: String = normalized.chars().filter(|c| c.is_ascii_digit()).collect();
    let unit_suffix: String = normalized.chars().filter(|c| !c.is_ascii_digit()).collect();

    if numeric_part.is_empty() || unit_suffix.is_empty() {
        return Err(format!("Invalid byte value: '{value}'"));
    }

    let base_value: i64 = numeric_part
        .parse()
        .map_err(|_| format!("Invalid byte value: '{value}'"))?;

    let multiplier: i64 = match unit_suffix.as_str() {
        "b" => 1,
        "k" | "kb" => 1024,
        "m" | "mb" => 1024 * 1024,
        "g" | "gb" => 1024 * 1024 * 1024,
        _ => return Err(format!("Unknown byte unit in value: '{value}'")),
    };

    base_value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("Byte value out of range: '{value}'"))
}

/// Serde deserializer for byte-valued config fields that accepts either a
/// plain integer or a unit-suffixed string (e.g. `1G`).
///
/// Applies to byte-valued profile fields such as `storage_limit`,
/// `bandwidth_limit`, and `spool_limit`. A malformed unit-suffixed string
/// surfaces the parse failure directly as a deserialization error.
pub fn deserialize_optional_bytes<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum IntOrString {
        Int(i64),
        Str(String),
    }

    match Option::<IntOrString>::deserialize(deserializer)? {
        None => Ok(None),
        Some(IntOrString::Int(value)) => Ok(Some(value)),
        Some(IntOrString::Str(text)) => parse_bytes(&text)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

/// Read an environment variable, returning `None` when it is unset, holds
/// non-UTF-8 bytes, **or is empty**. Every `NCD_*` / `NEURACORE_*` override
/// read through this helper treats an empty value as unset (so a shell that
/// exports an unset variable as the empty string falls through to the
/// configured profile value rather than clobbering it). This is most important
/// for the secret-bearing `NCD_API_KEY` / `NCD_CURRENT_ORG_ID`, but it applies
/// uniformly to the boolean and numeric overrides too — unlike Python's
/// `config.py`, which honours an empty string as a real override.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Whether an environment value should be treated as truthy.
fn is_truthy(value: &str) -> bool {
    YES_VALUES.contains(&value.to_lowercase().as_str())
}

/// Read `NCD_*` daemon-config overrides from the environment.
///
/// Mirrors `config.py::ConfigManager._read_env_overrides`: unparseable
/// numeric values are skipped (leaving the field unset) rather than failing.
pub fn env_config_overrides() -> DaemonConfig {
    let mut config = DaemonConfig::default();

    if let Some(value) = env_var("NCD_STORAGE_LIMIT") {
        if let Ok(parsed) = parse_bytes(&value) {
            config.storage_limit = Some(parsed);
        }
    }
    if let Some(value) = env_var("NCD_BANDWIDTH_LIMIT") {
        if let Ok(parsed) = parse_bytes(&value) {
            config.bandwidth_limit = Some(parsed);
        }
    }
    if let Some(value) = env_var("NCD_SPOOL_LIMIT") {
        if let Ok(parsed) = parse_bytes(&value) {
            config.spool_limit = Some(parsed);
        }
    }
    if let Some(value) = env_var("NCD_PATH_TO_STORE_RECORD") {
        config.path_to_store_record = Some(value);
    }
    if let Some(value) = env_var("NCD_NUM_THREADS") {
        if let Ok(parsed) = value.parse::<i64>() {
            config.num_threads = Some(parsed);
        }
    }
    if let Some(value) = env_var("NCD_KEEP_WAKELOCK_WHILE_UPLOAD") {
        config.keep_wakelock_while_upload = Some(is_truthy(&value));
    }
    if let Some(value) = env_var("NCD_OFFLINE") {
        config.offline = Some(is_truthy(&value));
    }
    if let Some(value) = env_var("NCD_API_KEY") {
        config.api_key = Some(value);
    }
    if let Some(value) = env_var("NCD_CURRENT_ORG_ID") {
        config.current_org_id = Some(value);
    }

    config
}

/// The active daemon profile name (`NEURACORE_DAEMON_PROFILE`), or `None` when
/// unset. The producer uses this to resolve profile-scoped settings (the spool
/// cap) without materialising the computed default profile.
pub fn active_profile_name() -> Option<String> {
    env_var("NEURACORE_DAEMON_PROFILE")
}

/// Home directory, used as the root for `~/.neuracore` paths.
///
/// Panics only if the home directory cannot be determined at all, in which
/// case the daemon cannot resolve any of its on-disk paths.
pub(crate) fn home_dir() -> PathBuf {
    dirs::home_dir().expect("could not determine the user's home directory")
}

/// Resolve the daemon PID file path.
///
/// Mirrors `helpers.py::get_daemon_pid_path`: `NEURACORE_DAEMON_PID_PATH` or
/// `~/.neuracore/daemon.pid`.
pub fn pid_path() -> PathBuf {
    match env_var("NEURACORE_DAEMON_PID_PATH") {
        Some(value) => PathBuf::from(value),
        None => home_dir().join(".neuracore").join("daemon.pid"),
    }
}

/// Resolve the daemon SQLite database path via the shared resolver, so the
/// daemon and the producer agree on its location. Panics with a clear message
/// when the home directory is required but unavailable — acceptable for the
/// daemon binary (it exits at startup before writing anything); the producer
/// surfaces the same condition as a Python error.
pub fn db_path() -> PathBuf {
    crate::paths::db_path().expect("home directory required to resolve the daemon database path")
}

/// Resolve the recordings root via the shared resolver (see [`db_path`]).
pub fn recordings_root_path() -> PathBuf {
    crate::paths::recordings_root().expect("home directory required to resolve the recordings root")
}

/// Resolved runtime environment: paths and flags read from
/// `NEURACORE_DAEMON_*`, `NDD_*`, and `NEURACORE_*` variables.
///
/// Consolidates the helpers in `data_daemon/helpers.py` and the env-derived
/// constants in `data_daemon/const.py`.
#[derive(Debug, Clone)]
pub struct RuntimeEnv {
    /// PID file path (`NEURACORE_DAEMON_PID_PATH`).
    pub pid_path: PathBuf,
    /// SQLite database path (`NEURACORE_DAEMON_DB_PATH`).
    pub db_path: PathBuf,
    /// Recordings root directory (`NEURACORE_DAEMON_RECORDINGS_ROOT`).
    pub recordings_root: PathBuf,
    /// Profile to launch with (`NEURACORE_DAEMON_PROFILE`).
    pub profile: Option<String>,
    /// Debug logging flag (`NDD_DEBUG`).
    pub debug: bool,
    /// Backend API base URL (`NEURACORE_API_URL`).
    pub api_url: String,
}

impl RuntimeEnv {
    /// Resolve the runtime environment from the current process environment.
    pub fn from_env() -> Self {
        RuntimeEnv {
            pid_path: pid_path(),
            db_path: db_path(),
            recordings_root: recordings_root_path(),
            profile: active_profile_name(),
            debug: env_var("NDD_DEBUG")
                .as_deref()
                .map(is_truthy)
                .unwrap_or(false),
            api_url: env_var("NEURACORE_API_URL").unwrap_or_else(|| DEFAULT_API_URL.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bytes_handles_plain_integers_and_units() {
        assert_eq!(parse_bytes("1024"), Ok(1024));
        assert_eq!(parse_bytes("  2048  "), Ok(2048));
        assert_eq!(parse_bytes("1b"), Ok(1));
        assert_eq!(parse_bytes("1K"), Ok(1024));
        assert_eq!(parse_bytes("2kb"), Ok(2048));
        assert_eq!(parse_bytes("1M"), Ok(1024 * 1024));
        assert_eq!(parse_bytes("3mb"), Ok(3 * 1024 * 1024));
        assert_eq!(parse_bytes("1G"), Ok(1024 * 1024 * 1024));
        assert_eq!(parse_bytes("2gb"), Ok(2 * 1024 * 1024 * 1024));
    }

    #[test]
    fn parse_bytes_rejects_invalid_values() {
        assert!(parse_bytes("").is_err());
        assert!(parse_bytes("abc").is_err());
        assert!(parse_bytes("12tb").is_err());
        assert!(parse_bytes("1 g").is_err());
    }

    #[test]
    fn is_truthy_recognises_only_yes_values() {
        for truthy in ["1", "true", "TRUE", "yes", "Yes", "y", "Y"] {
            assert!(is_truthy(truthy), "{truthy:?} should be truthy");
        }
        for falsy in ["0", "false", "no", "n", "", "enabled", "2"] {
            assert!(!is_truthy(falsy), "{falsy:?} should be falsy");
        }
    }

    #[test]
    fn deserialize_optional_bytes_accepts_int_unit_or_null() {
        #[derive(Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "deserialize_optional_bytes")]
            limit: Option<i64>,
        }
        let limit_of = |json: &str| serde_json::from_str::<Holder>(json).map(|holder| holder.limit);

        assert_eq!(limit_of(r#"{"limit": 1024}"#).unwrap(), Some(1024));
        assert_eq!(
            limit_of(r#"{"limit": "1G"}"#).unwrap(),
            Some(1024 * 1024 * 1024)
        );
        assert_eq!(limit_of(r#"{"limit": null}"#).unwrap(), None);
        assert_eq!(limit_of(r#"{}"#).unwrap(), None);
        assert!(
            limit_of(r#"{"limit": "not-a-size"}"#).is_err(),
            "a malformed unit string surfaces as a deserialization error"
        );
    }

    /// Every variable the env layer reads. Listed so the matrix test can
    /// save/restore them all (and so a future field is added here too).
    const MANAGED_VARS: &[&str] = &[
        "NCD_STORAGE_LIMIT",
        "NCD_BANDWIDTH_LIMIT",
        "NCD_SPOOL_LIMIT",
        "NCD_PATH_TO_STORE_RECORD",
        "NCD_NUM_THREADS",
        "NCD_KEEP_WAKELOCK_WHILE_UPLOAD",
        "NCD_OFFLINE",
        "NCD_API_KEY",
        "NCD_CURRENT_ORG_ID",
        "NEURACORE_DAEMON_PROFILE",
    ];

    #[test]
    fn env_config_overrides_reads_ncd_vars() {
        // Mutates process-wide env, so — per this crate's convention (see
        // `paths.rs::resolution_precedence`) — a single test drives the whole
        // matrix and saves/restores every variable it touches.
        let saved: Vec<(&str, Option<std::ffi::OsString>)> = MANAGED_VARS
            .iter()
            .map(|name| (*name, std::env::var_os(name)))
            .collect();

        // 1) Every override present and well-formed → every field populated.
        std::env::set_var("NCD_STORAGE_LIMIT", "2G");
        std::env::set_var("NCD_BANDWIDTH_LIMIT", "10mb");
        std::env::set_var("NCD_SPOOL_LIMIT", "512");
        std::env::set_var("NCD_PATH_TO_STORE_RECORD", "/srv/recordings");
        std::env::set_var("NCD_NUM_THREADS", "4");
        std::env::set_var("NCD_KEEP_WAKELOCK_WHILE_UPLOAD", "yes");
        std::env::set_var("NCD_OFFLINE", "1");
        std::env::set_var("NCD_API_KEY", "secret-key");
        std::env::set_var("NCD_CURRENT_ORG_ID", "org-42");
        std::env::set_var("NEURACORE_DAEMON_PROFILE", "lab");

        let config = env_config_overrides();
        assert_eq!(config.storage_limit, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(config.bandwidth_limit, Some(10 * 1024 * 1024));
        assert_eq!(config.spool_limit, Some(512));
        assert_eq!(
            config.path_to_store_record.as_deref(),
            Some("/srv/recordings")
        );
        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.keep_wakelock_while_upload, Some(true));
        assert_eq!(config.offline, Some(true));
        assert_eq!(config.api_key.as_deref(), Some("secret-key"));
        assert_eq!(config.current_org_id.as_deref(), Some("org-42"));
        assert_eq!(active_profile_name().as_deref(), Some("lab"));

        // 2) An empty string is treated as unset; an unparseable numeric is
        //    skipped (not fatal) and leaves its field unset.
        std::env::set_var("NCD_API_KEY", "");
        std::env::set_var("NCD_OFFLINE", "");
        std::env::set_var("NCD_STORAGE_LIMIT", "not-a-size");
        std::env::set_var("NCD_NUM_THREADS", "twelve");
        let config = env_config_overrides();
        assert_eq!(config.api_key, None, "empty string is treated as unset");
        assert_eq!(config.offline, None, "empty string is treated as unset");
        assert_eq!(
            config.storage_limit, None,
            "an unparseable size is skipped, not fatal"
        );
        assert_eq!(
            config.num_threads, None,
            "an unparseable thread count is skipped"
        );

        // 3) Nothing set → an all-default (empty) override layer.
        for name in MANAGED_VARS {
            std::env::remove_var(name);
        }
        let config = env_config_overrides();
        assert_eq!(config.storage_limit, None);
        assert_eq!(config.bandwidth_limit, None);
        assert_eq!(config.spool_limit, None);
        assert_eq!(config.path_to_store_record, None);
        assert_eq!(config.num_threads, None);
        assert_eq!(config.keep_wakelock_while_upload, None);
        assert_eq!(config.offline, None);
        assert_eq!(config.api_key, None);
        assert_eq!(config.current_org_id, None);
        assert_eq!(active_profile_name(), None);

        // Restore the pre-test environment for other tests in this binary.
        for (name, value) in saved {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}
