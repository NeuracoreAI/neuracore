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
/// `--bandwidth-limit` options.
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

    Ok(base_value * multiplier)
}

/// Serde deserializer for byte-valued config fields that accepts either a
/// plain integer or a unit-suffixed string (e.g. `1G`).
///
/// Applies to byte-valued profile fields such as `storage_limit` and
/// `bandwidth_limit`. A malformed unit-suffixed string surfaces the parse
/// failure directly as a deserialization error.
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
/// non-UTF-8 bytes, **or is empty**. Treating empty as unset is the safe
/// default: a blank `NCD_API_KEY` / `NCD_CURRENT_ORG_ID` (common when a shell
/// exports an unset variable as the empty string) must not clobber a profile's
/// real value — it should fall through to the configured one.
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
    data_daemon_ipc::paths::db_path()
        .expect("home directory required to resolve the daemon database path")
}

/// Resolve the recordings root via the shared resolver (see [`db_path`]).
pub fn recordings_root_path() -> PathBuf {
    data_daemon_ipc::paths::recordings_root()
        .expect("home directory required to resolve the recordings root")
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
    /// Whether this process owns the PID file (`NEURACORE_DAEMON_MANAGE_PID`).
    pub manage_pid: bool,
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
            profile: env_var("NEURACORE_DAEMON_PROFILE").filter(|value| !value.is_empty()),
            // Mirrors `runtime.py`: default "1" (manage), "0" disables.
            manage_pid: env_var("NEURACORE_DAEMON_MANAGE_PID").as_deref() != Some("0"),
            // Mirrors `helpers.py::is_debug_mode`.
            debug: env_var("NDD_DEBUG")
                .map(|value| value.to_lowercase() == "true")
                .unwrap_or(false),
            api_url: env_var("NEURACORE_API_URL")
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| DEFAULT_API_URL.to_string()),
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
}
