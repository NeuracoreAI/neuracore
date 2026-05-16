//! Environment-variable parsing: `NCD_*` configuration overrides and the
//! `NEURACORE_DAEMON_*` / `NDD_*` / `NEURACORE_*` runtime settings.
//!
//! Mirrors `config_manager/config.py`, `config_manager/helpers.py`,
//! `data_daemon/helpers.py`, and `data_daemon/const.py`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};

use crate::config::DaemonConfig;

/// Values treated as truthy for boolean environment variables, matching
/// `config.py::YES_CONFIRMATION`.
const YES_VALUES: [&str; 4] = ["1", "true", "yes", "y"];

/// Default backend API URL, from `const.py::API_URL`.
const DEFAULT_API_URL: &str = "https://api.neuracore.app/api";

/// Default spooled-chunk cap, from `data_bridge.py::DEFAULT_MAX_SPOOLED_CHUNKS`.
const DEFAULT_MAX_SPOOLED_CHUNKS: u32 = 128;

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
/// Mirrors the `parse_bytes` fix-up `ProfileManager.get_profile` applies to
/// `storage_limit` / `bandwidth_limit` when loading a profile YAML file, with
/// one intentional divergence: Python silently *skips* a malformed string and
/// leaves the raw value in the dict (pydantic then rejects it during
/// `DaemonConfig(**data)` validation, surfacing a less precise error). We
/// surface the parse failure directly here, which gives a clearer error
/// message but trips slightly sooner than the Python path.
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

/// Read an environment variable, returning `None` when it is unset or holds
/// non-UTF-8 bytes. An empty value is returned as `Some("")`, matching
/// Python's `os.getenv`.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
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
/// Mirrors Python's `Path.home()`. Panics only if the home directory cannot be
/// determined at all, which also fails the Python daemon.
pub(crate) fn home_dir() -> PathBuf {
    dirs::home_dir().expect("could not determine the user's home directory")
}

/// Expand a leading `~` in a path string against the home directory, matching
/// `pathlib.Path.expanduser`.
fn expand_user(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        return home_dir().join(stripped);
    }
    if path == "~" {
        return home_dir();
    }
    PathBuf::from(path)
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

/// Resolve the daemon SQLite database path.
///
/// Mirrors `helpers.py::get_daemon_db_path`: `NEURACORE_DAEMON_DB_PATH`
/// (with `~` expansion) or `~/.neuracore/data_daemon/state.db`.
pub fn db_path() -> PathBuf {
    match env_var("NEURACORE_DAEMON_DB_PATH").filter(|value| !value.is_empty()) {
        Some(value) => expand_user(&value),
        None => home_dir()
            .join(".neuracore")
            .join("data_daemon")
            .join("state.db"),
    }
}

/// Resolve the recordings root directory.
///
/// Mirrors `helpers.py::get_daemon_recordings_root_path`:
/// `NEURACORE_DAEMON_RECORDINGS_ROOT` or `<db_dir>/recordings`.
pub fn recordings_root_path() -> PathBuf {
    match env_var("NEURACORE_DAEMON_RECORDINGS_ROOT") {
        Some(value) => PathBuf::from(value),
        None => default_recordings_root(&db_path()),
    }
}

/// `<db_dir>/recordings` — the recordings root when no override is set.
fn default_recordings_root(db_path: &Path) -> PathBuf {
    db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("recordings")
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
    /// Spooled-chunk cap (`NCD_MAX_SPOOLED_CHUNKS`).
    pub max_spooled_chunks: u32,
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
            max_spooled_chunks: env_var("NCD_MAX_SPOOLED_CHUNKS")
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_MAX_SPOOLED_CHUNKS),
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
    fn expand_user_resolves_leading_tilde_only() {
        assert_eq!(expand_user("~"), home_dir());
        assert_eq!(expand_user("~/foo"), home_dir().join("foo"));
        assert_eq!(expand_user("/abs/path"), PathBuf::from("/abs/path"));
        assert_eq!(expand_user("rel/~/path"), PathBuf::from("rel/~/path"));
    }

    #[test]
    fn default_recordings_root_is_sibling_of_db() {
        assert_eq!(
            default_recordings_root(Path::new("/a/b/state.db")),
            PathBuf::from("/a/b/recordings")
        );
    }
}
