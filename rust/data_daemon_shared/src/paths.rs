//! Filesystem-path resolution shared by the daemon and the producer.
//!
//! The producer runs in a *separate* process from the daemon, yet both must
//! agree on where recordings live: the producer spools NUT chunks under the
//! recordings root and the daemon encodes them from the same place. Rather than
//! each maintaining its own copy of the "env override → db-sibling →
//! `~/.neuracore` default" precedence (which can silently drift), both call the
//! resolvers here — so the two processes are guaranteed to compute the same
//! paths from the same inputs.
//!
//! Resolution is fallible: when a path can only come from the home directory and
//! the home directory cannot be determined (e.g. a headless container with no
//! `$HOME`), the caller gets a [`HomeDirUnavailable`] error to surface
//! appropriately — the daemon exits at startup, the producer raises a Python
//! exception — instead of panicking or silently falling back to a scratch dir
//! the other process would never look in.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Env var overriding the recordings root (highest precedence).
pub const RECORDINGS_ROOT_ENV: &str = "NEURACORE_DAEMON_RECORDINGS_ROOT";

/// Env var overriding the SQLite DB path; the recordings root defaults to its
/// `recordings` sibling.
pub const DB_PATH_ENV: &str = "NEURACORE_DAEMON_DB_PATH";

/// Raised when a path can only be resolved from the home directory and the home
/// directory cannot be determined.
#[derive(Debug, Error)]
#[error(
    "could not determine the user's home directory; \
     set {RECORDINGS_ROOT_ENV} (or {DB_PATH_ENV}) to an absolute path"
)]
pub struct HomeDirUnavailable;

/// An env var's value, or `None` when unset or empty (an empty override is
/// treated as "unset" so a blank var doesn't resolve to an empty path).
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// The user's home directory, or [`HomeDirUnavailable`].
fn home_dir() -> Result<PathBuf, HomeDirUnavailable> {
    dirs::home_dir().ok_or(HomeDirUnavailable)
}

/// Expand a leading `~` or `~/…` against the home directory; `~user` forms are
/// left unchanged.
fn expand_user(path: &str) -> Result<PathBuf, HomeDirUnavailable> {
    if let Some(stripped) = path.strip_prefix("~/") {
        return Ok(home_dir()?.join(stripped));
    }
    if path == "~" {
        return home_dir();
    }
    Ok(PathBuf::from(path))
}

/// Resolve the daemon SQLite database path: [`DB_PATH_ENV`] (with `~`
/// expansion) or `~/.neuracore/data_daemon/state.db`.
pub fn db_path() -> Result<PathBuf, HomeDirUnavailable> {
    match non_empty_env(DB_PATH_ENV) {
        Some(value) => expand_user(&value),
        None => Ok(home_dir()?
            .join(".neuracore")
            .join("data_daemon")
            .join("state.db")),
    }
}

/// Resolve the recordings root: [`RECORDINGS_ROOT_ENV`] if set, otherwise the
/// `recordings` sibling of [`db_path`]. Identical for the daemon and producer.
pub fn recordings_root() -> Result<PathBuf, HomeDirUnavailable> {
    if let Some(value) = non_empty_env(RECORDINGS_ROOT_ENV) {
        return Ok(PathBuf::from(value));
    }
    let db_path = db_path()?;
    Ok(db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("recordings"))
}

/// Sum the byte count of every regular file beneath `root`, recursively.
///
/// Returns 0 when `root` does not exist (the expected state before the
/// recordings tree is created) and silently skips entries it cannot `stat`.
/// Symlinks are neither followed nor counted, so the walk cannot cycle.
///
/// Shared because two callers need the same number from the same tree: the
/// daemon's storage budget sums the recordings root, and the producer's writer
/// sums its spool inbox to enforce the backlog cap.
pub fn directory_bytes(root: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(file_type) if file_type.is_dir() => stack.push(entry.path()),
                Ok(file_type) if file_type.is_file() => {
                    if let Ok(metadata) = entry.metadata() {
                        total = total.saturating_add(metadata.len());
                    }
                }
                _ => {}
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn directory_bytes_sums_nested_files_and_ignores_missing_roots() {
        let tempdir = TempDir::new().unwrap();
        let root = tempdir.path().join("recordings");

        // Missing root: zero, no error.
        assert_eq!(directory_bytes(&root), 0);

        let nested = root.join("source").join("sensor");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("chunk.nut"), vec![0u8; 1024]).unwrap();
        std::fs::write(root.join("top.bin"), vec![0u8; 32]).unwrap();

        assert_eq!(directory_bytes(&root), 1024 + 32);
    }

    // These tests mutate process-wide env vars, so they must not run
    // concurrently with each other; a single test drives the whole matrix.
    #[test]
    fn resolution_precedence() {
        let saved_root = std::env::var_os(RECORDINGS_ROOT_ENV);
        let saved_db = std::env::var_os(DB_PATH_ENV);

        // Explicit recordings-root override wins outright.
        std::env::set_var(RECORDINGS_ROOT_ENV, "/data/records");
        std::env::set_var(DB_PATH_ENV, "/var/lib/ncd/state.db");
        assert_eq!(recordings_root().unwrap(), PathBuf::from("/data/records"));

        // Empty override is treated as unset → falls through to the db sibling.
        std::env::set_var(RECORDINGS_ROOT_ENV, "");
        assert_eq!(
            recordings_root().unwrap(),
            PathBuf::from("/var/lib/ncd/recordings")
        );
        assert_eq!(db_path().unwrap(), PathBuf::from("/var/lib/ncd/state.db"));

        // Restore the environment for other tests.
        match saved_root {
            Some(value) => std::env::set_var(RECORDINGS_ROOT_ENV, value),
            None => std::env::remove_var(RECORDINGS_ROOT_ENV),
        }
        match saved_db {
            Some(value) => std::env::set_var(DB_PATH_ENV, value),
            None => std::env::remove_var(DB_PATH_ENV),
        }
    }

    #[test]
    fn expand_user_only_touches_leading_tilde() {
        assert_eq!(
            expand_user("/abs/path").unwrap(),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            expand_user("rel/~/path").unwrap(),
            PathBuf::from("rel/~/path")
        );
    }
}
