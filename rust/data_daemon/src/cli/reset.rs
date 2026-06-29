//! `reset` subcommand handler.
//!
//! Wipes every piece of host state the daemon owns so a wedged host can be
//! returned to a clean slate without hand-deleting paths: the recordings tree,
//! the SQLite state database (plus its WAL/SHM sidecars), the PID file, and the
//! iceoryx2 discovery files together with the `/dev/shm` shared-memory segments
//! backing them. This mirrors the host cleanup that
//! `rust/scripts/run_integration_tests.sh` performs before a fresh run.
//!
//! The daemon is stopped first: removing its state while it is running would
//! corrupt an in-flight recording, and the live daemon would immediately
//! re-create the files we just deleted.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use iceoryx2::config::Config;

use crate::cli::stop;
use crate::config::env::{db_path, pid_path, recordings_root_path};

/// Prefix iceoryx2 gives the Python producer's shared frame slots in
/// `/dev/shm` (`_NEURACORE_SHARED_SLOT_PREFIX` in
/// `data_daemon/lifecycle/runtime_recovery.py`).
const NEURACORE_SHM_PREFIX: &str = "neuracore-";

/// POSIX shared-memory mount where both iceoryx2 and the producer place their
/// segments on Linux.
const SHM_DIR: &str = "/dev/shm";

/// Run the reset command.
///
/// `assume_yes` (the `--yes` flag) skips the interactive confirmation for
/// scripted use; otherwise the operator must confirm at the prompt.
pub fn run(assume_yes: bool) -> Result<()> {
    if !assume_yes && !confirm()? {
        println!("Reset aborted; nothing was removed.");
        return Ok(());
    }
    // Stop first so nothing re-creates the state we are about to remove. `stop`
    // is idempotent and a no-op when no daemon is running.
    stop::run()?;

    println!("Resetting daemon state:");

    purge_path("recordings", &recordings_root_path());
    purge_database(&db_path());
    purge_path("pid file", &pid_path());
    purge_iceoryx_state();

    println!("Reset complete.");
    Ok(())
}

/// Prompt the operator to confirm the destructive reset, listing exactly what
/// will be removed so the blast radius is visible before they commit.
///
/// Returns `Ok(true)` only when the operator types `y`/`yes`. A non-interactive
/// stdin (piped or redirected) cannot answer, so the reset is refused with
/// guidance to re-run with `--yes` rather than silently proceeding or hanging
/// on a read that never arrives.
fn confirm() -> Result<bool> {
    let recordings_root = recordings_root_path();
    let recording_count = count_recordings(&recordings_root);

    println!("This permanently removes all data daemon state, including:");
    println!("  {recording_count} recording(s)");
    if !std::io::stdin().is_terminal() {
        eprintln!("Refusing to reset: stdin is not a terminal. Re-run with --yes to confirm.");
        return Ok(false);
    }

    print!("Continue? [y/N] ");
    std::io::stdout().flush()?;

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let answer = answer.trim().to_lowercase();
    Ok(answer == "y" || answer == "yes")
}

/// Count recording directories under `recordings_root` for the confirmation
/// summary.
///
/// Recordings are the numeric per-recording directories; hidden entries (the
/// `.rgb_spool` staging tree) and stray files are ignored. A missing or
/// unreadable root counts as zero — the summary is advisory, not a precondition.
fn count_recordings(recordings_root: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(recordings_root) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            !entry.file_name().to_string_lossy().starts_with('.')
                && entry.file_type().is_ok_and(|file_type| file_type.is_dir())
        })
        .count()
}

/// Remove the SQLite state database and its WAL-mode sidecars.
///
/// SQLite keeps the write-ahead log and shared-memory index alongside the main
/// file; an orphaned sidecar would otherwise resurrect a half-state, so they are
/// purged together with the database itself.
fn purge_database(database: &Path) {
    purge_path("database", database);
    for suffix in ["-wal", "-shm", "-journal"] {
        purge_path("database sidecar", &sidecar(database, suffix));
    }
}

/// Remove iceoryx2's discovery files and the `/dev/shm` segments backing them.
///
/// The root path and segment prefix are read from iceoryx2's global config so
/// they track any host configuration override rather than assuming the
/// `/tmp/iceoryx2` + `iox2_` defaults.
fn purge_iceoryx_state() {
    let config = Config::global_config();

    let root_path = PathBuf::from(config.global.root_path().to_string());
    purge_path("iceoryx2 state", &root_path);

    let iceoryx_prefix = config.global.prefix.to_string();
    purge_shm_segments("iceoryx2 shared memory", &iceoryx_prefix);
    purge_shm_segments("producer shared memory", NEURACORE_SHM_PREFIX);
}

/// Remove `/dev/shm` segments whose name starts with `prefix`.
///
/// POSIX shared memory has no recursive-remove primitive, so the segments are
/// swept by listing the mount and unlinking matches by name.
fn purge_shm_segments(label: &str, prefix: &str) {
    let entries = match std::fs::read_dir(SHM_DIR) {
        Ok(entries) => entries,
        Err(error) => {
            eprintln!("  ! could not read {SHM_DIR} for {label}: {error}");
            return;
        }
    };

    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(prefix) {
            purge_path(label, &entry.path());
        }
    }
}

/// Remove a file, directory tree, or symlink at `path`, reporting the outcome.
///
/// Cleanup is best-effort: a missing target is silently skipped and any other
/// failure is reported but never aborts the reset, so one undeletable artefact
/// cannot leave the rest of the state in place.
fn purge_path(label: &str, path: &Path) {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
        Err(error) => {
            eprintln!("  ! {label} ({}): {error}", path.display());
            return;
        }
    };

    // `symlink_metadata` does not follow links, so a symlink reports as a
    // non-directory and is unlinked with `remove_file` rather than followed.
    let result = if metadata.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };

    match result {
        Ok(()) => println!("  - removed {label}: {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!("  ! {label} ({}): {error}", path.display()),
    }
}

/// Append `suffix` to a path's filename (`state.db` + `-wal` -> `state.db-wal`).
fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn sidecar_appends_suffix_to_filename() {
        assert_eq!(
            sidecar(Path::new("/a/b/state.db"), "-wal"),
            PathBuf::from("/a/b/state.db-wal")
        );
    }

    #[test]
    fn count_recordings_counts_dirs_and_ignores_spool_files_and_missing() {
        // Missing root counts as zero.
        assert_eq!(count_recordings(Path::new("/no/such/root")), 0);

        let dir = tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("1")).unwrap();
        std::fs::create_dir_all(root.join("2")).unwrap();
        // The hidden spool tree and stray files must not be counted.
        std::fs::create_dir_all(root.join(".rgb_spool/robot")).unwrap();
        std::fs::write(root.join("state.db"), b"x").unwrap();

        assert_eq!(count_recordings(root), 2);
    }

    #[test]
    fn purge_path_removes_files_dirs_and_is_quiet_on_missing() {
        let dir = tempdir().unwrap();

        let file = dir.path().join("state.db");
        std::fs::write(&file, b"x").unwrap();
        purge_path("file", &file);
        assert!(!file.exists());

        let tree = dir.path().join("recordings");
        std::fs::create_dir_all(tree.join("1/RGB")).unwrap();
        purge_path("tree", &tree);
        assert!(!tree.exists());

        // Missing path must not panic or error.
        purge_path("missing", &dir.path().join("nope"));
    }
}
