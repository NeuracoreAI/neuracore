//! Storage-budget tracking for the per-trace writers.
//!
//! Mirrors `recording_encoding_disk_manager/core/storage_budget.py`. Two
//! independent limits gate every write:
//!
//! - The configured `storage_limit_bytes` (from the active profile) caps how
//!   much room the daemon may consume under `recordings_root`. The tracker
//!   keeps an estimate that is refreshed by a full directory scan no more
//!   often than `refresh_seconds`.
//! - `min_free_disk_bytes` is the safety margin the daemon keeps free on the
//!   underlying filesystem. Defaults to `MIN_FREE_DISK_BYTES = 32 MiB` from
//!   the Python `const.py`.
//!
//! Phase 5a only exposes the policy + outcome enum used by the trace actor
//! when it lands in 5f; the storage-budget refresh loop and the warning event
//! it emits onto the daemon event bus are wired up in phase 6 alongside the
//! upload coordinator.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::paths::directory_bytes;

/// Safety margin matching Python `const.py::MIN_FREE_DISK_BYTES`.
pub const MIN_FREE_DISK_BYTES: u64 = 32 * 1024 * 1024;

/// Refresh interval matching Python `const.py::STORAGE_REFRESH_SECONDS`.
pub const STORAGE_REFRESH_SECONDS: f64 = 5.0;

/// Storage-budget configuration.
///
/// `storage_limit_bytes = None` disables the in-tree usage cap (matching
/// today's behaviour when the operator clears `storage_limit` in the
/// profile). The free-disk safety margin always applies.
#[derive(Debug, Clone, Copy)]
pub struct StoragePolicy {
    /// Maximum bytes the daemon may consume under the recordings root.
    pub storage_limit_bytes: Option<u64>,
    /// Minimum bytes that must remain free on the underlying filesystem.
    pub min_free_disk_bytes: u64,
    /// Maximum age of the cached used-bytes estimate before a rescan.
    pub refresh_interval: Duration,
}

impl Default for StoragePolicy {
    fn default() -> Self {
        Self {
            storage_limit_bytes: None,
            min_free_disk_bytes: MIN_FREE_DISK_BYTES,
            refresh_interval: Duration::from_secs_f64(STORAGE_REFRESH_SECONDS),
        }
    }
}

/// Outcome of a budget check.
///
/// Mirrors the binary "may I write" decision the Python `StorageBudget`
/// exposes via `reserve` / `has_free_disk_for_write`, but folds in the reason
/// so the per-trace actor can emit a useful tracing log line and the upload
/// coordinator can pick the right backpressure response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetCheck {
    /// The write is within both the storage limit and the free-disk margin.
    Available,
    /// The configured `storage_limit_bytes` would be exceeded.
    StorageLimitExceeded {
        /// Bytes the writer asked for.
        requested: u64,
        /// Current used-bytes estimate.
        used: u64,
        /// Configured cap.
        limit: u64,
    },
    /// The filesystem free-byte safety margin would be breached.
    FilesystemFull {
        /// Bytes the writer asked for.
        requested: u64,
        /// Free bytes reported by `statvfs`.
        free: u64,
        /// Safety margin from the policy.
        min_free: u64,
    },
}

impl BudgetCheck {
    /// True when the writer is cleared to proceed.
    pub fn is_available(self) -> bool {
        matches!(self, BudgetCheck::Available)
    }
}

/// Errors raised when interrogating the underlying filesystem.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// `statvfs` failed on the recordings root or one of its ancestors.
    #[error("failed to query filesystem at {path}: {source}")]
    Statvfs {
        /// Path passed to `statvfs`.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Storage-budget tracker.
///
/// Each method is thread-safe; internal state lives behind a `Mutex` so the
/// per-trace actors can reserve from a single shared instance without an
/// async hop. The estimate is updated optimistically on `reserve` and
/// reconciled by [`refresh_if_stale`](Self::refresh_if_stale) — the same
/// pattern as the Python implementation.
pub struct StorageBudget {
    recordings_root: PathBuf,
    policy: StoragePolicy,
    state: Mutex<BudgetState>,
}

struct BudgetState {
    used_bytes: u64,
    last_refresh: Instant,
}

impl StorageBudget {
    /// Open a budget tracker rooted at `recordings_root`.
    ///
    /// Runs an initial directory scan so the first `reserve` call sees an
    /// accurate baseline. The recordings root does not need to exist yet —
    /// the scan returns zero in that case.
    pub fn new(recordings_root: impl Into<PathBuf>, policy: StoragePolicy) -> Self {
        let recordings_root = recordings_root.into();
        let used_bytes = directory_bytes(&recordings_root);
        Self {
            recordings_root,
            policy,
            state: Mutex::new(BudgetState {
                used_bytes,
                last_refresh: Instant::now(),
            }),
        }
    }

    /// Borrow the recordings root used to seed this budget tracker.
    pub fn recordings_root(&self) -> &Path {
        &self.recordings_root
    }

    /// Borrow the active policy.
    pub fn policy(&self) -> &StoragePolicy {
        &self.policy
    }

    /// Current used-bytes estimate (may be stale; call
    /// [`refresh_if_stale`](Self::refresh_if_stale) for an accurate read).
    pub fn used_bytes(&self) -> u64 {
        self.state.lock().expect("budget state").used_bytes
    }

    /// Rescan the recordings tree if the estimate is older than
    /// `refresh_interval`.
    pub fn refresh_if_stale(&self) {
        let refresh_interval = self.policy.refresh_interval;
        if refresh_interval.is_zero() {
            return;
        }
        // The scan can be slow on large trees; do it outside the lock so
        // other reservers aren't blocked on the I/O.
        let needs_refresh = {
            let state = self.state.lock().expect("budget state");
            state.last_refresh.elapsed() >= refresh_interval
        };
        if !needs_refresh {
            return;
        }
        let scanned = directory_bytes(&self.recordings_root);
        let mut state = self.state.lock().expect("budget state");
        state.used_bytes = scanned;
        state.last_refresh = Instant::now();
    }

    /// Check (without committing) whether `bytes_to_write` would fit.
    pub fn check(&self, bytes_to_write: u64) -> Result<BudgetCheck, BudgetError> {
        self.refresh_if_stale();

        let free = free_disk_bytes(&self.recordings_root)?;
        if free < bytes_to_write.saturating_add(self.policy.min_free_disk_bytes) {
            return Ok(BudgetCheck::FilesystemFull {
                requested: bytes_to_write,
                free,
                min_free: self.policy.min_free_disk_bytes,
            });
        }

        if let Some(limit) = self.policy.storage_limit_bytes {
            let used = self.used_bytes();
            if used.saturating_add(bytes_to_write) > limit {
                return Ok(BudgetCheck::StorageLimitExceeded {
                    requested: bytes_to_write,
                    used,
                    limit,
                });
            }
        }

        Ok(BudgetCheck::Available)
    }

    /// Reserve `bytes_to_write` against the in-tree usage cap.
    ///
    /// Returns the same enum as [`check`](Self::check), but mutates the
    /// internal estimate when the result is [`BudgetCheck::Available`] so
    /// repeated calls add up across writers. The filesystem free-byte check
    /// is best-effort: when it fails (e.g. `statvfs` reports a transient
    /// error) the reservation is denied as if the disk were full, mirroring
    /// the Python `has_free_disk_for_write` fail-closed behaviour.
    pub fn reserve(&self, bytes_to_write: u64) -> Result<BudgetCheck, BudgetError> {
        let check = self.check(bytes_to_write)?;
        if let BudgetCheck::Available = check {
            let mut state = self.state.lock().expect("budget state");
            state.used_bytes = state.used_bytes.saturating_add(bytes_to_write);
        }
        Ok(check)
    }

    /// Release `bytes_to_release` from the in-tree usage estimate, e.g. after
    /// a recording is deleted post-upload.
    pub fn release(&self, bytes_to_release: u64) {
        let mut state = self.state.lock().expect("budget state");
        state.used_bytes = state.used_bytes.saturating_sub(bytes_to_release);
    }
}

/// Free bytes available on the filesystem holding `path`.
///
/// Walks up the directory tree until it finds an existing ancestor, which
/// matches the Python helper's `try: ... except FileNotFoundError: mkdir(...)`
/// safety net without actually creating directories.
fn free_disk_bytes(path: &Path) -> Result<u64, BudgetError> {
    let mut probe = path.to_path_buf();
    loop {
        match nix::sys::statvfs::statvfs(probe.as_path()) {
            Ok(stats) => {
                let blocks_available: u64 = stats.blocks_available();
                let fragment_size: u64 = stats.fragment_size();
                return Ok(blocks_available.saturating_mul(fragment_size));
            }
            Err(errno) => {
                if let Some(parent) = probe.parent() {
                    if parent != probe.as_path() {
                        probe = parent.to_path_buf();
                        continue;
                    }
                }
                return Err(BudgetError::Statvfs {
                    path: path.to_path_buf(),
                    source: std::io::Error::from(errno),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn policy_with_limit(limit: Option<u64>) -> StoragePolicy {
        StoragePolicy {
            storage_limit_bytes: limit,
            // Set the safety margin to zero so the test focuses on the
            // in-tree cap; the free-disk arm has its own test below.
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        }
    }

    #[test]
    fn reserve_accumulates_then_blocks_at_limit() {
        let tempdir = TempDir::new().unwrap();
        let budget = StorageBudget::new(tempdir.path(), policy_with_limit(Some(4096)));

        assert_eq!(budget.reserve(1024).unwrap(), BudgetCheck::Available);
        assert_eq!(budget.reserve(2048).unwrap(), BudgetCheck::Available);
        assert_eq!(budget.used_bytes(), 3072);

        let blocked = budget.reserve(2048).unwrap();
        assert!(
            matches!(
                blocked,
                BudgetCheck::StorageLimitExceeded {
                    requested: 2048,
                    used: 3072,
                    limit: 4096
                }
            ),
            "expected storage-limit exhaustion, got {blocked:?}"
        );

        budget.release(1024);
        assert_eq!(budget.used_bytes(), 2048);
    }

    #[test]
    fn unlimited_policy_never_blocks_on_in_tree_usage() {
        let tempdir = TempDir::new().unwrap();
        let budget = StorageBudget::new(tempdir.path(), policy_with_limit(None));
        // Request a non-trivial amount that should still comfortably fit on
        // the test filesystem; with `storage_limit_bytes = None` the in-tree
        // cap is disabled regardless. We deliberately stay well below disk
        // capacity so the free-disk arm doesn't trip.
        assert_eq!(budget.reserve(1024 * 1024).unwrap(), BudgetCheck::Available);
        // Reserving repeatedly should keep returning Available without
        // bookkeeping ever crossing a non-existent threshold.
        for _ in 0..16 {
            assert_eq!(budget.reserve(1024 * 1024).unwrap(), BudgetCheck::Available);
        }
    }

    #[test]
    fn filesystem_full_when_safety_margin_exceeds_free_bytes() {
        let tempdir = TempDir::new().unwrap();
        // A safety margin of u64::MAX is impossible to satisfy on any real
        // filesystem, so the check must report `FilesystemFull` regardless of
        // the in-tree usage estimate.
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: u64::MAX,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = StorageBudget::new(tempdir.path(), policy);
        let result = budget.check(1).unwrap();
        assert!(
            matches!(result, BudgetCheck::FilesystemFull { .. }),
            "expected filesystem-full, got {result:?}"
        );
    }

    #[test]
    fn refresh_picks_up_external_writes() {
        let tempdir = TempDir::new().unwrap();
        let policy = StoragePolicy {
            storage_limit_bytes: Some(8192),
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_millis(0).saturating_add(Duration::from_nanos(1)),
        };
        let budget = StorageBudget::new(tempdir.path(), policy);
        assert_eq!(budget.used_bytes(), 0);

        std::fs::write(tempdir.path().join("blob.bin"), vec![0u8; 4096]).unwrap();
        // Sleep just past the refresh interval so the rescan triggers.
        std::thread::sleep(Duration::from_millis(2));
        budget.refresh_if_stale();
        assert_eq!(budget.used_bytes(), 4096);
    }
}
