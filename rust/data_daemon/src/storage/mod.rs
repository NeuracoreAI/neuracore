//! On-disk storage layout and budget tracking for trace artefacts.
//!
//! [`paths`] resolves the per-trace directory layout under
//! `recordings/{recording_id}/{data_type}/{trace_id}/`. [`budget`] guards the
//! encoder against filling the disk past the `MIN_FREE_DISK_BYTES` (32 MiB)
//! safety margin and against exceeding the configured storage limit.
//!
//! Some helpers (the reservation/release pair, the typed-path getters) are
//! exercised only by unit tests, hence the module-wide `dead_code` allow.

#[allow(dead_code)]
pub mod budget;
#[allow(dead_code)]
pub mod paths;
