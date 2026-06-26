//! On-disk storage layout and budget tracking for trace artefacts.
//!
//! [`paths`] resolves the per-trace directory layout under
//! `recordings/{recording_id}/{data_type}/{trace_id}/`. [`budget`] guards the
//! encoder against filling the disk past the `MIN_FREE_DISK_BYTES` (32 MiB)
//! safety margin and against exceeding the configured storage limit.

pub mod budget;
pub mod paths;
