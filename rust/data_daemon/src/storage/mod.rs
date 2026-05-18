//! On-disk storage layout and budget tracking for trace artefacts.
//!
//! Phase 5a of the rewrite plan (see `docs/data-daemon-rewrite.md`).
//! [`paths`] resolves the per-trace directory layout, mirroring the Python
//! daemon's `recordings/{recording_id}/{data_type}/{trace_id}/` convention.
//! [`budget`] guards the encoder against filling the disk past the
//! `MIN_FREE_DISK_BYTES` (32 MiB) safety margin and against exceeding the
//! configured storage limit. The trace actor wires the in-tree check in at
//! sub-phase 5f; the reservation/release pair (used by Phase 7's quota
//! tightening) and the typed-path getters (used by Phase 6's upload
//! coordinator) ship now so their unit tests run, hence the module-wide
//! `dead_code` allow until those callers land.

#[allow(dead_code)]
pub mod budget;
#[allow(dead_code)]
pub mod paths;
