//! Sidecar `trace.json` for video traces.
//!
//! Sub-phase 5e of the rewrite plan (see `docs/data-daemon-rewrite.md`).
//! Mirrors `recording_encoding_disk_manager/encoding/video_trace.py` —
//! specifically the metadata-accumulation half: each frame's metadata
//! dictionary is captured as it arrives, and on finalize the writer flushes a
//! compact JSON array alongside the mp4 outputs.
//!
//! Byte layout matches the Python writer so the upload coordinator and any
//! offline diff tooling treat both implementations as interchangeable:
//!
//! - One JSON array; `serde_json` default rendering (no whitespace) matches
//!   `json.dumps(separators=(",", ":"))`.
//! - On finish, each entry gets `"frame_idx": <index>` and `"frame": null`
//!   added/overwritten — `frame_idx` is the 0-based position in the list and
//!   `frame` is the spot where the Python writer would have inlined a base64
//!   frame thumbnail (always null in the current pipeline, kept for forward
//!   compatibility with the dashboard schema).
//! - Map insertion order is preserved (enabled by `preserve_order` on
//!   `serde_json` in `data_daemon`'s `Cargo.toml`).

use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::storage::paths::TRACE_JSON_FILENAME;

/// Errors raised by [`VideoMetadataAccumulator`].
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    /// Failed to create the parent directory or open the output file.
    #[error("failed to open metadata file {path}: {source}")]
    Open {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// Failed to serialise the accumulated metadata array.
    #[error("failed to serialise video metadata: {0}")]
    Serialize(#[source] serde_json::Error),
    /// Failed to write buffered bytes to disk.
    #[error("failed to write metadata file {path}: {source}")]
    Write {
        /// Path being written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
}

/// Accumulator for per-frame metadata dictionaries.
///
/// Construction is allocation-only; the file isn't touched until
/// [`finish`](Self::finish) runs, mirroring the Python implementation which
/// writes the sidecar in one shot after both mp4 encoders close. Buffering in
/// memory is acceptable here because video metadata is tiny relative to the
/// raw frame payloads — a 30 minute capture at 30 fps caps out around 50 K
/// entries with a handful of small numeric fields each.
#[derive(Debug, Default)]
pub struct VideoMetadataAccumulator {
    entries: Vec<Map<String, Value>>,
}

impl VideoMetadataAccumulator {
    /// Construct an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of metadata entries currently buffered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no entries have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Record one frame's metadata. The supplied map is taken by value so the
    /// caller cannot accidentally mutate it after recording, and so we avoid
    /// an extra clone on the hot path. The `"frame"` slot is initialised to
    /// `null` immediately (matching the Python writer's first pass over the
    /// metadata); `finish` re-stamps it alongside `frame_idx`.
    pub fn record_frame(&mut self, mut entry: Map<String, Value>) {
        entry.insert("frame".to_string(), Value::Null);
        self.entries.push(entry);
    }

    /// Convenience: record a frame whose metadata is provided as a
    /// `serde_json::Value`. Non-object values (numbers, strings, arrays) are
    /// dropped silently — the Python writer behaves the same way via
    /// `if not isinstance(obj, dict): return`.
    pub fn record_value(&mut self, value: Value) {
        match value {
            Value::Object(map) => self.record_frame(map),
            Value::Array(items) => {
                for item in items {
                    self.record_value(item);
                }
            }
            _ => {
                tracing::trace!("ignoring non-object metadata entry");
            }
        }
    }

    /// Flush the accumulated metadata to `{output_dir}/trace.json`.
    ///
    /// Returns the total bytes written. The directory is created if missing.
    pub fn finish(self, output_dir: &Path) -> Result<u64, MetadataError> {
        self.finish_with_filename(output_dir, TRACE_JSON_FILENAME)
    }

    /// Variant of [`finish`](Self::finish) that lets the caller override the
    /// sidecar filename. Used by tests; the production code always uses
    /// [`TRACE_JSON_FILENAME`].
    pub fn finish_with_filename(
        mut self,
        output_dir: &Path,
        filename: &str,
    ) -> Result<u64, MetadataError> {
        std::fs::create_dir_all(output_dir).map_err(|source| MetadataError::Open {
            path: output_dir.to_path_buf(),
            source,
        })?;
        let path = output_dir.join(filename);

        for (index, entry) in self.entries.iter_mut().enumerate() {
            // `frame_idx` is the 0-based position; `frame` is the legacy
            // base64 thumbnail slot the dashboard still expects but the
            // pipeline no longer populates. Stamping both here keeps the
            // sidecar identical to the Python writer's output even when the
            // producer omitted them upstream.
            entry.insert("frame_idx".to_string(), Value::from(index as u64));
            entry.insert("frame".to_string(), Value::Null);
        }

        let bytes = serde_json::to_vec(&self.entries).map_err(MetadataError::Serialize)?;

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|source| MetadataError::Open {
                path: path.clone(),
                source,
            })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(&bytes)
            .map_err(|source| MetadataError::Write {
                path: path.clone(),
                source,
            })?;
        writer.flush().map_err(|source| MetadataError::Write {
            path: path.clone(),
            source,
        })?;
        Ok(bytes.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn read_back(path: &Path) -> Vec<u8> {
        std::fs::read(path).expect("read metadata file")
    }

    #[test]
    fn empty_accumulator_writes_empty_array() {
        let tempdir = TempDir::new().unwrap();
        let accumulator = VideoMetadataAccumulator::new();
        assert!(accumulator.is_empty());
        let bytes = accumulator.finish(tempdir.path()).unwrap();
        let written = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        assert_eq!(written, b"[]");
        assert_eq!(bytes, written.len() as u64);
    }

    #[test]
    fn fixture_matches_python_video_trace_output() {
        // Hand-rolled fixture mirroring the bytes a Python `VideoTrace.finish`
        // would produce for the same inputs. The Python writer:
        //   - calls `json.dumps(separators=(",", ":"), ensure_ascii=False)`
        //   - on every entry: `entry["frame_idx"] = i; entry["frame"] = None`
        //   - preserves dict insertion order (Python 3.7+)
        //
        // Inputs intentionally exercise: integer + float timestamps, string
        // values, nested objects, and an entry whose `frame` key was already
        // present (overwrite path).
        let tempdir = TempDir::new().unwrap();
        let mut accumulator = VideoMetadataAccumulator::new();

        let mut entry_a = Map::new();
        entry_a.insert("timestamp".to_string(), json!(1.5));
        entry_a.insert("width".to_string(), json!(640));
        entry_a.insert("height".to_string(), json!(480));
        accumulator.record_frame(entry_a);

        let mut entry_b = Map::new();
        entry_b.insert("timestamp".to_string(), json!(2));
        entry_b.insert("source".to_string(), json!("rgb-camera"));
        entry_b.insert("extra".to_string(), json!({"sequence": 17, "flag": true}));
        // Pre-existing `frame` payload — the Python writer overwrites it with
        // null on the second pass; this test confirms we do too.
        entry_b.insert("frame".to_string(), json!("stale"));
        accumulator.record_frame(entry_b);

        let written_bytes = accumulator.finish(tempdir.path()).unwrap();
        let actual = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));

        let expected = br#"[{"timestamp":1.5,"width":640,"height":480,"frame":null,"frame_idx":0},{"timestamp":2,"source":"rgb-camera","extra":{"sequence":17,"flag":true},"frame":null,"frame_idx":1}]"#.to_vec();
        assert_eq!(
            actual, expected,
            "metadata sidecar bytes diverged from Python writer fixture"
        );
        assert_eq!(written_bytes, expected.len() as u64);
    }

    #[test]
    fn frame_idx_starts_at_zero_and_is_contiguous() {
        let tempdir = TempDir::new().unwrap();
        let mut accumulator = VideoMetadataAccumulator::new();
        for index in 0..5 {
            let mut entry = Map::new();
            entry.insert("timestamp".to_string(), json!(index as f64 * 0.033));
            accumulator.record_frame(entry);
        }
        assert_eq!(accumulator.len(), 5);
        accumulator.finish(tempdir.path()).unwrap();

        let bytes = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        let array = parsed.as_array().unwrap();
        assert_eq!(array.len(), 5);
        for (index, entry) in array.iter().enumerate() {
            assert_eq!(entry["frame_idx"], json!(index as u64));
            assert!(entry["frame"].is_null());
        }
    }

    #[test]
    fn record_value_flattens_array_payloads() {
        // Python's `_handle_metadata` recurses through list payloads — the
        // producer is allowed to batch several frames into one envelope. The
        // accumulator must record each contained dict as its own entry.
        let tempdir = TempDir::new().unwrap();
        let mut accumulator = VideoMetadataAccumulator::new();
        accumulator.record_value(json!([
            {"timestamp": 0.1},
            {"timestamp": 0.2},
            42,           // non-object — dropped
            {"timestamp": 0.3},
        ]));
        assert_eq!(accumulator.len(), 3);
        accumulator.finish(tempdir.path()).unwrap();

        let bytes = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
        assert_eq!(parsed[2]["timestamp"], json!(0.3));
    }

    #[test]
    fn record_value_ignores_scalar_payloads() {
        let mut accumulator = VideoMetadataAccumulator::new();
        accumulator.record_value(json!(42));
        accumulator.record_value(json!("ignored"));
        accumulator.record_value(json!(null));
        assert!(accumulator.is_empty());
    }
}
