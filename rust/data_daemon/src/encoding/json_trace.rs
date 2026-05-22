//! Incremental JSON-array writer for scalar / sensor traces.
//!
//! Entries are buffered in memory, the file is opened lazily on the first
//! frame, and whole `CHUNK_SIZE` chunks are flushed to disk. The on-disk byte
//! layout is a single JSON array with one entry per frame, comma-separated,
//! no whitespace.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::storage::paths::TRACE_JSON_FILENAME;

/// Default flush threshold: buffered entries are written to disk once they
/// reach 4 MiB.
pub const DEFAULT_FLUSH_BYTES: usize = 4 * 1024 * 1024;

/// Errors raised by [`JsonTraceWriter`].
#[derive(Debug, thiserror::Error)]
pub enum JsonTraceError {
    /// Failed to create the parent directory or open the output file.
    #[error("failed to open trace file {path}: {source}")]
    Open {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// Failed to serialize an entry to JSON.
    #[error("failed to serialise trace entry: {0}")]
    Serialize(#[source] serde_json::Error),
    /// Failed to write buffered bytes to disk.
    #[error("failed to write trace file {path}: {source}")]
    Write {
        /// Path being written to.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
}

/// Incremental writer that streams a JSON array of entries to disk.
///
/// Entries are buffered in memory until either the buffer reaches
/// `flush_threshold` bytes or [`finish`](Self::finish) is called. Each entry
/// is rendered with compact `(",", ":")` separators, so the on-disk file
/// carries no insignificant whitespace.
pub struct JsonTraceWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    /// Pending JSON bytes not yet flushed to the file. Includes the leading
    /// `[` once the first entry has been added.
    buffer: Vec<u8>,
    flush_threshold: usize,
    started: bool,
    first_entry: bool,
    /// Bytes already flushed to disk (excludes the closing `]`, which is
    /// only appended on `finish`).
    bytes_on_disk: u64,
}

impl JsonTraceWriter {
    /// Open a writer producing `{output_dir}/trace.json`.
    pub fn open(output_dir: &Path) -> Result<Self, JsonTraceError> {
        Self::open_with(output_dir, TRACE_JSON_FILENAME, DEFAULT_FLUSH_BYTES)
    }

    /// Open a writer with a custom filename and flush threshold.
    pub fn open_with(
        output_dir: &Path,
        filename: &str,
        flush_threshold: usize,
    ) -> Result<Self, JsonTraceError> {
        std::fs::create_dir_all(output_dir).map_err(|source| JsonTraceError::Open {
            path: output_dir.to_path_buf(),
            source,
        })?;

        let path = output_dir.join(filename);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|source| JsonTraceError::Open {
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            buffer: Vec::with_capacity(flush_threshold.min(64 * 1024)),
            flush_threshold,
            started: false,
            first_entry: true,
            bytes_on_disk: 0,
        })
    }

    /// Path being written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes flushed to disk so far (does not include the trailing `]` until
    /// [`finish`](Self::finish) has run).
    pub fn bytes_on_disk(&self) -> u64 {
        self.bytes_on_disk
    }

    /// Append one entry to the JSON array.
    ///
    /// Generic over any `serde::Serialize` value so callers can hand the
    /// writer either a `serde_json::Value` or a typed struct without an extra
    /// `to_value` hop.
    pub fn add_entry<T: Serialize>(&mut self, entry: &T) -> Result<(), JsonTraceError> {
        if !self.started {
            self.buffer.push(b'[');
            self.started = true;
        }
        if self.first_entry {
            self.first_entry = false;
        } else {
            self.buffer.push(b',');
        }
        serde_json::to_writer(&mut self.buffer, entry).map_err(JsonTraceError::Serialize)?;

        if self.buffer.len() >= self.flush_threshold {
            self.flush_buffer()?;
        }
        Ok(())
    }

    /// Append a slice of entries, flushing once after the whole slice has
    /// been appended. Useful when the caller already has a `Vec<T>` to
    /// commit (e.g. a draining heartbeat).
    pub fn add_entries<T: Serialize>(&mut self, entries: &[T]) -> Result<(), JsonTraceError> {
        for entry in entries {
            self.add_entry(entry)?;
        }
        Ok(())
    }

    /// Append an already-serialised JSON entry verbatim.
    ///
    /// Skips the `Value → bytes` round trip so a float supplied by the SDK
    /// (e.g. `7/60 = 0.11666666666666667`) lands on disk with the exact same
    /// textual representation it was logged with — required by the
    /// integration test matrix, which compares `trace.json` floats with
    /// `actual != expected` rather than approximate tolerance. The caller
    /// must guarantee `entry_bytes` is a complete JSON value with no
    /// trailing comma or whitespace.
    pub fn add_raw_entry(&mut self, entry_bytes: &[u8]) -> Result<(), JsonTraceError> {
        if !self.started {
            self.buffer.push(b'[');
            self.started = true;
        }
        if self.first_entry {
            self.first_entry = false;
        } else {
            self.buffer.push(b',');
        }
        self.buffer.extend_from_slice(entry_bytes);

        if self.buffer.len() >= self.flush_threshold {
            self.flush_buffer()?;
        }
        Ok(())
    }

    /// Finalise the file: append `]`, flush, and close the buffered writer.
    ///
    /// An empty trace (no `add_entry` calls) is still finalised as `[]` so
    /// the file is always valid JSON.
    pub fn finish(mut self) -> Result<u64, JsonTraceError> {
        if !self.started {
            self.buffer.extend_from_slice(b"[]");
        } else {
            self.buffer.push(b']');
        }
        self.flush_buffer()?;
        self.writer
            .flush()
            .map_err(|source| JsonTraceError::Write {
                path: self.path.clone(),
                source,
            })?;
        Ok(self.bytes_on_disk)
    }

    fn flush_buffer(&mut self) -> Result<(), JsonTraceError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.writer
            .write_all(&self.buffer)
            .map_err(|source| JsonTraceError::Write {
                path: self.path.clone(),
                source,
            })?;
        self.bytes_on_disk = self.bytes_on_disk.saturating_add(self.buffer.len() as u64);
        self.buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use tempfile::TempDir;

    fn read_back(path: &Path) -> Value {
        let bytes = std::fs::read(path).expect("read");
        serde_json::from_slice(&bytes).expect("parse")
    }

    #[test]
    fn empty_trace_produces_valid_json_array() {
        let tempdir = TempDir::new().unwrap();
        let writer = JsonTraceWriter::open(tempdir.path()).unwrap();
        let bytes = writer.finish().unwrap();
        assert_eq!(bytes, b"[]".len() as u64);

        let parsed = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        assert_eq!(parsed, json!([]));
    }

    #[test]
    fn entries_round_trip_through_serde_json() {
        let tempdir = TempDir::new().unwrap();
        let mut writer = JsonTraceWriter::open(tempdir.path()).unwrap();
        writer
            .add_entry(&json!({"frame": 0, "timestamp": 1.5}))
            .unwrap();
        writer
            .add_entry(&json!({"frame": 1, "timestamp": 2.5}))
            .unwrap();
        let _bytes = writer.finish().unwrap();

        let parsed = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        assert_eq!(
            parsed,
            json!([
                {"frame": 0, "timestamp": 1.5},
                {"frame": 1, "timestamp": 2.5}
            ])
        );
    }

    #[test]
    fn buffer_flushes_when_threshold_reached() {
        let tempdir = TempDir::new().unwrap();
        // Tiny threshold so a single ~50-byte entry forces a flush. We then
        // check that `bytes_on_disk` advances mid-stream, proving the writer
        // doesn't buffer the whole file in memory.
        let mut writer =
            JsonTraceWriter::open_with(tempdir.path(), TRACE_JSON_FILENAME, 32).unwrap();
        writer
            .add_entry(&json!({"frame": 0, "padding": "xxxxxxxxxxxxxx"}))
            .unwrap();
        assert!(writer.bytes_on_disk() > 0, "writer must flush mid-stream");
        let total = writer.finish().unwrap();
        assert!(total > 0);

        let parsed = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        assert_eq!(parsed[0]["frame"], 0);
    }

    #[test]
    fn add_entries_writes_each_in_order() {
        let tempdir = TempDir::new().unwrap();
        let mut writer = JsonTraceWriter::open(tempdir.path()).unwrap();
        let entries = vec![json!({"i": 0}), json!({"i": 1}), json!({"i": 2})];
        writer.add_entries(&entries).unwrap();
        writer.finish().unwrap();

        let parsed = read_back(&tempdir.path().join(TRACE_JSON_FILENAME));
        assert_eq!(parsed, json!([{"i": 0}, {"i": 1}, {"i": 2}]));
    }
}
