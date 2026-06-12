//! Write-behind for per-trace `trace.json` files.
//!
//! Scalar / sensor trace actors append JSON entries on their hot path. Writing
//! each entry inline blocks the actor's tokio task on a `write()` which, on the
//! spool's shared ext4 (`data=ordered`, `discard`), periodically stalls for
//! hundreds of ms behind a journal commit. Because the actor sits on the
//! dispatcher → IPC-listener drain path, that stall back-pressures all the way
//! out to the producer's next `log_*` publish — the joint-logging latency
//! spikes we chased.
//!
//! This dedicated OS thread owns every open [`JsonTraceWriter`] and performs the
//! blocking appends / finishes off that path. Actors only *enqueue* (open,
//! append and cancel are fire-and-forget; finalise awaits a one-shot ack), so a
//! disk stall blocks this one thread instead of the IPC drain. It mirrors the
//! daemon's SQLite write-behind ([`crate::state::trace_writer`]); a single
//! thread is ample because trace JSON is tiny (the integration matrix's whole
//! workload is ~MB/s) and the stalls are intermittent, not a sustained
//! slowdown, so the thread always catches up between journal commits.
//!
//! Per-trace ordering is preserved: each `(open, append…, finish)` sequence for
//! one `trace_id` arrives on a single FIFO channel and is applied in order, and
//! different traces never share a writer (they key by `trace_id`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use serde_json::Value;
use tokio::sync::oneshot;

use crate::encoding::json_trace::{JsonTraceError, JsonTraceWriter};

/// Work item for the JSON write-behind thread.
enum JsonWriteMsg {
    /// Open (create) the `trace.json` for a trace. Errors are deferred to the
    /// matching [`JsonWriteMsg::Finish`] so the hot path never blocks on open.
    Open {
        trace_id: String,
        trace_dir: PathBuf,
    },
    /// Append one entry. `payload` is forwarded verbatim when it is itself valid
    /// JSON, else wrapped in a small fallback object stamped with `timestamp_ns`.
    Append {
        trace_id: String,
        timestamp_ns: i64,
        payload: Vec<u8>,
    },
    /// Finalise the trace (append `]`, flush, close) and report the on-disk byte
    /// total — or any deferred open/append error — back over `ack`.
    Finish {
        trace_id: String,
        ack: oneshot::Sender<Result<u64, JsonTraceError>>,
    },
    /// Discard an open writer without finalising it (cancel). No-op if the trace
    /// never opened a JSON writer (e.g. a video trace).
    Drop { trace_id: String },
}

/// Cloneable handle the per-trace actors use to drive the JSON write-behind
/// thread. Every method is non-blocking except [`finish`](Self::finish), which
/// awaits the thread's acknowledgement (off the actor's hot path, at finalise).
#[derive(Clone)]
pub struct JsonWriteHandle {
    tx: Sender<JsonWriteMsg>,
}

impl JsonWriteHandle {
    /// Open the trace's `trace.json` (fire-and-forget).
    pub fn open(&self, trace_id: &str, trace_dir: PathBuf) {
        let _ = self.tx.send(JsonWriteMsg::Open {
            trace_id: trace_id.to_string(),
            trace_dir,
        });
    }

    /// Append one entry (fire-and-forget). Takes ownership of `payload` so the
    /// caller's frame buffer is freed immediately.
    pub fn append(&self, trace_id: &str, timestamp_ns: i64, payload: Vec<u8>) {
        let _ = self.tx.send(JsonWriteMsg::Append {
            trace_id: trace_id.to_string(),
            timestamp_ns,
            payload,
        });
    }

    /// Finalise the trace and return its final on-disk byte count, surfacing any
    /// deferred open/append error. Awaited at finalise time, never on the hot
    /// path.
    pub async fn finish(&self, trace_id: &str) -> Result<u64, JsonTraceError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        if self
            .tx
            .send(JsonWriteMsg::Finish {
                trace_id: trace_id.to_string(),
                ack: ack_tx,
            })
            .is_err()
        {
            return Err(writer_gone());
        }
        ack_rx.await.unwrap_or_else(|_| Err(writer_gone()))
    }

    /// Discard the trace's open writer without finalising (fire-and-forget).
    pub fn drop_trace(&self, trace_id: &str) {
        let _ = self.tx.send(JsonWriteMsg::Drop {
            trace_id: trace_id.to_string(),
        });
    }
}

/// Spawn the JSON write-behind thread, returning a cloneable handle plus its
/// join handle. The thread exits when the last handle is dropped (the channel
/// closes), draining nothing further — finalise acks already in flight resolve
/// to [`writer_gone`].
pub fn spawn() -> (JsonWriteHandle, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let join = std::thread::Builder::new()
        .name("json-trace-writer".to_string())
        .spawn(move || writer_loop(rx))
        .expect("spawn json-trace-writer thread");
    (JsonWriteHandle { tx }, join)
}

fn writer_loop(rx: Receiver<JsonWriteMsg>) {
    // Open writers, plus a per-trace "first error" that a later `Finish`
    // surfaces — once a trace errors we stop touching its (possibly broken)
    // file and report failure at finalise, matching the old inline behaviour.
    let mut writers: HashMap<String, JsonTraceWriter> = HashMap::new();
    let mut errored: HashMap<String, JsonTraceError> = HashMap::new();

    while let Ok(msg) = rx.recv() {
        match msg {
            JsonWriteMsg::Open {
                trace_id,
                trace_dir,
            } => match JsonTraceWriter::open(&trace_dir) {
                Ok(writer) => {
                    writers.insert(trace_id, writer);
                }
                Err(error) => {
                    errored.insert(trace_id, error);
                }
            },
            JsonWriteMsg::Append {
                trace_id,
                timestamp_ns,
                payload,
            } => {
                if errored.contains_key(&trace_id) {
                    continue;
                }
                if let Some(writer) = writers.get_mut(&trace_id) {
                    if let Err(error) = append_entry(writer, timestamp_ns, &payload) {
                        errored.insert(trace_id, error);
                    }
                }
            }
            JsonWriteMsg::Finish { trace_id, ack } => {
                let result = if let Some(error) = errored.remove(&trace_id) {
                    writers.remove(&trace_id);
                    Err(error)
                } else if let Some(writer) = writers.remove(&trace_id) {
                    writer.finish()
                } else {
                    // Never opened (shouldn't happen — finalise always opens
                    // first), so there's an empty artefact's worth of nothing.
                    Ok(0)
                };
                let _ = ack.send(result);
            }
            JsonWriteMsg::Drop { trace_id } => {
                writers.remove(&trace_id);
                errored.remove(&trace_id);
            }
        }
    }
}

/// Append `payload` to `writer`, writing it verbatim when it parses as JSON and
/// wrapping it in a fallback object otherwise. A single parse decides the
/// branch, preserving the producer's bit-exact float formatting on the common
/// (already-JSON) path.
fn append_entry(
    writer: &mut JsonTraceWriter,
    timestamp_ns: i64,
    payload: &[u8],
) -> Result<(), JsonTraceError> {
    match serde_json::from_slice::<serde::de::IgnoredAny>(payload) {
        Ok(_) => writer.add_raw_entry(payload),
        Err(_) => writer.add_entry(&scalar_fallback_entry(timestamp_ns, payload)),
    }
}

/// Wrap a non-JSON scalar payload in a minimal object so the on-disk
/// `trace.json` array stays parseable. Only reached after a structural JSON
/// parse has already failed, so it never re-parses the bytes.
pub(crate) fn scalar_fallback_entry(timestamp_ns: i64, payload: &[u8]) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("timestamp_ns".to_string(), Value::from(timestamp_ns));
    map.insert("payload_len".to_string(), Value::from(payload.len() as u64));
    Value::Object(map)
}

/// The error reported when the write-behind thread has already exited (process
/// shutdown). Reuses [`JsonTraceError::Write`] so callers need no new variant.
fn writer_gone() -> JsonTraceError {
    JsonTraceError::Write {
        path: PathBuf::from("<json-trace-writer>"),
        source: std::io::Error::other("json trace writer thread stopped"),
    }
}
