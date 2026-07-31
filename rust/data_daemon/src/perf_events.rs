//! Opt-in structured performance events for local integration diagnostics.
//!
//! When `NCD_PERF_METRICS` is truthy and `NCD_PERF_EVENTS_PATH` is set, phase
//! boundaries are appended as JSONL. The normal daemon path pays only an
//! environment lookup and branch. Events deliberately contain no credentials,
//! URLs, or request/response bodies.

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// Append a single correlated phase event when local performance capture is on.
pub fn emit(
    phase: &str,
    event: &str,
    recording_index: Option<i64>,
    trace_id: Option<&str>,
    elapsed: Option<Duration>,
    details: Value,
) {
    if !enabled() {
        return;
    }
    let Ok(path) = std::env::var("NCD_PERF_EVENTS_PATH") else {
        return;
    };
    let timestamp_unix_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    let case_id = std::env::var("NCD_PERF_CASE_ID").unwrap_or_default();
    let mut payload = json!({
        "schema_version": 1,
        "timestamp_unix_ns": timestamp_unix_ns,
        "case_id": case_id,
        "component": "daemon",
        "phase": phase,
        "event": event,
        "details": details,
    });
    if let Some(recording_index) = recording_index {
        payload["recording_index"] = json!(recording_index);
    }
    if let Some(trace_id) = trace_id {
        payload["trace_id"] = json!(trace_id);
    }
    if let Some(elapsed) = elapsed {
        payload["elapsed_ms"] = json!(elapsed.as_secs_f64() * 1_000.0);
    }

    let Ok(mut encoded) = serde_json::to_vec(&payload) else {
        return;
    };
    encoded.push(b'\n');

    // One append write per event keeps each JSONL record intact. The mutex
    // protects concurrent daemon tasks; O_APPEND protects the shared offset.
    let Ok(_guard) = WRITE_LOCK.lock() else {
        return;
    };
    let Ok(mut output) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = output.write_all(&encoded);
}

fn enabled() -> bool {
    std::env::var("NCD_PERF_METRICS")
        .ok()
        .is_some_and(|value| value_is_enabled(&value))
}

fn value_is_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::value_is_enabled;

    #[test]
    fn capture_flag_is_explicitly_truthy() {
        for value in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(value_is_enabled(value));
        }
        for value in ["", "0", "false", "off", "unexpected"] {
            assert!(!value_is_enabled(value));
        }
    }
}
