//! Recording-id resolution over the `recording_ids` request-response service.
//!
//! The thin producer never mints recording identity — the daemon allocates the
//! cloud id asynchronously after `/recording/start`. These helpers ask the
//! daemon (identifying the recording by its source + capture `timestamp_ns`
//! marker) and return the id once minted.

use std::time::{Duration, Instant};

use data_daemon_shared::{HealthReply, HealthRequest, RecordingIdReply};

use crate::publisher::{now_ns, with_producer, ProducerError, ProducerState};

/// Interval between successive recording-id requests to the daemon.
const RECORDING_ID_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long a single request waits for the daemon's reply before re-asking.
const RECORDING_ID_RESPONSE_WAIT: Duration = Duration::from_millis(40);
/// Poll cadence while waiting for a single request's reply.
const RECORDING_ID_RECEIVE_POLL: Duration = Duration::from_millis(2);
/// Interval between successive health probes to the daemon.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// How long a single health request waits for the daemon's reply before re-asking.
const HEALTH_RESPONSE_WAIT: Duration = Duration::from_millis(20);
/// Poll cadence while waiting for one health reply.
const HEALTH_RECEIVE_POLL: Duration = Duration::from_millis(2);

fn bounded_timeout(timeout_s: f64) -> Duration {
    // Clamp before converting: `Duration::from_secs_f64` panics on a non-finite
    // or huge value, and `timeout_s` is caller-controlled across the FFI
    // boundary (e.g. `float('inf')` / `float('nan')`). `f64::clamp` propagates
    // NaN, so guard it explicitly (→ 0); +inf clamps to a day, well past any
    // sane wait.
    let bounded_timeout_s = if timeout_s.is_nan() {
        0.0
    } else {
        timeout_s.clamp(0.0, 86_400.0)
    };
    Duration::from_secs_f64(bounded_timeout_s)
}

/// Block (with the GIL released by the caller) until the daemon answers a
/// side-effect-free health probe or `timeout_s` elapses.
pub(crate) fn wait_until_ready(timeout_s: f64) -> Result<Option<u32>, ProducerError> {
    let nonce = now_ns() as u64;
    let request_bytes = HealthRequest { nonce }.encode()?;
    let deadline = Instant::now() + bounded_timeout(timeout_s);
    loop {
        if let Some(pid) = health_probe_once(&request_bytes, nonce)? {
            return Ok(Some(pid));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(HEALTH_POLL_INTERVAL);
    }
}

fn health_probe_once(request_bytes: &[u8], nonce: u64) -> Result<Option<u32>, ProducerError> {
    with_producer(|state| {
        let request = state
            .health_client
            .loan_slice_uninit(request_bytes.len())
            .map_err(|error| ProducerError::Loan(error.to_string()))?;
        let request = request.write_from_slice(request_bytes);
        let pending = request
            .send()
            .map_err(|error| ProducerError::Send(error.to_string()))?;

        let response_deadline = Instant::now() + HEALTH_RESPONSE_WAIT;
        loop {
            match pending.receive() {
                Ok(Some(response)) => {
                    let reply = HealthReply::decode(response.payload())?;
                    return Ok((reply.pid > 0 && reply.nonce == nonce).then_some(reply.pid));
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(%error, "health receive failed; treating as no reply");
                    return Ok(None);
                }
            }
            if Instant::now() >= response_deadline {
                return Ok(None);
            }
            std::thread::sleep(HEALTH_RECEIVE_POLL);
        }
    })
}

/// Block (with the GIL released by the caller) until the daemon-owned cloud
/// `recording_id` is available or `timeout_s` elapses, re-asking on each poll
/// interval. Returns `None` on timeout / when no daemon is answering.
pub(crate) fn resolve_recording_id(
    request_bytes: &[u8],
    timeout_s: f64,
) -> Result<Option<String>, ProducerError> {
    let deadline = Instant::now() + bounded_timeout(timeout_s);
    loop {
        let resolved = with_producer(|state| resolve_recording_id_once(state, request_bytes))?;
        if resolved.is_some() {
            return Ok(resolved);
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(RECORDING_ID_POLL_INTERVAL);
    }
}

/// Send one recording-id request and wait briefly for the daemon's reply.
///
/// Returns `Ok(Some(id))` once the daemon has minted the cloud id, or `Ok(None)`
/// when it replied "not yet" or did not reply within the per-request window
/// (e.g. no daemon is up). The caller re-asks until its overall timeout.
fn resolve_recording_id_once(
    state: &ProducerState,
    request_bytes: &[u8],
) -> Result<Option<String>, ProducerError> {
    let request = state
        .recording_id_client
        .loan_slice_uninit(request_bytes.len())
        .map_err(|error| ProducerError::Loan(error.to_string()))?;
    let request = request.write_from_slice(request_bytes);
    let pending = request
        .send()
        .map_err(|error| ProducerError::Send(error.to_string()))?;

    let response_deadline = Instant::now() + RECORDING_ID_RESPONSE_WAIT;
    loop {
        match pending.receive() {
            Ok(Some(response)) => {
                let reply = RecordingIdReply::decode(response.payload())?;
                return Ok(reply.recording_id);
            }
            Ok(None) => {}
            Err(error) => {
                // A transient receive error is not fatal: log it and report "no
                // reply" so the caller's outer loop re-asks until its real
                // deadline, matching this function's documented `Ok(None)`
                // contract (it is a receive failure, not a send failure).
                tracing::debug!(%error, "recording-id receive failed; treating as no reply");
                return Ok(None);
            }
        }
        if Instant::now() >= response_deadline {
            return Ok(None);
        }
        std::thread::sleep(RECORDING_ID_RECEIVE_POLL);
    }
}
