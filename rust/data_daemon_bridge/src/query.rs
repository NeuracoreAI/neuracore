//! Request-response calls the producer makes to the daemon.
//!
//! The thin producer owns no recording identity and no daemon state, so
//! everything it needs to know it asks for: whether a daemon is up
//! ([`wait_until_ready`]), which version it was built from ([`daemon_version`]),
//! and which recording a source has open ([`query_recording_state`]).

use std::time::{Duration, Instant};

use data_daemon_shared::{
    HealthReply, HealthRequest, LiveRecording, RecordingStateReply, VersionReply, VersionRequest,
};

use crate::publisher::{now_ns, with_producer, ProducerError};

/// Poll cadence while waiting for one recording-state reply.
const RECORDING_STATE_RECEIVE_POLL: Duration = Duration::from_millis(2);
/// Interval between successive health probes to the daemon.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// How long a single health request waits for the daemon's reply before re-asking.
const HEALTH_RESPONSE_WAIT: Duration = Duration::from_millis(20);
/// Poll cadence while waiting for one health reply.
const HEALTH_RECEIVE_POLL: Duration = Duration::from_millis(2);
/// Interval between successive version requests to the daemon.
const VERSION_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// How long a single version request waits for the daemon's reply before re-asking.
const VERSION_RESPONSE_WAIT: Duration = Duration::from_millis(20);
/// Poll cadence while waiting for one version reply.
const VERSION_RECEIVE_POLL: Duration = Duration::from_millis(2);

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

/// Block (with the GIL released by the caller) until the daemon reports the
/// neuracore version it was built from, or `timeout_s` elapses.
///
/// `Ok(None)` means nothing answered within `timeout_s`: either no daemon is
/// running, or the running daemon is older than the version service and has
/// no server on it. The caller runs this only after a passing health probe
/// and passes the probe's own time budget, so sustained silence means an old
/// daemon rather than a missing or briefly busy one.
pub(crate) fn daemon_version(timeout_s: f64) -> Result<Option<String>, ProducerError> {
    let nonce = now_ns() as u64;
    let request_bytes = VersionRequest { nonce }.encode()?;
    let deadline = Instant::now() + bounded_timeout(timeout_s);
    loop {
        if let Some(version) = version_probe_once(&request_bytes, nonce)? {
            return Ok(Some(version));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(VERSION_POLL_INTERVAL);
    }
}

fn version_probe_once(request_bytes: &[u8], nonce: u64) -> Result<Option<String>, ProducerError> {
    with_producer(|state| {
        let request = state
            .version_client
            .loan_slice_uninit(request_bytes.len())
            .map_err(|error| ProducerError::Loan(error.to_string()))?;
        let request = request.write_from_slice(request_bytes);
        let pending = request
            .send()
            .map_err(|error| ProducerError::Send(error.to_string()))?;

        let response_deadline = Instant::now() + VERSION_RESPONSE_WAIT;
        loop {
            match pending.receive() {
                Ok(Some(response)) => {
                    let reply = VersionReply::decode(response.payload())?;
                    return Ok((reply.nonce == nonce).then_some(reply.version));
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(%error, "version receive failed; treating as no reply");
                    return Ok(None);
                }
            }
            if Instant::now() >= response_deadline {
                return Ok(None);
            }
            std::thread::sleep(VERSION_RECEIVE_POLL);
        }
    })
}

/// Ask the daemon which recording, if any, `request_bytes` names a source for.
///
/// Single-shot: one request, one bounded wait, because the caller is a poll
/// loop that re-asks on its own cadence.
///
/// The two "no" answers are distinct and must stay so:
///
/// * `Ok(None)` — nothing answered; the caller leaves its state alone.
/// * `Ok(Some(None))` — the daemon answered: no open recording. That is the
///   edge a producer drains and seals its tail chunks on.
pub(crate) fn query_recording_state(
    request_bytes: &[u8],
    timeout_s: f64,
) -> Result<Option<Option<LiveRecording>>, ProducerError> {
    with_producer(|state| {
        let request = state
            .recording_state_client
            .loan_slice_uninit(request_bytes.len())
            .map_err(|error| ProducerError::Loan(error.to_string()))?;
        let request = request.write_from_slice(request_bytes);
        let pending = request
            .send()
            .map_err(|error| ProducerError::Send(error.to_string()))?;

        let response_deadline = Instant::now() + bounded_timeout(timeout_s);
        loop {
            match pending.receive() {
                Ok(Some(response)) => {
                    let reply = RecordingStateReply::decode(response.payload())?;
                    return Ok(Some(reply.recording));
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(%error, "recording-state receive failed; treating as no reply");
                    return Ok(None);
                }
            }
            if Instant::now() >= response_deadline {
                return Ok(None);
            }
            std::thread::sleep(RECORDING_STATE_RECEIVE_POLL);
        }
    })
}
