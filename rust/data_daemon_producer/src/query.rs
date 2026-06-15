//! Recording-id resolution over the `queries` request-response service.
//!
//! The thin producer never mints recording identity — the daemon allocates the
//! cloud id asynchronously after `/recording/start`. These helpers ask the
//! daemon (identifying the recording by its source + capture `timestamp_ns`
//! marker) and return the id once minted.

use std::time::{Duration, Instant};

use data_daemon_ipc::RecordingIdReply;

use crate::publisher::{with_producer, ProducerError, ProducerState};

/// Interval between successive recording-id requests to the daemon.
const RECORDING_ID_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long a single request waits for the daemon's reply before re-asking.
const RECORDING_ID_RESPONSE_WAIT: Duration = Duration::from_millis(40);
/// Poll cadence while waiting for a single request's reply.
const RECORDING_ID_RECEIVE_POLL: Duration = Duration::from_millis(2);

/// Block (with the GIL released by the caller) until the daemon-owned cloud
/// `recording_id` is available or `timeout_s` elapses, re-asking on each poll
/// interval. Returns `None` on timeout / when no daemon is answering.
pub(crate) fn resolve_recording_id(
    request_bytes: &[u8],
    timeout_s: f64,
) -> Result<Option<String>, ProducerError> {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
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
        .queries_client
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
            Err(error) => return Err(ProducerError::Send(error.to_string())),
        }
        if Instant::now() >= response_deadline {
            return Ok(None);
        }
        std::thread::sleep(RECORDING_ID_RECEIVE_POLL);
    }
}
