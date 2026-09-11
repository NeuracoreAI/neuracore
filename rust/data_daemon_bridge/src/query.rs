//! Request-response calls the producer makes to the daemon.
//!
//! The thin producer owns no recording identity and no daemon state, so
//! everything it needs to know it asks for: whether a daemon is up
//! ([`wait_until_ready`]), which version it was built from ([`daemon_version`]),
//! and which recording a source has open ([`query_recording_state`]).
//!
//! Two concerns, kept apart. [`request_reply_once`] and [`poll_until`] know how
//! to talk to an iceoryx2 request-response port — loan, send, poll for a reply,
//! give up on a deadline — and nothing about what is being asked. Each service
//! below supplies only its own meaning: which client port, how long one attempt
//! may wait, and how to read a reply.

use std::time::{Duration, Instant};

use data_daemon_shared::{
    HealthReply, HealthRequest, LiveRecording, RecordingStateReply, VersionReply, VersionRequest,
};
use iceoryx2::port::client::Client;
use iceoryx2::prelude::ipc;

use crate::publisher::{now_ns, with_producer, ProducerError};

/// Poll cadence while waiting for any single reply.
///
/// A property of the client port rather than of any one service: `receive()`
/// does not block, so this is simply how often it is worth re-asking.
const RECEIVE_POLL: Duration = Duration::from_millis(2);
/// Interval between successive health probes to the daemon.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// How long a single health request waits for the daemon's reply before re-asking.
const HEALTH_RESPONSE_WAIT: Duration = Duration::from_millis(20);
/// Interval between successive version requests to the daemon.
const VERSION_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// How long a single version request waits for the daemon's reply before re-asking.
const VERSION_RESPONSE_WAIT: Duration = Duration::from_millis(20);

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

/// Send one request on `client` and wait up to `response_wait` for its reply.
///
/// `decode` turns a reply payload into `Some(value)` when it is the answer the
/// caller wanted, or `None` when it is not — a nonce that does not match the
/// request, say. Either way the attempt is over; re-asking is [`poll_until`]'s
/// job.
///
/// `Ok(None)` therefore covers three things a caller treats alike: no reply
/// within the window, a reply that did not match, and a receive error. Only a
/// failure to *send* is an `Err`, because that says the port itself is unusable.
fn request_reply_once<T>(
    client: &Client<ipc::Service, [u8], (), [u8], ()>,
    request_bytes: &[u8],
    response_wait: Duration,
    service: &'static str,
    decode: impl Fn(&[u8]) -> Result<Option<T>, ProducerError>,
) -> Result<Option<T>, ProducerError> {
    let request = client
        .loan_slice_uninit(request_bytes.len())
        .map_err(|error| ProducerError::Loan(error.to_string()))?;
    let request = request.write_from_slice(request_bytes);
    let pending = request
        .send()
        .map_err(|error| ProducerError::Send(error.to_string()))?;

    let response_deadline = Instant::now() + response_wait;
    loop {
        match pending.receive() {
            Ok(Some(response)) => return decode(response.payload()),
            Ok(None) => {}
            Err(error) => {
                tracing::debug!(%error, service, "receive failed; treating as no reply");
                return Ok(None);
            }
        }
        if Instant::now() >= response_deadline {
            return Ok(None);
        }
        std::thread::sleep(RECEIVE_POLL);
    }
}

/// Re-run `once` on `interval` until it yields a value or `deadline` passes.
fn poll_until<T>(
    deadline: Instant,
    interval: Duration,
    mut once: impl FnMut() -> Result<Option<T>, ProducerError>,
) -> Result<Option<T>, ProducerError> {
    loop {
        if let Some(value) = once()? {
            return Ok(Some(value));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(interval);
    }
}

/// Block (with the GIL released by the caller) until the daemon answers a
/// side-effect-free health probe or `timeout_s` elapses.
pub(crate) fn wait_until_ready(timeout_s: f64) -> Result<Option<u32>, ProducerError> {
    let nonce = now_ns() as u64;
    let request_bytes = HealthRequest { nonce }.encode()?;
    poll_until(
        Instant::now() + bounded_timeout(timeout_s),
        HEALTH_POLL_INTERVAL,
        || {
            with_producer(|state| {
                request_reply_once(
                    &state.health_client,
                    &request_bytes,
                    HEALTH_RESPONSE_WAIT,
                    "health",
                    |payload| {
                        let reply = HealthReply::decode(payload)?;
                        Ok((reply.pid > 0 && reply.nonce == nonce).then_some(reply.pid))
                    },
                )
            })
        },
    )
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
    poll_until(
        Instant::now() + bounded_timeout(timeout_s),
        VERSION_POLL_INTERVAL,
        || {
            with_producer(|state| {
                request_reply_once(
                    &state.version_client,
                    &request_bytes,
                    VERSION_RESPONSE_WAIT,
                    "version",
                    |payload| {
                        let reply = VersionReply::decode(payload)?;
                        Ok((reply.nonce == nonce).then_some(reply.version))
                    },
                )
            })
        },
    )
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
        request_reply_once(
            &state.recording_state_client,
            request_bytes,
            bounded_timeout(timeout_s),
            "recording-state",
            |payload| Ok(Some(RecordingStateReply::decode(payload)?.recording)),
        )
    })
}
