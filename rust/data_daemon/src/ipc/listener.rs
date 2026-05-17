//! Tokio task that drains the iceoryx2 subscribers.
//!
//! iceoryx2 0.8 does not expose an `async`/`Notify`-style adaptor, so the
//! listener polls every subscriber it owns on a short tick. The cadence is
//! fast enough for the phase 4 smoke test (a few hundred frames in ≤1 s) and
//! slow enough to leave idle daemons effectively quiescent. Phase 5 replaces
//! the timer with a `WaitSet`-backed notifier when frame throughput matters.
//!
//! ## Subscriber families
//!
//! - `commands` (opened by [`IpcTransport`] at startup) — lifecycle envelopes.
//! - `scalars` (opened by [`IpcTransport`] at startup) — joint / sensor JSON.
//! - `frames/<WxH>` (opened lazily by *this* task on receipt of an
//!   [`Envelope::OpenFrameStream`] command) — RGB video.
//!
//! All three carry JSON-encoded [`Envelope`]s in phase 4. Phase 5 migrates the
//! video stream to a typed zero-copy sample; the routing logic here doesn't
//! change because [`Envelope::Frame`] is already the only data variant the
//! dispatcher needs to recognise.
//!
//! ## Send-safety
//!
//! iceoryx2's [`Subscriber`] is `Send` but `!Sync`, so a `&Subscriber` borrow
//! held across an `await` would make this task's future `!Send` — tokio's
//! multi-thread runtime would refuse to spawn it. The loop therefore drains
//! every subscriber synchronously into a local `Vec<Envelope>` before
//! awaiting the dispatcher: the borrow lifetimes are confined to the
//! synchronous batching phase, and only the (Send) batch survives across the
//! await.

use std::collections::HashMap;
use std::time::Duration;

use data_daemon_ipc::Envelope;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::ipc;
use tokio::select;
use tokio::sync::{broadcast, mpsc};
use tokio::time::sleep;

use crate::ipc::node::{FrameSubscription, IpcTransport};
use crate::lifecycle::signals::ShutdownSignal;

/// How often the listener polls the iceoryx2 subscribers.
///
/// Picked to bound wake-up latency to ~10 ms — well below the 1 s registration
/// debounce in §4 of the rewrite plan, so the trace lifecycle isn't gated on
/// the listener loop.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Drain the iceoryx2 subscribers until a shutdown signal arrives.
///
/// Each successfully decoded envelope is forwarded to the dispatcher's
/// [`mpsc::Sender`]. If the dispatcher's queue fills (the dispatcher has
/// stalled), the listener blocks instead of dropping samples — backpressure
/// propagates back to iceoryx2's publisher queue and the producer SDK.
///
/// The listener takes ownership of the [`IpcTransport`]; when this task
/// returns the transport's destructor releases the iceoryx2 node and all
/// per-resolution frame services back to the OS.
pub async fn run(
    transport: IpcTransport,
    dispatcher_tx: mpsc::Sender<Envelope>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    tracing::info!(
        commands = data_daemon_ipc::service_name::COMMANDS,
        scalars = data_daemon_ipc::service_name::SCALARS,
        "ipc listener started"
    );

    let mut counters = LoopCounters::default();
    // TODO(phase 5): evict entries when their owning traces finalise. Phase 4
    // only opens a handful of resolutions per run, so unbounded growth is
    // bounded by the producer's resolution variety in practice; once
    // `trace_actor` learns when a frame stream is done, the listener can drop
    // the matching subscription here.
    let mut frame_subs: HashMap<(u32, u32), FrameSubscription> = HashMap::new();
    let mut batch: Vec<(&'static str, Envelope)> = Vec::with_capacity(64);

    loop {
        // -- Synchronous drain --------------------------------------------------
        // The subscriber borrows MUST stay inside this block (no `.await` in
        // any of these calls). The local `batch` is `Send`, so it can survive
        // across the awaits below without infecting the task with !Send.
        drain_subscriber(
            "commands",
            transport.commands_subscriber(),
            &mut batch,
            &mut counters,
        );
        drain_subscriber(
            "scalars",
            transport.scalars_subscriber(),
            &mut batch,
            &mut counters,
        );
        for subscription in frame_subs.values() {
            drain_subscriber(
                "frames",
                subscription.subscriber(),
                &mut batch,
                &mut counters,
            );
        }

        // -- Async forward ------------------------------------------------------
        // OpenFrameStream is dual-purpose: it opens the per-resolution
        // service (a synchronous side-effect on `frame_subs` and
        // `transport`) AND propagates to the dispatcher so the per-trace
        // actor learns the resolution.
        for (label, envelope) in batch.drain(..) {
            if let Envelope::OpenFrameStream {
                trace_id,
                width,
                height,
            } = &envelope
            {
                open_frame_stream(&transport, &mut frame_subs, *width, *height, trace_id);
            }
            let kind = envelope.kind();
            if dispatcher_tx.send(envelope).await.is_err() {
                tracing::debug!(
                    subscriber = label,
                    envelope = kind,
                    "ipc listener stopping: dispatcher receiver dropped"
                );
                return;
            }
            tracing::trace!(
                subscriber = label,
                envelope = kind,
                "ipc envelope forwarded"
            );
        }

        // -- Yield / shutdown ---------------------------------------------------
        select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "ipc listener shutting down");
                return;
            }
            _ = sleep(POLL_INTERVAL) => {}
        }
    }
}

/// Counters reported in the slow-path warning logs.
#[derive(Default)]
struct LoopCounters {
    decode_failures: u64,
    receive_failures: u64,
}

/// Synchronously drain every available sample on `subscriber`, appending
/// decoded envelopes to `batch`.
///
/// Receive errors and decode failures are logged with a saturating counter
/// rather than returned — both are recoverable (a malformed sample doesn't
/// invalidate the next one) and the only path that escalates to a listener
/// exit is the dispatcher-receiver-closed branch, which is handled in
/// [`run`].
fn drain_subscriber(
    label: &'static str,
    subscriber: &Subscriber<ipc::Service, [u8], ()>,
    batch: &mut Vec<(&'static str, Envelope)>,
    counters: &mut LoopCounters,
) {
    loop {
        match subscriber.receive() {
            Ok(Some(sample)) => match Envelope::decode(sample.payload()) {
                Ok(envelope) => batch.push((label, envelope)),
                Err(error) => {
                    counters.decode_failures = counters.decode_failures.saturating_add(1);
                    tracing::warn!(
                        %error,
                        subscriber = label,
                        decode_failures = counters.decode_failures,
                        "ipc envelope decode failed; dropping sample"
                    );
                }
            },
            Ok(None) => return,
            Err(error) => {
                counters.receive_failures = counters.receive_failures.saturating_add(1);
                tracing::warn!(
                    error = %error,
                    subscriber = label,
                    receive_failures = counters.receive_failures,
                    "ipc subscriber receive failed"
                );
                return;
            }
        }
    }
}

/// Lazily open the per-resolution frame service.
///
/// Idempotent: a repeat `OpenFrameStream` for the same `(width, height)` from
/// a different trace reuses the existing subscriber. Failure is logged at
/// `warn` and the dispatcher still receives the envelope so the trace_actor
/// observes the lifecycle transition; the actor will discover the missing
/// stream the first time it tries to read a frame.
fn open_frame_stream(
    transport: &IpcTransport,
    registry: &mut HashMap<(u32, u32), FrameSubscription>,
    width: u32,
    height: u32,
    trace_id: &str,
) {
    if registry.contains_key(&(width, height)) {
        tracing::debug!(
            trace_id,
            width,
            height,
            "frames service already attached; reusing subscriber"
        );
        return;
    }
    match transport.open_frame_subscriber(width, height) {
        Ok(subscription) => {
            tracing::info!(
                trace_id,
                width,
                height,
                "opened frames service for new resolution"
            );
            registry.insert((width, height), subscription);
        }
        Err(error) => {
            tracing::warn!(
                %error,
                trace_id,
                width,
                height,
                "failed to open frames service; samples for this resolution will be dropped"
            );
        }
    }
}
