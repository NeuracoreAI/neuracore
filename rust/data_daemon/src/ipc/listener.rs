//! Tokio task that drains the iceoryx2 `commands` subscriber.
//!
//! iceoryx2 0.8 does not expose an `async`/`Notify`-style adaptor, so the
//! listener polls the subscriber on a short tick. The cadence is fast enough
//! to keep frame latency low and slow enough to leave idle daemons
//! effectively quiescent.
//!
//! ## Send-safety
//!
//! iceoryx2's [`Subscriber`] is `Send` but `!Sync`, so a `&Subscriber` borrow
//! held across an `await` would make this task's future `!Send` — tokio's
//! multi-thread runtime would refuse to spawn it. The loop therefore drains
//! the subscriber synchronously into a local `Vec<Envelope>` before awaiting
//! the dispatcher.

use std::time::Duration;

use data_daemon_ipc::Envelope;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::ipc;
use tokio::select;
use tokio::sync::{broadcast, mpsc};
use tokio::time::sleep;

use crate::ipc::node::IpcTransport;
use crate::lifecycle::signals::ShutdownSignal;

/// How often the listener polls the iceoryx2 subscriber.
///
/// 1 ms bounds the worst-case producer-block time on a full subscriber
/// buffer. At the integration matrix's heaviest fanout (8 multiprocess
/// workers × ~4 producer threads each = ~32 publishers competing for
/// LIFECYCLE_SUBSCRIBER_BUFFER_SIZE=64 slots), 10 ms left producer-side
/// `log_*` calls blocked for ~1 s at a stretch on 2-vCPU hosts when the
/// listener task was preempted off-CPU by ffmpeg / per-trace work.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Drain the iceoryx2 subscriber until a shutdown signal arrives.
///
/// Each successfully decoded envelope is forwarded to the dispatcher's
/// [`mpsc::Sender`]. If the dispatcher's queue fills (the dispatcher has
/// stalled), the listener blocks instead of dropping samples — backpressure
/// propagates back to iceoryx2's publisher queue and the producer SDK.
///
/// The listener takes ownership of the [`IpcTransport`]; when this task
/// returns the transport's destructor releases the iceoryx2 node back to the
/// OS.
pub async fn run(
    transport: IpcTransport,
    dispatcher_tx: mpsc::Sender<Envelope>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    tracing::info!(
        commands = data_daemon_ipc::service_name::COMMANDS,
        "ipc listener started"
    );

    let mut counters = LoopCounters::default();
    let mut batch: Vec<Envelope> = Vec::with_capacity(64);

    loop {
        // -- Synchronous drain --------------------------------------------------
        // The subscriber borrow MUST stay inside this block (no `.await` in
        // any of these calls). The local `batch` is `Send`, so it can survive
        // across the awaits below without infecting the task with !Send.
        drain_subscriber(transport.commands_subscriber(), &mut batch, &mut counters);

        // -- Async forward ------------------------------------------------------
        for envelope in batch.drain(..) {
            let kind = envelope.kind();
            if dispatcher_tx.send(envelope).await.is_err() {
                tracing::debug!(
                    envelope = kind,
                    "ipc listener stopping: dispatcher receiver dropped"
                );
                return;
            }
            tracing::debug!(envelope = kind, "ipc envelope forwarded");
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
    subscriber: &Subscriber<ipc::Service, [u8], ()>,
    batch: &mut Vec<Envelope>,
    counters: &mut LoopCounters,
) {
    loop {
        match subscriber.receive() {
            Ok(Some(sample)) => match Envelope::decode(sample.payload()) {
                // A `BatchedFrames` envelope is a wire-only optimisation: the
                // producer packs N joint samples into one iceoryx2 message.
                // Unpack it here, at the transport boundary, so the dispatcher
                // and per-trace actors only ever see standalone `Frame`s.
                Ok(Envelope::BatchedFrames {
                    timestamp_ns,
                    timestamp_s,
                    frames,
                }) => {
                    for item in frames {
                        batch.push(Envelope::frame(
                            item.trace_id,
                            item.sequence_number,
                            timestamp_ns,
                            timestamp_s,
                            item.payload,
                        ));
                    }
                }
                Ok(envelope) => batch.push(envelope),
                Err(error) => {
                    counters.decode_failures = counters.decode_failures.saturating_add(1);
                    tracing::warn!(
                        %error,
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
                    receive_failures = counters.receive_failures,
                    "ipc subscriber receive failed"
                );
                return;
            }
        }
    }
}
