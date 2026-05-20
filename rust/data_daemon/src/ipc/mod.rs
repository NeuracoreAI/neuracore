//! iceoryx2 transport bring-up and async listener.
//!
//! Phase 4 owns:
//!
//! - [`node`]: creates the per-daemon iceoryx2 [`Node`](iceoryx2::node::Node)
//!   and opens the `commands` and `frames` services defined in
//!   [`data_daemon_ipc::service_name`].
//! - [`listener`]: a tokio task that polls both subscribers on a short
//!   interval and forwards decoded [`Envelope`](data_daemon_ipc::Envelope)s to
//!   the per-trace dispatcher via an `mpsc::Sender`.
//!
//! The polling cadence (10 ms) is a deliberate phase 4 choice: iceoryx2 0.8
//! does not yet ship a `tokio::sync::Notify` adaptor, so we trade a small
//! amount of wakeup latency for a simple implementation. Phase 5 replaces this
//! with a `WaitSet`-backed notifier once frame throughput becomes relevant.

pub mod listener;
pub mod node;
