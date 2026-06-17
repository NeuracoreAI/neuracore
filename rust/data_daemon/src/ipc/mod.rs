//! iceoryx2 transport bring-up and async listener.
//!
//! - [`node`]: creates the per-daemon iceoryx2 [`Node`](iceoryx2::node::Node)
//!   and opens the `commands` and `queries` services defined in
//!   [`data_daemon_ipc::service_name`].
//! - [`listener`]: a tokio task that drains the single `commands` subscriber
//!   and answers `queries` requests, forwarding decoded
//!   [`Envelope`](data_daemon_ipc::Envelope)s to the per-trace dispatcher via
//!   an `mpsc::Sender`.
//!
//! iceoryx2 0.8 does not ship a `tokio::sync::Notify` adaptor, so the listener
//! polls. The cadence decays from `POLL_INTERVAL` (200 µs) to
//! `IDLE_POLL_INTERVAL` (25 ms) once the subscriber has been empty for a while,
//! keeping active-load latency low while leaving idle daemons near-quiescent.

pub mod listener;
pub mod node;
