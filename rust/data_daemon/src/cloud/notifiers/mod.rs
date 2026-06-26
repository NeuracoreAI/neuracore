//! Backend recording-lifecycle notifiers (start / stop / cancel), each built on
//! the shared notifier framework in [`notifier`] that subscribes to the event
//! bus and POSTs the matching `/recording/*` endpoint.

pub mod notifier;
pub mod recording_cancel_notifier;
pub mod recording_start_notifier;
pub mod recording_stop_notifier;
