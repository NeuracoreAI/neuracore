//! Producer-side tracing, so this crate's own diagnostics can be seen.
//!
//! Without a subscriber every `tracing` event here is dropped: only the daemon
//! binary installs one, and the producer is a library living inside somebody
//! else's process. The events that most need to arrive are the ones reporting a
//! *silent* loss — a gate that cannot subscribe to window state records nothing
//! for a source it did not open, and looks exactly like an idle sensor.
//!
//! Hence a default of `warn` on stderr rather than off: the failures worth a
//! line are the ones a caller would otherwise never learn about. `RUST_LOG`
//! overrides the filter and `NDD_DEBUG` raises the default to `debug`, matching
//! the daemon's own `init_tracing`.

use std::sync::Once;

use data_daemon_shared::config::env::RuntimeEnv;

static INIT: Once = Once::new();

/// Install this process's subscriber, at most once.
///
/// `try_init` rather than `init`: an embedding application — or a test harness —
/// may already own the global subscriber, and losing our events to theirs is far
/// better than panicking inside a Python import.
pub(crate) fn init() {
    INIT.call_once(|| {
        let default_level = if RuntimeEnv::from_env().debug {
            "debug"
        } else {
            "warn"
        };
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_writer(std::io::stderr)
            .try_init();
    });
}
