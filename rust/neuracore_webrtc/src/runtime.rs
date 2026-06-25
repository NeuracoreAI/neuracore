//! The process-global tokio runtime owned and run by the Rust WebRTC core.
//!
//! The Python boundary is deliberately synchronous (see the crate docs): the
//! tokio runtime is created and driven entirely by Rust on its own worker
//! threads, and Python never holds, polls, or awaits anything on it. Every
//! `#[pymethods]` entry point is a plain blocking call that either touches a
//! thread-safe queue or hands work to a task `spawn`ed onto this runtime.
//!
//! The runtime is a lazily-initialised singleton so the first `Producer` or
//! `Consumer` constructed in the process stands it up, and it then lives for
//! the lifetime of the process. There is no shutdown hook: dropping the last
//! handle does not tear the runtime down, which keeps the close path simple and
//! avoids racing a half-torn-down runtime against in-flight tasks.

use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

/// Number of tokio worker threads the core runs on. Two is plenty for the
/// PR0 scaffolding (one stub frame-drain task per producer); PR2+ may revisit
/// this once the real transport tasks land.
const WORKER_THREADS: usize = 2;

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .thread_name("neuracore-webrtc")
        .enable_all()
        .build()
        .expect("failed to build the neuracore-webrtc tokio runtime")
});

/// Return the shared, Rust-owned runtime, initialising it on first use.
pub(crate) fn runtime() -> &'static Runtime {
    &RUNTIME
}

/// Ensure the runtime is stood up. Called from the `Producer`/`Consumer`
/// constructors so the runtime is owned by Rust the moment a peer exists, even
/// if the peer never spawns a task itself (the answer-only `Consumer`).
pub(crate) fn ensure_started() {
    Lazy::force(&RUNTIME);
}
