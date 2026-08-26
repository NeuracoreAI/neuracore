//! Which window, if any, THIS process opened for a source.
//!
//! The daemon is wholly responsible for whether data is written — nothing here
//! gates logging, and nothing here talks to the daemon. The one thing a
//! producer still needs to know locally is what to call the window a later
//! `stop_recording()` from THIS process means, so a duplicate or delayed stop
//! can't close the wrong recording. A process that opens its own window knows
//! that boundary first-hand from the very call that opened it, so this is
//! nothing more than a fork-safe cache of that value.
//!
//! A process that did **not** open the window itself has nothing cached here
//! and asks the daemon instead, once, at `stop_recording()` time — see
//! [`crate::query::resolve_window_started_at`].

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// A source key, matching the daemon's own `(robot_id, robot_instance)`.
pub(crate) type Source = (String, i64);

struct Registry {
    owner_pid: u32,
    /// The `started_at_ns` of the window this process most recently opened,
    /// per source. Absent once closed, or for a source never opened here.
    opened: HashMap<Source, i64>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        owner_pid: 0,
        opened: HashMap::new(),
    })
});

/// Lock the registry, healing it across a `fork()` first.
///
/// A forked child inherits the parent's map but none of its windows: without
/// this, it could claim to have opened a window it never did, and wrongly name
/// a stop against a boundary from before the fork.
fn with_registry<R>(operation: impl FnOnce(&mut HashMap<Source, i64>) -> R) -> R {
    let mut registry = REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.opened.clear();
        registry.owner_pid = pid;
    }
    operation(&mut registry.opened)
}

/// Record that *this* process opened a window for the source, so a later stop
/// from this process can name it without asking the daemon.
pub(crate) fn note_opened(robot_id: &str, robot_instance: i64, started_at_ns: i64) {
    with_registry(|opened| {
        opened.insert((robot_id.to_string(), robot_instance), started_at_ns);
    });
}

/// The `started_at_ns` this process used to open a window for the source, if
/// it was the one that opened it. `None` for a source this process never
/// opened (or has since closed) — the caller falls back to asking the daemon.
pub(crate) fn opened_locally(robot_id: &str, robot_instance: i64) -> Option<i64> {
    with_registry(|opened| opened.get(&(robot_id.to_string(), robot_instance)).copied())
}

/// Forget this process's claim on the source's window.
pub(crate) fn note_closed(robot_id: &str, robot_instance: i64) {
    with_registry(|opened| {
        opened.remove(&(robot_id.to_string(), robot_instance));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_then_closing_forgets_the_window() {
        note_opened("identity-robot", 0, 100);
        assert_eq!(opened_locally("identity-robot", 0), Some(100));
        note_closed("identity-robot", 0);
        assert_eq!(opened_locally("identity-robot", 0), None);
    }

    #[test]
    fn a_never_opened_source_reads_none() {
        assert_eq!(opened_locally("identity-never-opened", 0), None);
    }

    #[test]
    fn reopening_replaces_the_boundary() {
        note_opened("identity-cycling", 0, 100);
        note_opened("identity-cycling", 0, 900);
        assert_eq!(opened_locally("identity-cycling", 0), Some(900));
    }
}
