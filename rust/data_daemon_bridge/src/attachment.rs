//! Which sources THIS process is bound to.
//!
//! Deliberately not recording state. Nothing here knows whether a recording is
//! in progress, which one, or whether this process started it — the daemon owns
//! all of that. This records only that a process exists and is bound to a
//! source, which is a fact about the process and cannot be wrong about a
//! recording.
//!
//! It is published as [`Envelope::ProducerAttached`], and the daemon uses it to
//! tell whether an org-wide backend recording notification concerns a source
//! with a producer on this machine. A source nothing has attached to never has a
//! window opened on its behalf.
//!
//! Restated on a heartbeat from the publisher thread (see
//! [`crate::publisher::publish_loop`]) so a daemon that starts after its
//! producers still learns they are there.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

/// A source key, matching the daemon's own `(robot_id, robot_instance)`.
pub(crate) type Source = (String, i64);

struct Registry {
    owner_pid: u32,
    attached: HashSet<Source>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        owner_pid: 0,
        attached: HashSet::new(),
    })
});

/// Lock the registry, healing it across a `fork()` first.
///
/// A forked child inherits the parent's set but is a different process under a
/// different pid. Without this it would heartbeat under its own pid for sources
/// only its parent ever bound, and the daemon would hold a source present for a
/// producer that never attached.
fn with_registry<R>(operation: impl FnOnce(&mut HashSet<Source>) -> R) -> R {
    let mut registry = REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.attached.clear();
        registry.owner_pid = pid;
    }
    operation(&mut registry.attached)
}

/// Record that this process is bound to the source. `true` when this is a new
/// binding, so the caller announces it immediately rather than waiting a
/// heartbeat.
pub(crate) fn attach(robot_id: &str, robot_instance: i64) -> bool {
    with_registry(|attached| attached.insert((robot_id.to_string(), robot_instance)))
}

/// Forget this process's binding to the source. It drops out of the heartbeat, and
/// the daemon drops it once the liveness bound elapses.
pub(crate) fn detach(robot_id: &str, robot_instance: i64) {
    with_registry(|attached| {
        attached.remove(&(robot_id.to_string(), robot_instance));
    });
}

/// Every source this process is currently bound to, for the heartbeat.
pub(crate) fn attached_sources() -> Vec<Source> {
    with_registry(|attached| attached.iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attaching_is_reported_once_per_source() {
        assert!(attach("attach-once", 0), "first bind is new");
        assert!(!attach("attach-once", 0), "re-binding is not new");
        detach("attach-once", 0);
    }

    #[test]
    fn a_detached_source_drops_out_of_the_heartbeat() {
        attach("attach-detach", 7);
        assert!(attached_sources().contains(&("attach-detach".to_string(), 7)));
        detach("attach-detach", 7);
        assert!(!attached_sources().contains(&("attach-detach".to_string(), 7)));
    }

    #[test]
    fn instances_of_one_robot_attach_separately() {
        attach("attach-instances", 0);
        attach("attach-instances", 1);
        detach("attach-instances", 0);
        let sources = attached_sources();
        assert!(!sources.contains(&("attach-instances".to_string(), 0)));
        assert!(sources.contains(&("attach-instances".to_string(), 1)));
        detach("attach-instances", 1);
    }
}
