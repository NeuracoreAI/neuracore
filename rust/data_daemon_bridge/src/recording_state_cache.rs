//! Per-source recording state, cached from the daemon.
//!
//! One cache, two readers, both of them memory reads of the same entry:
//!
//! * [`epoch`] — has this source crossed into a *different* recording? The log
//!   path asks per frame. The daemon owns recording identity and
//!   [`crate::query`] can ask it, but that is a request-response round trip
//!   bounded by the daemon's inbox poll, far too expensive at frame rate.
//! * [`display`] — what is open right now, and under what cloud id? This is the
//!   whole answer `is_recording` and `get_cloud_recording_id` need; each used
//!   to make its own blocking round trip for a reply this cache already held.
//!
//! Two writers keep it current, and they cover different ground:
//!
//! * **This process's own lifecycle calls.** `start_recording` returns the very
//!   capture timestamp the daemon stores as
//!   [`data_daemon_shared::LiveRecording::start_timestamp_ns`], so a recording
//!   bracketed here is known exactly, the instant it opens, with no IPC at all.
//! * **A refresh through the `recording_state` query**, scheduled off the log
//!   path when an entry goes stale. This is what finds a recording started
//!   somewhere else — the web, or another process — which this process would
//!   otherwise never hear about.
//!
//! What [`epoch`] hands out is a per-source counter, bumped whenever this
//! process observes the source cross a recording boundary — never a property of
//! the recording itself. A start timestamp cannot serve: two recordings may be
//! opened with the same one (`nc.start_recording(timestamp=...)` takes it from
//! the caller), and a caller that saw no change would carry the previous
//! recording's timeline into the new one.
//!
//! Enforcing anything against that timeline is the caller's own: the monotonic
//! check lives per stream, in `DataStream._enforce_monotonic_timestamp`, and
//! only its scope comes from here.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

use data_daemon_shared::{LiveRecording, RecordingStateQuery, Source};

use crate::query::query_recording_state;

/// How stale an entry may get before a read schedules a refresh.
///
/// Bounds how long this process can keep enforcing against a recording that
/// ended elsewhere, or miss one that started there. Well under a human-visible
/// delay, and far above the daemon's own 25 ms inbox poll, so a refresh is
/// answered comfortably within one interval.
const REFRESH_INTERVAL: Duration = Duration::from_millis(50);

/// How long a scheduled refresh may be outstanding before another is allowed.
///
/// Guards the one case the in-flight flag cannot: a `fork()` leaves the child
/// with the parent's entries and none of its refresh thread, so an entry
/// scheduled just before the fork would otherwise never be refreshed again.
const REFRESH_REARM: Duration = Duration::from_secs(1);

/// Bound on a single refresh query. Generous — it runs off the log path, and
/// giving up early just leaves the entry stale for another interval.
const REFRESH_TIMEOUT_S: f64 = 0.1;

/// State nested `robot_id -> instance` rather than keyed on a
/// `(String, i64)` tuple. A tuple key cannot be looked up by borrow, so every
/// read would allocate a `String` purely to build one — on the log path, which
/// the joint fast path was rewritten to keep allocation-free. Nesting costs a
/// second lookup and allocates only when an entry is created.
type Entries = HashMap<String, HashMap<i64, Entry>>;

/// Whether `found` is a different recording from `held`, rather than the same
/// one seen in more detail.
///
/// A refresh routinely learns fields of a recording already held — the daemon
/// assigns `recording_index` when data opens the window, and mints
/// `recording_id` asynchronously after that, both of them after a local
/// `start_recording` returns. Neither is a boundary, and treating one as such
/// would clear a timeline mid-recording. So identity is `recording_index` when
/// both sides have it, and the start timestamp otherwise; the cloud id is
/// deliberately not part of it.
fn is_other(held: &LiveRecording, found: &LiveRecording) -> bool {
    match (held.recording_index, found.recording_index) {
        (Some(held_index), Some(found_index)) => held_index != found_index,
        _ => held.start_timestamp_ns != found.start_timestamp_ns,
    }
}

struct Entry {
    /// The recording this source has open, or `None` for none.
    open: Option<LiveRecording>,
    /// Bumped every time `open` crosses a boundary, and handed out as the
    /// epoch. A counter rather than anything drawn from the recording, so two
    /// recordings sharing a start timestamp are still two.
    epoch: i64,
    /// Bumped on every local write. A refresh carries the value it read before
    /// querying and is discarded if it changed meanwhile, so a daemon answer
    /// gathered before a local stop cannot undo it.
    seq: u64,
    /// When the value was last written by either writer. `None` until the first
    /// answer, which is what makes a brand-new entry refresh immediately.
    written_at: Option<Instant>,
    /// When a refresh was last scheduled, if one is outstanding.
    refresh_scheduled_at: Option<Instant>,
}

static ENTRIES: LazyLock<RwLock<Entries>> = LazyLock::new(|| RwLock::new(Entries::new()));

/// The refresh thread's channel, keyed by owning pid so a forked child spawns
/// its own rather than sending into a thread that did not survive the fork.
struct RefreshThread {
    owner_pid: u32,
    tx: Option<Sender<Source>>,
}

static REFRESH: LazyLock<Mutex<RefreshThread>> = LazyLock::new(|| {
    Mutex::new(RefreshThread {
        owner_pid: 0,
        tx: None,
    })
});

/// This source's boundary counter, or `None` when it has no recording open —
/// or when nothing has answered yet.
///
/// Never blocks and never asks the daemon: a stale entry only schedules the
/// refresh. "Nothing has answered yet" reads the same as "not recording", so a
/// caller gating on this under-enforces for the first refresh rather than
/// acting on a recording that may not exist.
pub(crate) fn epoch(robot_id: &str, robot_instance: i64) -> Option<i64> {
    let (epoch, stale) = {
        let entries = ENTRIES.read().unwrap_or_else(|p| p.into_inner());
        match entries
            .get(robot_id)
            .and_then(|instances| instances.get(&robot_instance))
        {
            Some(entry) => (entry.open.as_ref().map(|_| entry.epoch), is_stale(entry)),
            None => (None, true),
        }
    };
    if stale {
        schedule_refresh((robot_id.to_string(), robot_instance));
    }
    epoch
}

/// The recording this source has open, as this process knows it.
///
/// Three answers, and the caller must keep them apart:
///
/// * `None` — unknown. Nothing has ever answered for this source and asking now
///   did not help, so the daemon is down or not listening. Never read this as
///   "not recording": a caller that does will skip a stop for a recording that
///   is still running.
/// * `Some(None)` — the source has no open recording.
/// * `Some(Some(recording))` — what it has open, cloud id included when minted.
///
/// Blocks *once* per source, and only ever on the first read: an entry nothing
/// has answered for is queried on the calling thread, bounded by
/// [`REFRESH_TIMEOUT_S`]. Every read after that is a memory read against a
/// cache the refresh thread keeps within [`REFRESH_INTERVAL`], which is why a
/// recording started elsewhere still shows up here.
pub(crate) fn display(robot_id: &str, robot_instance: i64) -> Option<Option<LiveRecording>> {
    let (answered, open, stale) = {
        let entries = ENTRIES.read().unwrap_or_else(|p| p.into_inner());
        match entries
            .get(robot_id)
            .and_then(|instances| instances.get(&robot_instance))
        {
            Some(entry) => (
                entry.written_at.is_some(),
                entry.open.clone(),
                is_stale(entry),
            ),
            None => (false, None, true),
        }
    };

    let source = (robot_id.to_string(), robot_instance);
    if answered {
        if stale {
            schedule_refresh(source);
        }
        return Some(open);
    }

    // Nothing has ever answered for this source. Ask on this thread rather than
    // reporting "not recording" for the one read that has no cache to fall back
    // on — the caller cannot tell that apart from a real answer.
    if !refresh_once(&source) {
        return None;
    }
    let entries = ENTRIES.read().unwrap_or_else(|p| p.into_inner());
    Some(lookup(&entries, &source).and_then(|entry| entry.open.clone()))
}

/// Record the recording this process just opened for `source`.
///
/// `started_at_ns` is `start_recording`'s return value, which is exactly what
/// the daemon stores as the recording's start — so this entry already agrees
/// with what a refresh would fetch, bar the ids the daemon has yet to mint.
pub(crate) fn note_local_start(robot_id: &str, robot_instance: i64, started_at_ns: i64) {
    let mut entries = ENTRIES.write().unwrap_or_else(|p| p.into_inner());
    let entry = entry_mut(&mut entries, robot_id, robot_instance);
    // Unconditional, unlike a refresh: this call *is* a new recording, whatever
    // timestamp it carries, so a caller that reuses one still sees a boundary.
    entry.epoch = entry.epoch.wrapping_add(1);
    entry.open = Some(LiveRecording {
        recording_index: None,
        recording_id: None,
        start_timestamp_ns: Some(started_at_ns),
    });
    entry.seq = entry.seq.wrapping_add(1);
    entry.written_at = Some(Instant::now());
}

/// Record that this process just closed `source`'s recording (stop or cancel).
pub(crate) fn note_local_end(robot_id: &str, robot_instance: i64) {
    let mut entries = ENTRIES.write().unwrap_or_else(|p| p.into_inner());
    let entry = entry_mut(&mut entries, robot_id, robot_instance);
    entry.open = None;
    entry.seq = entry.seq.wrapping_add(1);
    entry.written_at = Some(Instant::now());
}

/// The entry for a source, created empty if it has none.
fn entry_mut<'a>(entries: &'a mut Entries, robot_id: &str, robot_instance: i64) -> &'a mut Entry {
    entries
        .entry(robot_id.to_string())
        .or_default()
        .entry(robot_instance)
        .or_insert(Entry {
            open: None,
            epoch: 0,
            seq: 0,
            written_at: None,
            refresh_scheduled_at: None,
        })
}

fn is_stale(entry: &Entry) -> bool {
    entry
        .written_at
        .is_none_or(|written| written.elapsed() >= REFRESH_INTERVAL)
        && entry
            .refresh_scheduled_at
            .is_none_or(|scheduled| scheduled.elapsed() >= REFRESH_REARM)
}

/// Mark `source` as awaiting a refresh and hand it to the refresh thread.
///
/// Re-checks staleness under the write lock so concurrent logging threads
/// queue one refresh between them, not one each.
fn schedule_refresh(source: Source) {
    {
        let mut entries = ENTRIES.write().unwrap_or_else(|p| p.into_inner());
        let entry = entry_mut(&mut entries, &source.0, source.1);
        if !is_stale(entry) {
            return;
        }
        entry.refresh_scheduled_at = Some(Instant::now());
    }
    if let Some(tx) = refresh_tx() {
        let _ = tx.send(source);
    }
}

/// This process's refresh channel, spawning the thread on first use and after
/// a fork. `None` when the spawn failed — the caller drops the refresh, and the
/// re-arm window lets a later read try again.
fn refresh_tx() -> Option<Sender<Source>> {
    let mut registry = REFRESH.lock().unwrap_or_else(|p| p.into_inner());
    let pid = std::process::id();
    if registry.owner_pid == pid {
        if let Some(tx) = registry.tx.as_ref() {
            return Some(tx.clone());
        }
    }
    let (tx, rx) = channel();
    match std::thread::Builder::new()
        .name("nc-recording-state".to_string())
        .spawn(move || refresh_loop(rx))
    {
        Ok(_handle) => {
            registry.owner_pid = pid;
            registry.tx = Some(tx.clone());
            Some(tx)
        }
        Err(error) => {
            tracing::warn!(%error, "failed to spawn recording-state refresh thread");
            None
        }
    }
}

/// Ask the daemon about each queued source and apply the answer.
fn refresh_loop(rx: Receiver<Source>) {
    while let Ok(source) = rx.recv() {
        refresh_once(&source);
    }
}

/// Ask the daemon about one source and apply the answer, reporting whether it
/// answered at all.
///
/// Shared by the refresh thread and [`display`]'s first-read path so the
/// seq-guard is written once: whoever asks, an answer gathered before a local
/// start or stop must lose to it.
fn refresh_once(source: &Source) -> bool {
    let seq_before = {
        let entries = ENTRIES.read().unwrap_or_else(|p| p.into_inner());
        lookup(&entries, source).map(|entry| entry.seq).unwrap_or(0)
    };
    let query = RecordingStateQuery {
        robot_id: source.0.clone(),
        robot_instance: source.1,
    };
    let Ok(bytes) = query.encode() else {
        clear_scheduled(source);
        return false;
    };
    match query_recording_state(&bytes, REFRESH_TIMEOUT_S) {
        // The daemon answered. `None` inside is a real "not recording".
        Ok(Some(live)) => {
            apply_refresh(source, seq_before, live);
            true
        }
        // Nothing answered in time; leave the entry as it was.
        Ok(None) => {
            clear_scheduled(source);
            false
        }
        Err(error) => {
            tracing::debug!(%error, robot_id = source.0, "recording-state refresh failed");
            clear_scheduled(source);
            false
        }
    }
}

fn lookup<'a>(entries: &'a Entries, source: &Source) -> Option<&'a Entry> {
    entries
        .get(&source.0)
        .and_then(|by_instance| by_instance.get(&source.1))
}

fn apply_refresh(source: &Source, seq_before: u64, found: Option<LiveRecording>) {
    let mut entries = ENTRIES.write().unwrap_or_else(|p| p.into_inner());
    let entry = entry_mut(&mut entries, &source.0, source.1);
    entry.refresh_scheduled_at = None;
    if entry.seq != seq_before {
        // A local start or stop landed while this answer was in flight; it
        // speaks for this process's own recording and outranks the daemon's
        // older view.
        return;
    }
    if crossed_a_boundary(entry.open.as_ref(), found.as_ref()) {
        entry.epoch = entry.epoch.wrapping_add(1);
    }
    entry.open = found;
    entry.written_at = Some(Instant::now());
}

/// Whether moving from `held` to `found` leaves one recording for another.
///
/// Not simply inequality: a refresh that fills in a `recording_index` or a
/// cloud id for the recording already held has found the same one, and bumping
/// the epoch there would clear a timeline in the middle of a recording.
fn crossed_a_boundary(held: Option<&LiveRecording>, found: Option<&LiveRecording>) -> bool {
    match (held, found) {
        (None, Some(_)) => true,
        (Some(held), Some(found)) => is_other(held, found),
        // Nothing open now; the epoch a caller sees is `None` either way, so
        // there is no boundary to mark.
        (_, None) => false,
    }
}

fn clear_scheduled(source: &Source) {
    let mut entries = ENTRIES.write().unwrap_or_else(|p| p.into_inner());
    if let Some(by_instance) = entries.get_mut(&source.0) {
        if let Some(entry) = by_instance.get_mut(&source.1) {
            entry.refresh_scheduled_at = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This state is process-global, so every test uses its own source.
    fn source(name: &str) -> Source {
        (name.to_string(), 0)
    }

    fn seq_of(source: &Source) -> u64 {
        let entries = ENTRIES.read().unwrap();
        lookup(&entries, source).map(|entry| entry.seq).unwrap_or(0)
    }

    fn opened(recording_index: Option<i64>, started_at_ns: i64) -> Option<LiveRecording> {
        Some(LiveRecording {
            recording_index,
            recording_id: None,
            start_timestamp_ns: Some(started_at_ns),
        })
    }

    fn opened_with_id(
        recording_index: Option<i64>,
        started_at_ns: i64,
        recording_id: &str,
    ) -> Option<LiveRecording> {
        Some(LiveRecording {
            recording_index,
            recording_id: Some(recording_id.to_string()),
            start_timestamp_ns: Some(started_at_ns),
        })
    }

    #[test]
    fn a_locally_started_recording_is_known_without_asking_the_daemon() {
        // The whole point of seeding from `start_recording`: no daemon runs in
        // this test, and a read still has to answer.
        note_local_start("boundary-local-start", 0, 4_200);
        assert!(epoch("boundary-local-start", 0).is_some());

        note_local_end("boundary-local-start", 0);
        assert_eq!(epoch("boundary-local-start", 0), None);
    }

    #[test]
    fn two_recordings_opened_with_one_timestamp_are_still_two() {
        // `nc.start_recording(timestamp=...)` takes the start from the caller,
        // so consecutive recordings can carry the same one. Reporting the same
        // epoch for both would carry the first's timeline into the second and
        // refuse its opening frames.
        let source = source("boundary-repeated-start");
        note_local_start(&source.0, source.1, 7_000);
        let first = epoch(&source.0, source.1);

        note_local_end(&source.0, source.1);
        note_local_start(&source.0, source.1, 7_000);

        assert!(first.is_some());
        assert_ne!(
            epoch(&source.0, source.1),
            first,
            "a second recording is a second timeline, whatever its clock says"
        );
    }

    #[test]
    fn learning_a_recording_index_is_not_a_boundary() {
        // The daemon assigns the index when data opens the window, which is
        // after a local start returns — so a refresh routinely learns it for a
        // recording already held. Treating that as a new recording would clear
        // the timeline mid-recording and let a backwards timestamp through.
        let source = source("boundary-index-learned");
        note_local_start(&source.0, source.1, 9_000);
        let before = epoch(&source.0, source.1);

        apply_refresh(&source, seq_of(&source), opened(Some(42), 9_000));

        assert_eq!(epoch(&source.0, source.1), before);
    }

    #[test]
    fn learning_a_cloud_id_is_not_a_boundary() {
        // The cloud id is minted asynchronously, later still than the index, so
        // a refresh learns it mid-recording. It must not read as a new
        // recording for exactly the same reason the index must not.
        let source = source("boundary-cloud-id-learned");
        apply_refresh(&source, seq_of(&source), opened(Some(3), 5_000));
        let before = epoch(&source.0, source.1);

        apply_refresh(
            &source,
            seq_of(&source),
            opened_with_id(Some(3), 5_000, "cloud-1"),
        );

        assert_eq!(epoch(&source.0, source.1), before);
    }

    #[test]
    fn the_cloud_id_survives_a_refresh() {
        // The refresh used to project the reply into a boundary-only struct and
        // drop this field, which is what forced `get_cloud_recording_id` to
        // make its own blocking round trip for a reply already in hand.
        let source = source("state-cloud-id-kept");
        apply_refresh(
            &source,
            seq_of(&source),
            opened_with_id(Some(9), 1_500, "cloud-9"),
        );

        let shown = display(&source.0, source.1).expect("the entry has been answered for");
        assert_eq!(
            shown.and_then(|recording| recording.recording_id),
            Some("cloud-9".to_string())
        );
    }

    #[test]
    fn a_local_stop_displays_as_not_recording() {
        // `Some(None)` is a real answer and must stay distinct from `None`.
        let source = source("state-local-stop");
        note_local_start(&source.0, source.1, 2_000);
        note_local_end(&source.0, source.1);

        assert_eq!(display(&source.0, source.1), Some(None));
    }

    #[test]
    fn an_unanswered_source_reads_as_unknown_not_as_not_recording() {
        // No daemon runs in this test, so the inline first read finds nothing.
        // Answering `Some(None)` here would let `nc.stop_recording` skip a
        // recording that is still running.
        let source = source("state-never-answered");

        assert_eq!(
            display(&source.0, source.1),
            None,
            "silence is unknown, never 'not recording'"
        );
    }

    #[test]
    fn a_refresh_gathered_before_a_local_stop_does_not_undo_it() {
        // The daemon's answer can be older than it looks: it is fetched off the
        // log path, and a stop published meanwhile is the newer truth.
        let source = source("boundary-stale-refresh");
        note_local_start(&source.0, source.1, 1_000);
        let seq_at_query = seq_of(&source);

        note_local_end(&source.0, source.1);
        apply_refresh(&source, seq_at_query, opened(None, 1_000));

        assert_eq!(
            epoch(&source.0, source.1),
            None,
            "the stop stands; the in-flight answer is discarded"
        );
    }

    #[test]
    fn a_refresh_finds_a_recording_this_process_never_started() {
        // A recording opened from the web or another process reaches this one
        // only here — nothing local ever wrote it.
        let source = source("boundary-remote-start");
        assert_eq!(epoch(&source.0, source.1), None);

        apply_refresh(&source, seq_of(&source), opened(Some(7), 7_700));

        assert!(epoch(&source.0, source.1).is_some());
    }

    #[test]
    fn a_refresh_onto_a_different_recording_is_a_boundary() {
        let source = source("boundary-remote-boundary");
        apply_refresh(&source, seq_of(&source), opened(Some(1), 100));
        let first = epoch(&source.0, source.1);

        apply_refresh(&source, seq_of(&source), opened(Some(2), 200));

        assert_ne!(epoch(&source.0, source.1), first);
    }

    #[test]
    fn a_fresh_entry_is_not_refreshed_again() {
        let source = source("boundary-staleness");
        note_local_start(&source.0, source.1, 1);

        let entries = ENTRIES.read().unwrap();
        assert!(
            !is_stale(lookup(&entries, &source).unwrap()),
            "a just-written entry must not schedule a refresh"
        );
    }
}
