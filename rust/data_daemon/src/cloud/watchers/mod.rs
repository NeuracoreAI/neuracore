//! Periodic watchers: the org-id config poller that publishes the live org for
//! every coordinator, and the recording reaper that reclaims durably-settled
//! recordings.

pub mod org_watcher;
pub mod recording_reaper;
