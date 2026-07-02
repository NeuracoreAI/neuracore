//! Periodic watchers: the org-id config poller that publishes the live org for
//! every coordinator, the daemon-profile config poller that publishes the live
//! effective config (chiefly the video codec), and the recording reaper that
//! reclaims durably-settled recordings.

pub mod config_watcher;
pub mod org_watcher;
pub mod recording_reaper;
