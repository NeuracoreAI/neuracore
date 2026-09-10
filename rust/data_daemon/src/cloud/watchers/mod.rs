//! Periodic watchers: the org-id config poller that publishes the live org for
//! every coordinator, the daemon-profile config poller that publishes the live
//! effective config (chiefly the video codec), and the recording reaper that
//! reclaims durably-settled recordings. Alongside them, the recording
//! notification stream: a long-lived subscription rather than a poller.

pub mod config_watcher;
pub mod org_watcher;
pub mod recording_notifications;
pub mod recording_reaper;
