//! Periodic watchers: the org-id config poller that publishes the live org for
//! every coordinator, the daemon-profile config poller that publishes the live
//! effective config (chiefly the video codec), the recording reaper that
//! reclaims durably-settled recordings, and the backend recording-notification
//! stream that delivers web-initiated start and stop straight to the daemon.

pub mod config_watcher;
pub mod org_watcher;
pub mod recording_notification_stream;
pub mod recording_reaper;
