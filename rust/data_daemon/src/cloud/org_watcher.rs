//! Live `org_id` resolution.
//!
//! The organisation that owns a recording is no longer frozen onto the
//! recording row at creation time. Instead this module watches the
//! SDK-managed `~/.neuracore/config.json` and publishes the *current*
//! `current_org_id` into a [`watch::channel`] that every cloud coordinator
//! reads at the moment it issues a backend POST. A recording opened before
//! the org was selected therefore picks the org up as soon as it lands in
//! config — no daemon restart, and no per-recording backfill.

use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::cloud::read_org_id_from_config;
use crate::lifecycle::signals::ShutdownSignal;

/// Shared read handle for the current `org_id`. Cheap to clone; read the
/// current value with `org_rx.borrow().clone()`.
pub type OrgIdRx = tokio::sync::watch::Receiver<Option<String>>;

/// How often the watcher re-reads the config file. Config writes are rare and
/// user-driven, and the file is tiny, so a coarse poll that re-parses each
/// tick is plenty — cheaper to reason about than mtime gating, which would
/// miss two writes landing within the same mtime granularity (e.g. `login`
/// immediately followed by `set_organization`).
const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Handle for the config-file watcher task.
pub struct OrgWatcherHandle {
    join: JoinHandle<()>,
}

impl OrgWatcherHandle {
    /// Wait for the watcher task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "org watcher join failed");
        }
    }
}

/// Spawn the config-file watcher.
///
/// Returns a [`OrgIdRx`] seeded with the org resolved at spawn time and the
/// task handle. `fallback` is the daemon-profile override (`NCD_CURRENT_ORG_ID`
/// / YAML profile) used whenever the config file has no `current_org_id`,
/// matching the launch-time resolution order.
pub fn spawn_org_watcher(
    config_path: PathBuf,
    fallback: Option<String>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (OrgIdRx, OrgWatcherHandle) {
    let initial = resolve(&config_path, &fallback);
    let (org_tx, org_rx) = watch::channel(initial);

    let join = tokio::spawn(async move {
        let mut ticker = interval(POLL_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "org watcher shutting down");
                    break;
                }
                _ = ticker.tick() => {
                    let current = resolve(&config_path, &fallback);
                    org_tx.send_if_modified(|existing| {
                        if *existing == current {
                            false
                        } else {
                            tracing::info!(
                                org_id = ?current,
                                "config change picked up; updating current org_id"
                            );
                            *existing = current;
                            true
                        }
                    });
                }
            }
        }
    });

    (org_rx, OrgWatcherHandle { join })
}

/// Resolve the org from config, falling back to the daemon-profile override.
fn resolve(config_path: &std::path::Path, fallback: &Option<String>) -> Option<String> {
    read_org_id_from_config(config_path).or_else(|| fallback.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;
    use tokio::time::timeout;

    fn write_config(path: &std::path::Path, org_id: Option<&str>) {
        let body = match org_id {
            Some(org) => format!(r#"{{"current_org_id": "{org}"}}"#),
            None => "{}".to_string(),
        };
        let mut file = std::fs::File::create(path).expect("write config");
        file.write_all(body.as_bytes()).expect("write body");
    }

    #[tokio::test]
    async fn seeds_initial_value_from_config() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        write_config(&path, Some("org-initial"));

        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let (org_rx, handle) = spawn_org_watcher(path, None, shutdown_tx.subscribe());
        assert_eq!(org_rx.borrow().as_deref(), Some("org-initial"));

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn falls_back_when_config_has_no_org() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        write_config(&path, None);

        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let (org_rx, handle) = spawn_org_watcher(
            path,
            Some("profile-org".to_string()),
            shutdown_tx.subscribe(),
        );
        assert_eq!(org_rx.borrow().as_deref(), Some("profile-org"));

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn picks_up_org_written_after_launch() {
        // The recording-blocking case: the daemon comes up before any org is
        // selected, then the SDK writes one. The watcher must publish it
        // without a restart.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        write_config(&path, None);

        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let (mut org_rx, handle) = spawn_org_watcher(path.clone(), None, shutdown_tx.subscribe());
        assert_eq!(org_rx.borrow().as_deref(), None, "starts org-less");

        // Select an org after launch.
        write_config(&path, Some("org-late"));

        timeout(Duration::from_secs(5), org_rx.changed())
            .await
            .expect("watcher must observe the config change within 5s")
            .expect("sender alive");
        assert_eq!(org_rx.borrow().as_deref(), Some("org-late"));

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
