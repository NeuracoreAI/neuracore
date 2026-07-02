//! Live daemon-profile resolution.
//!
//! The effective [`DaemonConfig`] (profile YAML + `NCD_*` env overlay) is held
//! in memory and refreshed on an interval, so consumers — the per-trace actors
//! and the registration coordinator — read the current video codec from a
//! [`watch::channel`] instead of re-parsing the profile YAML on every trace.
//! This mirrors the org watcher (`org_watcher.rs`), which does the same for
//! `config.json`.
//!
//! Two things drive a refresh: the periodic [`crate::intervals::CONFIG_POLL`]
//! tick (covers CLI `profile update` and external edits), and an on-demand
//! request over an `mpsc`. The dispatcher sends the latter for a
//! [`crate::Envelope::RefreshConfig`] command and awaits the paired `oneshot`
//! ack, so the SDK's `set_video_encoding_options → start_recording` sequence
//! observes the new codec before the recording's traces resolve it.

use std::path::{Path, PathBuf};

use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::config::profile::ProfileManager;
use crate::config::{resolve_effective_config, DaemonConfig};
use crate::lifecycle::shutdown::ShutdownSignal;

/// Shared read handle for the current effective [`DaemonConfig`]. Cheap to
/// clone; read the current value with `config_rx.borrow()`.
pub type ConfigRx = watch::Receiver<DaemonConfig>;

/// A refresh request: the watcher fires the paired `oneshot` once the
/// re-resolve has been published, letting the caller order subsequent work
/// (e.g. dispatching `StartRecording`) strictly after the config update.
pub type ConfigRefreshRequest = oneshot::Sender<()>;

/// Handle for the config-watcher task.
pub struct ConfigWatcherHandle {
    join: JoinHandle<()>,
}

impl ConfigWatcherHandle {
    /// Wait for the watcher task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "config watcher join failed");
        }
    }
}

/// Spawn the config watcher.
///
/// The caller owns the [`watch::channel`] (seeded with the launch-resolved
/// config, so the initial value is available before this task's first tick) and
/// passes the [`watch::Sender`] here; the matching [`ConfigRx`] is handed to the
/// actor context and the registration coordinator. `refresh_rx` receives
/// on-demand refresh requests; `profile` is the active profile name to resolve.
/// `home` overrides the profiles root — `None` uses the real user home (the
/// production path); tests pass a temp dir to avoid touching env or the
/// developer's profile.
pub fn spawn_config_watcher(
    profile: Option<String>,
    home: Option<PathBuf>,
    config_tx: watch::Sender<DaemonConfig>,
    mut refresh_rx: mpsc::Receiver<ConfigRefreshRequest>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> ConfigWatcherHandle {
    let join = tokio::spawn(async move {
        let mut ticker = interval(crate::intervals::CONFIG_POLL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // Disabled once all refresh senders drop, so a closed channel can't
        // busy-loop the `select!` (a closed `mpsc::recv` returns immediately).
        let mut refresh_open = true;

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "config watcher shutting down");
                    break;
                }
                maybe_request = refresh_rx.recv(), if refresh_open => {
                    match maybe_request {
                        Some(ack) => {
                            refresh(&config_tx, profile.as_deref(), home.as_deref()).await;
                            // Best-effort: the requester may have timed out.
                            let _ = ack.send(());
                        }
                        None => refresh_open = false,
                    }
                }
                _ = ticker.tick() => {
                    refresh(&config_tx, profile.as_deref(), home.as_deref()).await;
                }
            }
        }
    });

    ConfigWatcherHandle { join }
}

/// Re-resolve the effective config off the runtime (the YAML read is blocking)
/// and publish it when it changed. A resolve failure keeps the last-good value.
async fn refresh(
    config_tx: &watch::Sender<DaemonConfig>,
    profile: Option<&str>,
    home: Option<&Path>,
) {
    let profile = profile.map(str::to_owned);
    let home = home.map(Path::to_path_buf);
    let resolved = tokio::task::spawn_blocking(move || {
        let profiles = match home {
            Some(home) => ProfileManager::with_home(home),
            None => ProfileManager::new(),
        };
        resolve_effective_config(&profiles, profile.as_deref(), None)
    })
    .await;

    match resolved {
        Ok(Ok(config)) => {
            config_tx.send_if_modified(|existing| {
                if *existing == config {
                    false
                } else {
                    tracing::info!("daemon profile change picked up; updating in-memory config");
                    *existing = config;
                    true
                }
            });
        }
        Ok(Err(error)) => {
            tracing::warn!(%error, "failed to resolve daemon config; keeping last-good value");
        }
        Err(error) => {
            tracing::warn!(%error, "config resolve task panicked; keeping last-good value");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use tempfile::TempDir;
    use tokio::time::timeout;

    fn channels() -> (
        watch::Sender<DaemonConfig>,
        ConfigRx,
        mpsc::Sender<ConfigRefreshRequest>,
        mpsc::Receiver<ConfigRefreshRequest>,
    ) {
        let (config_tx, config_rx) = watch::channel(DaemonConfig::default());
        let (refresh_tx, refresh_rx) = mpsc::channel(4);
        (config_tx, config_rx, refresh_tx, refresh_rx)
    }

    #[tokio::test]
    async fn on_demand_refresh_picks_up_profile_codec() {
        // Write a lossy codec into a profile under a temp home (no env / real
        // profile touched), then force a refresh and assert the in-memory copy
        // reflects it once the ack fires.
        let home = TempDir::new().expect("temp home");
        let manager = ProfileManager::with_home(home.path().to_path_buf());
        manager.create_profile("default_profile").expect("create");
        manager
            .update_profile(
                "default_profile",
                &DaemonConfig {
                    video_codec: Some("h264_medium".to_string()),
                    ..DaemonConfig::default()
                },
            )
            .expect("update");

        let (config_tx, config_rx, refresh_tx, refresh_rx) = channels();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_config_watcher(
            Some("default_profile".to_string()),
            Some(home.path().to_path_buf()),
            config_tx,
            refresh_rx,
            shutdown_tx.subscribe(),
        );

        let (ack_tx, ack_rx) = oneshot::channel();
        refresh_tx.send(ack_tx).await.expect("send refresh");
        timeout(Duration::from_secs(5), ack_rx)
            .await
            .expect("ack within 5s")
            .expect("ack sender alive");
        assert_eq!(
            config_rx.borrow().video_codec.as_deref(),
            Some("h264_medium")
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn periodic_tick_picks_up_profile_change() {
        // The interval branch (not just the on-demand refresh) must pick up an
        // external profile edit, and `send_if_modified` must publish it — proven
        // by `changed()` firing only on a real change.
        let home = TempDir::new().expect("temp home");
        let manager = ProfileManager::with_home(home.path().to_path_buf());
        manager.create_profile("default_profile").expect("create");
        manager
            .update_profile(
                "default_profile",
                &DaemonConfig {
                    video_codec: Some("h264_medium".to_string()),
                    ..DaemonConfig::default()
                },
            )
            .expect("update");

        // Seed with the default (no codec); the tick must overwrite it.
        let (config_tx, mut config_rx) = watch::channel(DaemonConfig::default());
        let (_refresh_tx, refresh_rx) = mpsc::channel(4);
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_config_watcher(
            Some("default_profile".to_string()),
            Some(home.path().to_path_buf()),
            config_tx,
            refresh_rx,
            shutdown_tx.subscribe(),
        );

        timeout(Duration::from_secs(5), config_rx.changed())
            .await
            .expect("periodic tick published a change within 5s")
            .expect("sender alive");
        assert_eq!(
            config_rx.borrow().video_codec.as_deref(),
            Some("h264_medium")
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn seed_value_is_available_immediately() {
        // The caller seeds the channel, so the current value is readable before
        // the watcher's first tick (no initial resolve latency).
        let (config_tx, config_rx) = watch::channel(DaemonConfig {
            video_codec: Some("h264_medium".to_string()),
            ..DaemonConfig::default()
        });
        let (_refresh_tx, refresh_rx) = mpsc::channel(4);
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle =
            spawn_config_watcher(None, None, config_tx, refresh_rx, shutdown_tx.subscribe());

        assert_eq!(
            config_rx.borrow().video_codec.as_deref(),
            Some("h264_medium")
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn missing_profile_keeps_last_good_value() {
        // Pointing at a home with no profile yields a resolve error; the watcher
        // must keep the seeded value rather than clobbering it.
        let home = TempDir::new().expect("temp home");
        let (config_tx, config_rx) = watch::channel(DaemonConfig {
            video_codec: Some("h264_medium".to_string()),
            ..DaemonConfig::default()
        });
        let (refresh_tx, refresh_rx) = mpsc::channel(4);
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_config_watcher(
            Some("does_not_exist".to_string()),
            Some(home.path().to_path_buf()),
            config_tx,
            refresh_rx,
            shutdown_tx.subscribe(),
        );

        let (ack_tx, ack_rx) = oneshot::channel();
        refresh_tx.send(ack_tx).await.expect("send refresh");
        timeout(Duration::from_secs(5), ack_rx)
            .await
            .expect("ack within 5s")
            .expect("ack sender alive");
        assert_eq!(
            config_rx.borrow().video_codec.as_deref(),
            Some("h264_medium"),
            "a failed resolve must not clobber the last-good value"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
