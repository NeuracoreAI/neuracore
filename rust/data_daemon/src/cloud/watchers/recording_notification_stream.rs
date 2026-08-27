//! Backend → daemon recording lifecycle.
//!
//! The daemon reads web-initiated start and stop from the backend directly,
//! over the same authenticated, org-scoped connection it already POSTs
//! `/recording/{start,stop}` on.
//!
//! ## Why the daemon, and not the producers
//!
//! A web-initiated start or stop is one event. Delivered to N producer
//! processes it becomes N lifecycle envelopes, and each process then has to
//! work out *which* recording it means — which it cannot, because the daemon
//! owns recording identity. Every attempt to give a producer that knowledge
//! (a query, an announcement lease, a local map, an ownership flag) is a
//! reconstruction of something that was never ambiguous at the source.
//!
//! So the notification is consumed where identity already lives. Producers are
//! told nothing, publish no lifecycle envelope for it, and hold no claim about
//! any recording. `nc.start_recording()` / `nc.stop_recording()` are untouched:
//! those are one process expressing intent, with no fan-out to disambiguate.
//!
//! ## Shape of the stream
//!
//! `GET /stream/org/{org}/recording/notifications` is Server-Sent Events, framed
//! as `event:data\ndata:{json}\n\n`, with `event:heartbeat\ndata:ping\n\n`
//! keep-alive frames. On connect the backend replays an `INIT` carrying every
//! currently-live recording for the org, so a daemon that restarts — or one
//! whose connection dropped — recovers the full picture without a reconciliation
//! pass of its own. That snapshot is why reconnecting is a complete repair and
//! not just a resumption.
//!
//! Everything the stream reports is org-wide, so it includes robots whose
//! producers run on entirely different machines. The dispatcher drops anything
//! for a source with no producer attached here; see
//! [`crate::pipeline::dispatcher::RecordingCommand`].

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderValue, ACCEPT, AUTHORIZATION};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::pipeline::dispatcher::RecordingCommand;

/// How long to wait before redialling after the stream ends or fails.
///
/// The backend replays a full `INIT` snapshot on connect, so a reconnect is a
/// repair rather than a resumption and there is no urgency to redial instantly —
/// but a recording cannot start on this machine while the stream is down, so
/// this stays short.
const RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// How long to wait for the org id to appear before re-checking.
///
/// A daemon can outlive `nc select-org`, so "no org yet" is a normal startup
/// state rather than an error.
const ORG_WAIT: Duration = Duration::from_secs(1);

/// Connect timeout for the stream. Deliberately the *only* timeout: the request
/// itself must be allowed to stay open indefinitely, which is the whole point.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Largest SSE frame buffer to hold before giving up on the connection.
///
/// An `INIT` carrying every live recording in a busy org is the biggest frame
/// that can legitimately arrive, and is still far below this. A buffer growing
/// past it means the framing is wrong (a proxy rewriting the body, say), which a
/// reconnect handles better than unbounded growth.
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Handle for the notification-stream task.
pub struct RecordingNotificationHandle {
    join: JoinHandle<()>,
}

impl RecordingNotificationHandle {
    /// Wait for the task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "recording notification stream join failed");
        }
    }
}

/// Lifecycle notification types the backend emits. Everything else is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum NotificationType {
    /// Snapshot of every live recording, replayed on connect.
    #[serde(rename = "INIT")]
    Init,
    #[serde(rename = "START")]
    Start,
    #[serde(rename = "STOP")]
    Stop,
    /// A recording discarded rather than kept — still an end to the window.
    #[serde(rename = "DISCARDED")]
    Discarded,
    /// A recording that ran past the backend's maximum duration.
    #[serde(rename = "EXPIRED")]
    Expired,
    /// Post-stop disposition; the window is already closed by the time it lands.
    #[serde(rename = "SAVED")]
    Saved,
}

/// The envelope every notification arrives in. `payload` stays untyped until the
/// type is known, because `INIT` carries a *list* where the rest carry one
/// object.
#[derive(Debug, Deserialize)]
struct Notification {
    #[serde(rename = "type")]
    kind: NotificationType,
    payload: serde_json::Value,
}

/// A recording the backend reports as started.
#[derive(Debug, Deserialize)]
struct StartPayload {
    recording_id: String,
    robot_id: String,
    instance: i64,
    start_time: f64,
    #[serde(default)]
    dataset_ids: Vec<String>,
}

/// The minimum any non-start notification carries.
#[derive(Debug, Deserialize)]
struct UpdatePayload {
    recording_id: String,
    robot_id: String,
    instance: i64,
}

/// Spawn the notification stream reader.
///
/// Spawned only when the daemon is online — offline has no backend to hear from,
/// and no web-initiated recording can exist there, so the local
/// `nc.start_recording()` path is the only one and needs nothing from here.
pub fn spawn_recording_notification_stream(
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    command_tx: mpsc::Sender<RecordingCommand>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RecordingNotificationHandle {
    let join = tokio::spawn(async move {
        let stream_client = match Client::builder().connect_timeout(CONNECT_TIMEOUT).build() {
            Ok(stream_client) => stream_client,
            Err(error) => {
                tracing::warn!(
                    %error,
                    "failed to build the notification stream client; \
                     web-initiated recordings will not reach this daemon"
                );
                return;
            }
        };

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording notification stream shutting down");
                    return;
                }
                outcome = consume_stream(&stream_client, &client, &org_rx, &command_tx) => {
                    match outcome {
                        // A clean end still means we are no longer being told
                        // about recordings, so it is redialled like a failure.
                        Ok(()) => tracing::debug!("recording notification stream ended; redialling"),
                        Err(error) => {
                            tracing::warn!(%error, "recording notification stream failed; redialling")
                        }
                    }
                }
            }

            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording notification stream shutting down");
                    return;
                }
                _ = tokio::time::sleep(RECONNECT_DELAY) => {}
            }
        }
    });

    RecordingNotificationHandle { join }
}

/// Open one connection and read it until it ends.
async fn consume_stream(
    stream_client: &Client,
    api: &Arc<ApiClient>,
    org_rx: &OrgIdRx,
    command_tx: &mpsc::Sender<RecordingCommand>,
) -> Result<(), String> {
    let Some(org_id) = org_rx.borrow().clone() else {
        // Not an error: the daemon can start before an org is selected.
        tokio::time::sleep(ORG_WAIT).await;
        return Ok(());
    };

    let url = api.url(&format!("/stream/org/{org_id}/recording/notifications"));
    let token = api
        .auth()
        .bearer_token()
        .await
        .map_err(|error| format!("no bearer token: {error}"))?;
    let authorization = HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|_| "bearer token contains invalid header characters".to_string())?;

    let response = stream_client
        .get(&url)
        .header(AUTHORIZATION, authorization)
        .header(ACCEPT, HeaderValue::from_static("text/event-stream"))
        .send()
        .await
        .map_err(|error| format!("connect failed: {error}"))?;

    if response.status() == StatusCode::UNAUTHORIZED {
        // Drop the cached JWT so the redial exchanges a fresh one, matching the
        // API client's own 401 handling.
        if let Err(error) = api.auth().reload().await {
            tracing::debug!(%error, "failed to reload auth after 401");
        }
        return Err("unauthorized".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("unexpected status {}", response.status()));
    }

    tracing::info!(org_id, "listening for recording notifications");
    read_frames(response, command_tx).await
}

/// Read SSE frames off an open response until it ends.
async fn read_frames(
    mut response: reqwest::Response,
    command_tx: &mpsc::Sender<RecordingCommand>,
) -> Result<(), String> {
    // `chunk()` is an inherent method on `Response` under reqwest's `stream`
    // feature, so framing needs no `Stream` adapter crate.
    let mut buffer = String::new();
    loop {
        let chunk = response
            .chunk()
            .await
            .map_err(|error| format!("read failed: {error}"))?;
        let Some(chunk) = chunk else { return Ok(()) };
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(split_at) = buffer.find("\n\n") {
            let frame = buffer[..split_at].to_string();
            buffer.drain(..split_at + 2);
            for command in commands_from_frame(&frame) {
                if command_tx.send(command).await.is_err() {
                    return Err("dispatcher command inbox closed".to_string());
                }
            }
        }

        if buffer.len() > MAX_FRAME_BYTES {
            return Err(format!(
                "frame exceeded {MAX_FRAME_BYTES} bytes without a terminator"
            ));
        }
    }
}

/// Turn one SSE frame into the lifecycle commands it implies.
///
/// A frame yields more than one command only for `INIT`, which carries every
/// live recording at once. Anything unparseable is dropped with a log rather
/// than killing the connection: one malformed notification must not stop the
/// daemon hearing about the next recording.
fn commands_from_frame(frame: &str) -> Vec<RecordingCommand> {
    let mut lines = Vec::new();
    for line in frame.lines() {
        if let Some(value) = line.strip_prefix("data:") {
            // The SSE spec strips a single leading space; the backend emits
            // none, so this handles both. Repeated `data:` lines join with a
            // newline, per the spec — the backend sends one, and JSON does not
            // care either way.
            lines.push(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    let data = lines.join("\n");
    // A keep-alive frame carries a literal `ping`, not JSON.
    if data.is_empty() || data == "ping" {
        return Vec::new();
    }

    let notification: Notification = match serde_json::from_str(&data) {
        Ok(notification) => notification,
        Err(error) => {
            tracing::debug!(%error, "dropping unparseable recording notification");
            return Vec::new();
        }
    };

    match notification.kind {
        NotificationType::Init => {
            match serde_json::from_value::<Vec<StartPayload>>(notification.payload) {
                Ok(payloads) => payloads.into_iter().map(start_command).collect(),
                Err(error) => {
                    tracing::debug!(%error, "dropping malformed INIT payload");
                    Vec::new()
                }
            }
        }
        NotificationType::Start => {
            match serde_json::from_value::<StartPayload>(notification.payload) {
                Ok(payload) => vec![start_command(payload)],
                Err(error) => {
                    tracing::debug!(%error, "dropping malformed START payload");
                    Vec::new()
                }
            }
        }
        NotificationType::Stop | NotificationType::Discarded | NotificationType::Expired => {
            match serde_json::from_value::<UpdatePayload>(notification.payload) {
                Ok(payload) => vec![RecordingCommand::Stop {
                    source: (payload.robot_id, payload.instance),
                    recording_id: payload.recording_id,
                }],
                Err(error) => {
                    tracing::debug!(%error, "dropping malformed stop payload");
                    Vec::new()
                }
            }
        }
        // The window is already closed by the time this lands.
        NotificationType::Saved => Vec::new(),
    }
}

fn start_command(payload: StartPayload) -> RecordingCommand {
    RecordingCommand::Start {
        source: (payload.robot_id, payload.instance),
        recording_id: payload.recording_id,
        // A recording is created in exactly one dataset; take the first rather
        // than rejecting a payload that grows a second.
        dataset_id: payload.dataset_ids.into_iter().next(),
        start_time_s: payload.start_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a frame the way the backend does: `event:data`, then the JSON on
    /// one `data:` line.
    fn frame(json: &str) -> String {
        let single_line = json.replace('\n', "");
        format!("event:data\ndata:{single_line}")
    }

    #[test]
    fn a_heartbeat_yields_nothing() {
        assert!(commands_from_frame("event:heartbeat\ndata:ping").is_empty());
        assert!(commands_from_frame("").is_empty());
    }

    #[test]
    fn a_start_becomes_a_start_command() {
        let commands = commands_from_frame(&frame(
            r#"{"type":"START","id":"n1","payload":{
                "recording_id":"rec-1","robot_id":"robot-1","instance":0,
                "created_by":"user-1","dataset_ids":["ds-1"],
                "data_types":[],"start_time":1700000000.5}}"#,
        ));
        match commands.as_slice() {
            [RecordingCommand::Start {
                source,
                recording_id,
                dataset_id,
                start_time_s,
            }] => {
                assert_eq!(source, &("robot-1".to_string(), 0));
                assert_eq!(recording_id, "rec-1");
                assert_eq!(dataset_id.as_deref(), Some("ds-1"));
                assert_eq!(*start_time_s, 1_700_000_000.5);
            }
            other => panic!("expected one start command, got {other:?}"),
        }
    }

    #[test]
    fn init_replays_every_live_recording() {
        // The reconnect repair path: one frame, N recordings.
        let commands = commands_from_frame(&frame(
            r#"{"type":"INIT","id":"n0","payload":[
                {"recording_id":"rec-1","robot_id":"robot-1","instance":0,
                 "created_by":"u","dataset_ids":["ds-1"],"data_types":[],
                 "start_time":1700000000.0},
                {"recording_id":"rec-2","robot_id":"robot-2","instance":3,
                 "created_by":"u","dataset_ids":["ds-2"],"data_types":[],
                 "start_time":1700000001.0}]}"#,
        ));
        assert_eq!(commands.len(), 2);
        assert!(matches!(
            &commands[1],
            RecordingCommand::Start { source, recording_id, .. }
                if source == &("robot-2".to_string(), 3) && recording_id == "rec-2"
        ));
    }

    #[test]
    fn every_ending_notification_stops_the_window() {
        // A discarded or expired recording ends its window exactly as a stop
        // does — the data stops being accepted either way.
        for kind in ["STOP", "DISCARDED", "EXPIRED"] {
            let commands = commands_from_frame(&frame(&format!(
                r#"{{"type":"{kind}","id":"n2","payload":{{
                    "recording_id":"rec-1","robot_id":"robot-1","instance":0}}}}"#
            )));
            assert!(
                matches!(
                    commands.as_slice(),
                    [RecordingCommand::Stop { source, recording_id }]
                        if source == &("robot-1".to_string(), 0) && recording_id == "rec-1"
                ),
                "{kind} must end the window, got {commands:?}"
            );
        }
    }

    #[test]
    fn saved_is_not_a_window_event() {
        let commands = commands_from_frame(&frame(
            r#"{"type":"SAVED","id":"n3","payload":{
                "recording_id":"rec-1","robot_id":"robot-1","instance":0}}"#,
        ));
        assert!(commands.is_empty(), "SAVED lands after the window closed");
    }

    #[test]
    fn a_malformed_notification_is_dropped_not_fatal() {
        // One bad frame must not cost us the next recording.
        assert!(commands_from_frame(&frame("{not json")).is_empty());
        assert!(commands_from_frame(&frame(r#"{"type":"START","payload":{}}"#)).is_empty());
        assert!(commands_from_frame(&frame(r#"{"type":"WHAT","payload":{}}"#)).is_empty());
    }
}
