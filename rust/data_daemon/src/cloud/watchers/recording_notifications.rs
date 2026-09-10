//! Backend recording-lifecycle stream.
//!
//! Subscribes to `/stream/org/{org}/recording/notifications` — the same
//! server-sent-event stream the SDK consumes — and turns the recordings the
//! backend reports into [`RecordingCommand`]s for the dispatcher.
//!
//! Producers announce the recordings they bracket themselves; this carries the
//! ones nobody local brackets, started or stopped from the web. Reading the
//! stream here keeps the recording's identity attached all the way to the
//! dispatcher, which acts on the named recording or not at all.
//!
//! Availability, not correctness: producer-bracketed recordings work whether or
//! not this task is connected. A dropped connection is retried with capped
//! backoff.

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::pipeline::dispatcher::RecordingCommand;

/// Backoff floor after a failed or dropped connection.
const RECONNECT_MIN: Duration = Duration::from_secs(1);
/// Backoff ceiling.
const RECONNECT_MAX: Duration = Duration::from_secs(30);
/// Cap on a single SSE frame; anything larger is a malformed stream.
const MAX_FRAME_BYTES: usize = 256 * 1024;

/// Handle for the notification-stream task.
pub struct RecordingNotificationsHandle {
    join: JoinHandle<()>,
}

impl RecordingNotificationsHandle {
    /// Wait for the task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "recording notifications join failed");
        }
    }
}

/// Spawn the notification-stream consumer.
///
/// Returns the receiver the dispatcher selects on alongside its IPC inbox, and
/// the task handle. The channel is bounded so a busy dispatcher slows this task
/// rather than letting notifications queue without limit.
pub fn spawn_recording_notifications(
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (
    mpsc::Receiver<RecordingCommand>,
    RecordingNotificationsHandle,
) {
    let (tx, rx) = mpsc::channel::<RecordingCommand>(64);
    let join = tokio::spawn(run(client, org_rx, tx, shutdown_rx));
    (rx, RecordingNotificationsHandle { join })
}

async fn run(
    client: Arc<ApiClient>,
    mut org_rx: OrgIdRx,
    tx: mpsc::Sender<RecordingCommand>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    let stream_client = match reqwest::Client::builder().build() {
        Ok(stream_client) => stream_client,
        Err(error) => {
            tracing::warn!(%error, "could not build the notification stream client");
            return;
        }
    };

    let mut backoff = RECONNECT_MIN;
    loop {
        let Some(org_id) = org_rx.borrow().clone() else {
            tokio::select! {
                biased;
                _ = shutdown_rx.recv() => return,
                changed = org_rx.changed() => {
                    if changed.is_err() {
                        return;
                    }
                    continue;
                }
            }
        };

        let outcome = tokio::select! {
            biased;
            _ = shutdown_rx.recv() => return,
            changed = org_rx.changed() => {
                if changed.is_err() {
                    return;
                }
                backoff = RECONNECT_MIN;
                continue;
            }
            outcome = consume(&stream_client, &client, &org_id, &tx) => outcome,
        };

        match outcome {
            // A clean end-of-stream is the backend closing an idle connection.
            Ok(()) => {
                tracing::debug!(org_id, "recording notification stream ended; reconnecting");
                backoff = RECONNECT_MIN;
            }
            Err(StreamEnded::Closed) => {
                tracing::debug!("dispatcher gone; stopping recording notifications");
                return;
            }
            Err(StreamEnded::Failed(error)) => {
                tracing::warn!(
                    %error,
                    org_id,
                    backoff_s = backoff.as_secs(),
                    "recording notification stream failed; retrying"
                );
            }
        }

        tokio::select! {
            biased;
            _ = shutdown_rx.recv() => return,
            _ = tokio::time::sleep(backoff) => {}
        }
        backoff = (backoff * 2).min(RECONNECT_MAX);
    }
}

/// Why a subscription stopped.
enum StreamEnded {
    /// The dispatcher's receiver is gone — nothing left to feed.
    Closed,
    /// Connection or protocol failure; the caller retries.
    Failed(String),
}

/// Hold one subscription open, forwarding every recognised event.
///
/// Returns `Ok(())` when the backend closes the stream cleanly.
async fn consume(
    stream_client: &reqwest::Client,
    client: &ApiClient,
    org_id: &str,
    tx: &mpsc::Sender<RecordingCommand>,
) -> Result<(), StreamEnded> {
    let token = client
        .auth()
        .bearer_token()
        .await
        .map_err(|error| StreamEnded::Failed(format!("auth: {error}")))?;
    let url = client.url(&format!("/stream/org/{org_id}/recording/notifications"));

    let response = stream_client
        .get(&url)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .bearer_auth(&token)
        .send()
        .await
        .map_err(|error| StreamEnded::Failed(error.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        if status == reqwest::StatusCode::UNAUTHORIZED {
            let _ = client.auth().reload().await;
        }
        return Err(StreamEnded::Failed(format!("stream returned {status}")));
    }

    tracing::info!(org_id, "subscribed to recording notifications");
    let mut response = response;
    let mut buffer: Vec<u8> = Vec::new();

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| StreamEnded::Failed(error.to_string()))?
    {
        buffer.extend_from_slice(&chunk);
        for frame in take_frames(&mut buffer) {
            let Some(command) = parse_notification(&frame) else {
                continue;
            };
            if tx.send(command).await.is_err() {
                return Err(StreamEnded::Closed);
            }
        }
        if buffer.len() > MAX_FRAME_BYTES {
            return Err(StreamEnded::Failed(format!(
                "notification frame exceeded {MAX_FRAME_BYTES} bytes"
            )));
        }
    }
    Ok(())
}

/// Split every complete SSE frame out of `buffer`, leaving any partial tail.
///
/// A frame ends at a blank line; within one, a repeated `data:` field is joined
/// with newlines. Every other field is ignored — the payload is all in `data`.
fn take_frames(buffer: &mut Vec<u8>) -> Vec<String> {
    let mut frames = Vec::new();
    while let Some(end) = find_frame_end(buffer) {
        let raw = buffer.drain(..end.boundary).collect::<Vec<u8>>();
        let raw = &raw[..end.content];
        if let Ok(text) = std::str::from_utf8(raw) {
            let mut data = String::new();
            for line in text.split('\n') {
                let line = line.strip_suffix('\r').unwrap_or(line);
                let Some(value) = line.strip_prefix("data:") else {
                    continue;
                };
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.strip_prefix(' ').unwrap_or(value));
            }
            if !data.is_empty() {
                frames.push(data);
            }
        }
    }
    frames
}

/// Where the first complete frame ends: `content` bytes of frame, `boundary`
/// bytes to consume including the terminator.
struct FrameEnd {
    content: usize,
    boundary: usize,
}

fn find_frame_end(buffer: &[u8]) -> Option<FrameEnd> {
    for (index, window) in buffer.windows(2).enumerate() {
        if window == b"\n\n" {
            return Some(FrameEnd {
                content: index,
                boundary: index + 2,
            });
        }
    }
    for (index, window) in buffer.windows(4).enumerate() {
        if window == b"\r\n\r\n" {
            return Some(FrameEnd {
                content: index,
                boundary: index + 4,
            });
        }
    }
    None
}

/// The subset of the backend's recording notification this daemon acts on.
#[derive(Debug, Deserialize)]
struct Notification {
    #[serde(rename = "type")]
    kind: String,
    payload: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct StartPayload {
    recording_id: String,
    robot_id: String,
    instance: i64,
    #[serde(default)]
    dataset_ids: Vec<String>,
    start_time: f64,
}

#[derive(Debug, Deserialize)]
struct StopPayload {
    recording_id: String,
}

/// Turn one frame's JSON into a command, or `None` for anything this daemon
/// does not act on (an unparsable frame, or a lifecycle type it ignores).
fn parse_notification(data: &str) -> Option<RecordingCommand> {
    let notification: Notification = match serde_json::from_str(data) {
        Ok(notification) => notification,
        Err(error) => {
            tracing::debug!(%error, "ignoring unparsable recording notification");
            return None;
        }
    };

    match notification.kind.as_str() {
        "START" => {
            let payload: StartPayload = serde_json::from_value(notification.payload).ok()?;
            Some(RecordingCommand::Open {
                cloud_recording_id: payload.recording_id,
                robot_id: payload.robot_id,
                robot_instance: payload.instance,
                dataset_id: payload.dataset_ids.into_iter().next(),
                start_timestamp_ns: seconds_to_nanos(payload.start_time),
            })
        }
        "STOP" | "DISCARDED" | "EXPIRED" => {
            let payload: StopPayload = serde_json::from_value(notification.payload).ok()?;
            Some(RecordingCommand::Close {
                cloud_recording_id: payload.recording_id,
                observed_at_ns: wall_clock_ns(),
            })
        }
        _ => None,
    }
}

fn seconds_to_nanos(seconds: f64) -> i64 {
    (seconds * 1_000_000_000.0) as i64
}

fn wall_clock_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_frames_on_the_blank_line() {
        let mut buffer = b"data: one\n\ndata: two\n\n".to_vec();
        assert_eq!(take_frames(&mut buffer), vec!["one", "two"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn keeps_a_partial_frame_for_the_next_chunk() {
        let mut buffer = b"data: one\n\ndata: hand".to_vec();
        assert_eq!(take_frames(&mut buffer), vec!["one"]);
        buffer.extend_from_slice(b"over\n\n");
        assert_eq!(take_frames(&mut buffer), vec!["handover"]);
    }

    #[test]
    fn joins_repeated_data_fields_and_ignores_other_fields() {
        let mut buffer = b": keepalive\nevent: message\ndata: a\ndata: b\nid: 7\n\n".to_vec();
        assert_eq!(take_frames(&mut buffer), vec!["a\nb"]);
    }

    #[test]
    fn handles_crlf_frames() {
        let mut buffer = b"data: one\r\n\r\n".to_vec();
        assert_eq!(take_frames(&mut buffer), vec!["one"]);
    }

    #[test]
    fn start_becomes_an_open_naming_its_recording() {
        let command = parse_notification(
            r#"{"type":"START","payload":{"recording_id":"rec-1","robot_id":"robot-1",
               "instance":2,"created_by":"someone","dataset_ids":["ds-1"],
               "start_time":1.5}}"#,
        )
        .expect("a start command");
        assert_eq!(
            command,
            RecordingCommand::Open {
                cloud_recording_id: "rec-1".into(),
                robot_id: "robot-1".into(),
                robot_instance: 2,
                dataset_id: Some("ds-1".into()),
                start_timestamp_ns: 1_500_000_000,
            }
        );
    }

    #[test]
    fn every_terminal_type_becomes_a_close() {
        for kind in ["STOP", "DISCARDED", "EXPIRED"] {
            let command = parse_notification(&format!(
                r#"{{"type":"{kind}","payload":{{"recording_id":"rec-1",
                   "robot_id":"robot-1","instance":0}}}}"#
            ))
            .expect("a close command");
            match command {
                RecordingCommand::Close {
                    cloud_recording_id, ..
                } => assert_eq!(cloud_recording_id, "rec-1"),
                other => panic!("{kind} produced {other:?}"),
            }
        }
    }

    #[test]
    fn ignores_init_and_unparsable_frames() {
        assert!(parse_notification(r#"{"type":"INIT","payload":[]}"#).is_none());
        assert!(parse_notification("not json").is_none());
        assert!(parse_notification(r#"{"type":"SAVED","payload":{}}"#).is_none());
    }
}
