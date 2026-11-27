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
//! A recording started from the web app names its robot and dataset by their
//! server ids, while this daemon labels a robot's logged data with its robot
//! row. Each start is matched to the robot row that holds the server id, and
//! its dataset to the dataset row that does (made, already confirmed, when
//! there is none). A start that no robot row matches yet is held until its
//! stop, as the robot may connect on this machine, or be registered, after
//! the recording started. It is matched again each time a robot is
//! registered, and dropped when its stop or discard arrives first, or when the
//! snapshot after a reconnect no longer lists it. A start for a robot on
//! another machine is held the same way and never matches.
//!
//! Availability, not correctness: producer-bracketed recordings work whether or
//! not this task is connected. A dropped connection is retried with capped
//! backoff, and every subscribe opens with the backend's `INIT` snapshot, so
//! the recordings that were minted while nothing was listening arrive too.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use data_daemon_shared::seconds_to_microseconds;
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::pipeline::dispatcher::RecordingCommand;
use crate::state::{DaemonEvent, EventBus, SqliteStateStore, StateStore, StateStoreError};

/// Backoff floor after a failed or dropped connection.
const RECONNECT_MIN: Duration = Duration::from_secs(1);
/// Backoff ceiling.
const RECONNECT_MAX: Duration = Duration::from_secs(30);
/// How often a held start is matched again when no robot is registered, in
/// case a registration was missed.
const HOLD_RECHECK: Duration = Duration::from_secs(30);
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
    store: SqliteStateStore,
    bus: EventBus,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (
    mpsc::Receiver<RecordingCommand>,
    RecordingNotificationsHandle,
) {
    let (tx, rx) = mpsc::channel::<RecordingCommand>(64);
    let join = tokio::spawn(run(client, org_rx, Arc::new(store), bus, tx, shutdown_rx));
    (rx, RecordingNotificationsHandle { join })
}

async fn run(
    client: Arc<ApiClient>,
    mut org_rx: OrgIdRx,
    store: Arc<SqliteStateStore>,
    bus: EventBus,
    tx: mpsc::Sender<RecordingCommand>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    // Kept across reconnects: a start held on one subscription is ended by a
    // stop that arrives on the next.
    let held: Held = Arc::default();
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
            outcome = consume(&stream_client, &client, Matcher {
                store: Arc::clone(&store),
                org_id: org_id.clone(),
                tx: tx.clone(),
                held: Arc::clone(&held),
                bus: bus.clone(),
            }) => outcome,
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
#[derive(Debug)]
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
    matcher: Matcher,
) -> Result<(), StreamEnded> {
    let org_id = matcher.org_id.as_str();
    let started = Instant::now();
    let token = client
        .auth()
        .bearer_token()
        .await
        .map_err(|error| StreamEnded::Failed(format!("auth: {error}")))?;
    let auth_ms = started.elapsed().as_millis() as u64;
    let url = client.url(&format!("/stream/org/{org_id}/recording/notifications"));

    let requested = Instant::now();
    let response = stream_client
        .get(&url)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .bearer_auth(&token)
        .send()
        .await
        .map_err(|error| {
            let request_ms = requested.elapsed().as_millis();
            StreamEnded::Failed(format!("{error:?} (after {request_ms} ms)"))
        })?;
    let request_ms = requested.elapsed().as_millis() as u64;

    let status = response.status();
    if !status.is_success() {
        if status == reqwest::StatusCode::UNAUTHORIZED {
            let _ = client.auth().reload().await;
        }
        return Err(StreamEnded::Failed(format!("stream returned {status}")));
    }

    tracing::info!(
        org_id,
        auth_ms,
        request_ms,
        "subscribed to recording notifications"
    );
    let mut response = response;
    let mut buffer: Vec<u8> = Vec::new();
    let mut opened = HashSet::new();

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| StreamEnded::Failed(error.to_string()))?
    {
        buffer.extend_from_slice(&chunk);
        for frame in take_frames(&mut buffer) {
            for mut command in parse_notification(&frame) {
                keep_opened_in_snapshot(&mut command, &mut opened);
                matcher.forward(command).await?;
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

/// Starts held until a robot row matches them, by recording id. An async
/// lock: a held start and a stop are each sent while holding it, so a stop can
/// never overtake the start it ends.
type Held = Arc<tokio::sync::Mutex<HashSet<String>>>;

/// What matching a recording started from the web app to this daemon's rows
/// needs, for one subscription.
#[derive(Clone)]
struct Matcher {
    store: Arc<SqliteStateStore>,
    org_id: String,
    tx: mpsc::Sender<RecordingCommand>,
    held: Held,
    bus: EventBus,
}

impl Matcher {
    /// Send `command` to the dispatcher, a start matched to this daemon's
    /// rows. A start no robot row matches yet is held until its stop, and sent
    /// once one does: the robot may connect on this machine, or be registered,
    /// after the recording started. Errs when the dispatcher is gone.
    async fn forward(&self, command: RecordingCommand) -> Result<(), StreamEnded> {
        match command {
            RecordingCommand::Open { .. } => {
                match with_rows(&command, &self.store, &self.org_id).await {
                    Ok(Some(open)) => self.send(open).await,
                    Ok(None) => {
                        self.hold(command).await;
                        Ok(())
                    }
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "failed to match a recording started from the web app"
                        );
                        Ok(())
                    }
                }
            }
            RecordingCommand::Close {
                ref recording_id, ..
            } => {
                // Under the lock, a held start of this recording is either sent
                // before this stop or never.
                let mut held = self.held.lock().await;
                held.remove(recording_id);
                let sent = self.send(command).await;
                drop(held);
                sent
            }
            RecordingCommand::Discard {
                ref cloud_recording_id,
                ..
            } => {
                self.held.lock().await.remove(cloud_recording_id);
                self.send(command).await
            }
            RecordingCommand::Snapshot {
                ref open_recording_ids,
                ..
            } => {
                // A held start whose stop came while the stream was down is
                // no longer running.
                self.held
                    .lock()
                    .await
                    .retain(|recording_id| open_recording_ids.contains(recording_id));
                self.send(command).await
            }
        }
    }

    async fn send(&self, command: RecordingCommand) -> Result<(), StreamEnded> {
        self.tx.send(command).await.map_err(|_| StreamEnded::Closed)
    }

    /// Hold `open` until a robot row matches it or its stop arrives.
    async fn hold(&self, open: RecordingCommand) {
        let RecordingCommand::Open { recording_id, .. } = &open else {
            return;
        };
        // The snapshot after a reconnect lists a held start again.
        if !self.held.lock().await.insert(recording_id.clone()) {
            return;
        }
        tracing::debug!(
            recording_id,
            "recording started from the web app waits for its robot to connect here"
        );
        let events = self.bus.subscribe();
        let matcher = self.clone();
        let recording_id = recording_id.clone();
        tokio::spawn(async move { matcher.wait_for_robot(open, recording_id, events).await });
    }

    /// Match a held start again each time a robot is registered, and send it
    /// once it matches. It is dropped once it is no longer held: its stop or
    /// discard arrived, or a reconnect's snapshot no longer lists it.
    async fn wait_for_robot(
        self,
        open: RecordingCommand,
        recording_id: String,
        mut events: broadcast::Receiver<DaemonEvent>,
    ) {
        loop {
            if !self.held.lock().await.contains(&recording_id) {
                return;
            }
            match with_rows(&open, &self.store, &self.org_id).await {
                Ok(Some(matched)) => {
                    let mut held = self.held.lock().await;
                    if held.remove(&recording_id) {
                        let _ = self.tx.send(matched).await;
                    }
                    return;
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(%error, recording_id, "failed to match a held start")
                }
            }
            if !robot_registered_or_timeout(&mut events).await {
                return;
            }
        }
    }
}

/// Wait until a robot row is registered, or [`HOLD_RECHECK`] passes. `false`
/// when the daemon is shutting down.
async fn robot_registered_or_timeout(events: &mut broadcast::Receiver<DaemonEvent>) -> bool {
    tokio::time::timeout(HOLD_RECHECK, async {
        loop {
            match events.recv().await {
                Ok(DaemonEvent::RobotResolved { .. })
                | Err(broadcast::error::RecvError::Lagged(_)) => return true,
                Ok(_) => {}
                Err(broadcast::error::RecvError::Closed) => return false,
            }
        }
    })
    .await
    .unwrap_or(true)
}

/// `open` with its robot and dataset rows, or `None` when no robot row holds
/// its robot's server id yet.
async fn with_rows(
    open: &RecordingCommand,
    store: &SqliteStateStore,
    org_id: &str,
) -> Result<Option<RecordingCommand>, StateStoreError> {
    let RecordingCommand::Open {
        recording_id,
        cloud_robot_id,
        robot_instance,
        cloud_dataset_id,
        start_timestamp_ns,
        start_timestamp_us,
        ..
    } = open
    else {
        return Ok(Some(open.clone()));
    };
    let Some(robot_index) = store
        .robot_index_for_cloud_id(org_id, cloud_robot_id, *robot_instance)
        .await?
    else {
        return Ok(None);
    };
    let dataset_index = match cloud_dataset_id {
        Some(cloud_dataset_id) => Some(
            store
                .dataset_index_for_cloud_id(org_id, cloud_dataset_id)
                .await?,
        ),
        None => None,
    };
    Ok(Some(RecordingCommand::Open {
        recording_id: recording_id.clone(),
        robot_index: Some(robot_index),
        cloud_robot_id: cloud_robot_id.clone(),
        robot_instance: *robot_instance,
        dataset_index,
        cloud_dataset_id: None,
        start_timestamp_ns: *start_timestamp_ns,
        start_timestamp_us: *start_timestamp_us,
    }))
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
    /// Only a STOP from a backend that sends it carries one.
    #[serde(default)]
    end_time: Option<f64>,
}

/// Turn one frame's JSON into commands, empty for anything this daemon does
/// not act on (an unparsable frame, or a lifecycle type it ignores).
fn parse_notification(data: &str) -> Vec<RecordingCommand> {
    let notification: Notification = match serde_json::from_str(data) {
        Ok(notification) => notification,
        Err(error) => {
            tracing::debug!(%error, "ignoring unparsable recording notification");
            return Vec::new();
        }
    };

    match notification.kind.as_str() {
        "START" => serde_json::from_value::<StartPayload>(notification.payload)
            .map(|payload| vec![open_command(payload)])
            .unwrap_or_default(),
        "INIT" => serde_json::from_value::<Vec<StartPayload>>(notification.payload)
            .map(|payloads| {
                let open_recording_ids = payloads
                    .iter()
                    .map(|payload| payload.recording_id.clone())
                    .collect();
                let mut commands: Vec<RecordingCommand> =
                    payloads.into_iter().map(open_command).collect();
                commands.push(RecordingCommand::Snapshot {
                    open_recording_ids,
                    observed_at_ns: wall_clock_ns(),
                });
                commands
            })
            .unwrap_or_default(),
        "STOP" | "EXPIRED" => serde_json::from_value::<StopPayload>(notification.payload)
            .map(|payload| {
                vec![RecordingCommand::Close {
                    recording_id: payload.recording_id,
                    end_timestamp_ns: payload.end_time.map(seconds_to_nanos),
                    observed_at_ns: wall_clock_ns(),
                }]
            })
            .unwrap_or_default(),
        "DISCARDED" => serde_json::from_value::<StopPayload>(notification.payload)
            .map(|payload| {
                vec![RecordingCommand::Discard {
                    cloud_recording_id: payload.recording_id,
                    observed_at_ns: wall_clock_ns(),
                }]
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The backend subscribes a stream before it reads the snapshot it sends as
/// INIT, so a START can arrive first and be missing from that snapshot.
fn keep_opened_in_snapshot(command: &mut RecordingCommand, opened: &mut HashSet<String>) {
    match command {
        RecordingCommand::Open { recording_id, .. } => {
            opened.insert(recording_id.clone());
        }
        RecordingCommand::Snapshot {
            open_recording_ids, ..
        } => open_recording_ids.extend(opened.iter().cloned()),
        _ => {}
    }
}

fn open_command(payload: StartPayload) -> RecordingCommand {
    RecordingCommand::Open {
        recording_id: payload.recording_id,
        robot_index: None,
        cloud_robot_id: payload.robot_id,
        robot_instance: payload.instance,
        dataset_index: None,
        cloud_dataset_id: payload.dataset_ids.into_iter().next(),
        start_timestamp_ns: seconds_to_nanos(payload.start_time),
        start_timestamp_us: seconds_to_microseconds(payload.start_time),
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
        let commands = parse_notification(
            r#"{"type":"START","payload":{"recording_id":"rec-1","robot_id":"robot-1",
               "instance":2,"created_by":"someone","dataset_ids":["ds-1"],
               "start_time":1.5}}"#,
        );
        assert_eq!(
            commands,
            vec![RecordingCommand::Open {
                recording_id: "rec-1".into(),
                robot_index: None,
                cloud_robot_id: "robot-1".into(),
                robot_instance: 2,
                dataset_index: None,
                cloud_dataset_id: Some("ds-1".into()),
                start_timestamp_ns: 1_500_000_000,
                start_timestamp_us: 1_500_000,
            }]
        );
    }

    fn split_init(mut commands: Vec<RecordingCommand>) -> (Vec<RecordingCommand>, Vec<String>) {
        let Some(RecordingCommand::Snapshot {
            open_recording_ids,
            observed_at_ns,
        }) = commands.pop()
        else {
            panic!("an INIT must end with its snapshot");
        };
        assert!(observed_at_ns > 0);
        let mut open: Vec<String> = open_recording_ids.into_iter().collect();
        open.sort();
        (commands, open)
    }

    #[test]
    fn start_carries_its_start_rounded_to_the_microsecond() {
        let commands = parse_notification(
            r#"{"type":"START","payload":{"recording_id":"rec-1","robot_id":"robot-1",
               "instance":0,"created_by":"someone","dataset_ids":[],
               "start_time":1771669371.383135}}"#,
        );
        match commands.as_slice() {
            [RecordingCommand::Open {
                start_timestamp_us, ..
            }] => assert_eq!(*start_timestamp_us, 1_771_669_371_383_135),
            other => panic!("START produced {other:?}"),
        }
    }

    #[test]
    fn init_opens_every_recording_already_running() {
        let (opens, snapshot) = split_init(parse_notification(
            r#"{"type":"INIT","payload":[
               {"recording_id":"rec-1","robot_id":"robot-1","instance":0,
                "created_by":"someone","dataset_ids":["ds-1"],"start_time":1.5},
               {"recording_id":"rec-2","robot_id":"robot-2","instance":1,
                "created_by":"someone","dataset_ids":[],"start_time":2.0}]}"#,
        ));
        assert_eq!(
            opens,
            vec![
                RecordingCommand::Open {
                    recording_id: "rec-1".into(),
                    robot_index: None,
                    cloud_robot_id: "robot-1".into(),
                    robot_instance: 0,
                    dataset_index: None,
                    cloud_dataset_id: Some("ds-1".into()),
                    start_timestamp_ns: 1_500_000_000,
                    start_timestamp_us: 1_500_000,
                },
                RecordingCommand::Open {
                    recording_id: "rec-2".into(),
                    robot_index: None,
                    cloud_robot_id: "robot-2".into(),
                    robot_instance: 1,
                    dataset_index: None,
                    cloud_dataset_id: None,
                    start_timestamp_ns: 2_000_000_000,
                    start_timestamp_us: 2_000_000,
                },
            ]
        );
        assert_eq!(snapshot, vec!["rec-1".to_string(), "rec-2".to_string()]);
    }

    #[test]
    fn an_empty_init_opens_nothing_and_says_nothing_is_open() {
        let (opens, snapshot) = split_init(parse_notification(r#"{"type":"INIT","payload":[]}"#));
        assert!(opens.is_empty());
        assert!(snapshot.is_empty());
    }

    #[test]
    fn a_start_received_before_init_stays_in_the_snapshot() {
        let mut opened = HashSet::new();
        let mut commands = parse_notification(
            r#"{"type":"START","payload":{"recording_id":"rec-new","robot_id":"robot-1",
               "instance":0,"created_by":"someone","dataset_ids":[],"start_time":3.0}}"#,
        );
        commands.extend(parse_notification(r#"{"type":"INIT","payload":[]}"#));
        for command in &mut commands {
            keep_opened_in_snapshot(command, &mut opened);
        }

        let (_, snapshot) = split_init(commands);
        assert_eq!(snapshot, vec!["rec-new".to_string()]);
    }

    #[test]
    fn an_unparsable_init_sends_no_snapshot() {
        assert!(parse_notification(r#"{"type":"INIT","payload":{"bad":1}}"#).is_empty());
    }

    #[test]
    fn stop_and_expired_become_a_close() {
        for kind in ["STOP", "EXPIRED"] {
            let commands = parse_notification(&format!(
                r#"{{"type":"{kind}","payload":{{"recording_id":"rec-1",
                   "robot_id":"robot-1","instance":0}}}}"#
            ));
            match commands.as_slice() {
                [RecordingCommand::Close {
                    recording_id,
                    end_timestamp_ns,
                    ..
                }] => {
                    assert_eq!(recording_id, "rec-1");
                    assert_eq!(*end_timestamp_ns, None);
                }
                other => panic!("{kind} produced {other:?}"),
            }
        }
    }

    #[test]
    fn a_stop_carries_its_end_time() {
        let commands = parse_notification(
            r#"{"type":"STOP","payload":{"recording_id":"rec-1",
               "robot_id":"robot-1","instance":0,"end_time":1.5}}"#,
        );
        match commands.as_slice() {
            [RecordingCommand::Close {
                end_timestamp_ns, ..
            }] => assert_eq!(*end_timestamp_ns, Some(1_500_000_000)),
            other => panic!("STOP produced {other:?}"),
        }
    }

    #[test]
    fn discarded_becomes_a_discard() {
        let commands = parse_notification(
            r#"{"type":"DISCARDED","payload":{"recording_id":"rec-1",
               "robot_id":"robot-1","instance":0}}"#,
        );
        match commands.as_slice() {
            [RecordingCommand::Discard {
                cloud_recording_id, ..
            }] => assert_eq!(cloud_recording_id, "rec-1"),
            other => panic!("DISCARDED produced {other:?}"),
        }
    }

    async fn open_store() -> (Arc<SqliteStateStore>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (Arc::new(store), dir)
    }

    /// Connect instance 2 of "arm" in org-1, as a process does.
    async fn connect_arm(store: &SqliteStateStore) -> i64 {
        store
            .connect_robot(&data_daemon_shared::RobotSpec {
                org_id: "org-1".into(),
                name: "arm".into(),
                shared: false,
                instance: 2,
                overwrite_urdf: false,
                package_path: None,
            })
            .await
            .unwrap()
            .robot_index
    }

    fn web_start(recording_id: &str) -> RecordingCommand {
        parse_notification(&format!(
            r#"{{"type":"START","payload":{{"recording_id":"{recording_id}",
               "robot_id":"cloud-robot-1","instance":2,"created_by":"someone",
               "dataset_ids":["cloud-dataset-1"],"start_time":1.5}}}}"#
        ))
        .remove(0)
    }

    fn matcher(store: &Arc<SqliteStateStore>) -> (Matcher, mpsc::Receiver<RecordingCommand>) {
        let (tx, rx) = mpsc::channel(8);
        let matcher = Matcher {
            store: Arc::clone(store),
            org_id: "org-1".into(),
            tx,
            held: Arc::default(),
            bus: EventBus::new(),
        };
        (matcher, rx)
    }

    /// The server registers robot row `robot_index`, as the robot resolver
    /// does.
    async fn register(matcher: &Matcher, robot_index: i64) {
        matcher
            .store
            .resolve_robot(robot_index, "cloud-robot-1", None)
            .await
            .unwrap();
        matcher
            .bus
            .publish(DaemonEvent::RobotResolved { robot_index });
    }

    async fn nothing_sent(rx: &mut mpsc::Receiver<RecordingCommand>) -> bool {
        tokio::time::timeout(Duration::from_millis(500), rx.recv())
            .await
            .is_err()
    }

    #[tokio::test]
    async fn a_web_start_is_matched_to_the_robot_row_holding_its_server_id() {
        let (store, _dir) = open_store().await;
        let robot_index = connect_arm(&store).await;
        store
            .resolve_robot(robot_index, "cloud-robot-1", None)
            .await
            .unwrap();
        let (matcher, mut rx) = matcher(&store);

        matcher.forward(web_start("rec-1")).await.unwrap();

        let Some(RecordingCommand::Open {
            robot_index: matched,
            dataset_index,
            ..
        }) = rx.recv().await
        else {
            panic!("expected an open");
        };
        assert_eq!(matched, Some(robot_index));
        let dataset = store
            .get_dataset(dataset_index.expect("a dataset row"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(dataset.cloud_dataset_id.as_deref(), Some("cloud-dataset-1"));
        assert_eq!(dataset.status, crate::state::ResolutionStatus::Done);
    }

    #[tokio::test]
    async fn a_web_start_waits_for_its_robot_to_be_registered() {
        let (store, _dir) = open_store().await;
        let robot_index = connect_arm(&store).await;
        let (matcher, mut rx) = matcher(&store);

        matcher.forward(web_start("rec-1")).await.unwrap();
        // The snapshot after a reconnect lists it again: still one start.
        matcher.forward(web_start("rec-1")).await.unwrap();
        assert!(nothing_sent(&mut rx).await);
        register(&matcher, robot_index).await;

        let sent = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("sent once the robot is registered")
            .unwrap();
        assert!(matches!(
            sent,
            RecordingCommand::Open { robot_index: Some(matched), .. } if matched == robot_index
        ));
        assert!(nothing_sent(&mut rx).await, "the start is sent once");
    }

    #[tokio::test]
    async fn a_held_web_start_whose_stop_comes_first_is_dropped() {
        let (store, _dir) = open_store().await;
        let robot_index = connect_arm(&store).await;
        let (matcher, mut rx) = matcher(&store);
        matcher.forward(web_start("rec-1")).await.unwrap();

        let stop =
            parse_notification(r#"{"type":"STOP","payload":{"recording_id":"rec-1"}}"#).remove(0);
        matcher.forward(stop).await.unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(RecordingCommand::Close { .. })
        ));
        register(&matcher, robot_index).await;

        assert!(
            nothing_sent(&mut rx).await,
            "nothing is opened for a recording that already ended"
        );
    }

    #[tokio::test]
    async fn a_web_start_before_its_robot_connects_is_sent_once_it_connects() {
        let (store, _dir) = open_store().await;
        let (matcher, mut rx) = matcher(&store);

        // No robot row on this machine yet: held, not dropped.
        matcher.forward(web_start("rec-1")).await.unwrap();
        assert!(nothing_sent(&mut rx).await);
        let robot_index = connect_arm(&store).await;
        register(&matcher, robot_index).await;

        let sent = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("sent once the robot connects and is registered")
            .unwrap();
        assert!(matches!(
            sent,
            RecordingCommand::Open { robot_index: Some(matched), .. } if matched == robot_index
        ));
    }

    #[tokio::test]
    async fn a_held_web_start_is_dropped_when_discarded_or_no_longer_running() {
        let (store, _dir) = open_store().await;
        let (matcher, mut rx) = matcher(&store);
        matcher.forward(web_start("rec-1")).await.unwrap();
        matcher.forward(web_start("rec-2")).await.unwrap();
        matcher.forward(web_start("rec-3")).await.unwrap();

        let discard =
            parse_notification(r#"{"type":"DISCARDED","payload":{"recording_id":"rec-1"}}"#)
                .remove(0);
        matcher.forward(discard).await.unwrap();
        // After a reconnect only rec-3 is still running.
        matcher
            .forward(RecordingCommand::Snapshot {
                open_recording_ids: HashSet::from(["rec-3".to_string()]),
                observed_at_ns: 0,
            })
            .await
            .unwrap();

        assert_eq!(
            *matcher.held.lock().await,
            HashSet::from(["rec-3".to_string()])
        );
        assert!(matches!(
            rx.recv().await,
            Some(RecordingCommand::Discard { .. })
        ));
        assert!(matches!(
            rx.recv().await,
            Some(RecordingCommand::Snapshot { .. })
        ));
    }

    #[test]
    fn ignores_unparsable_and_unhandled_frames() {
        assert!(parse_notification("not json").is_empty());
        assert!(parse_notification(r#"{"type":"SAVED","payload":{}}"#).is_empty());
    }
}
