//! Tokio task that drains the iceoryx2 `commands` subscriber.
//!
//! iceoryx2 0.8 does not expose an `async`/`Notify`-style adaptor, so the
//! listener polls the subscriber on a short tick. The cadence is fast enough
//! to keep frame latency low and slow enough to leave idle daemons
//! effectively quiescent.
//!
//! ## Borrow lifetime
//!
//! iceoryx2's [`Subscriber`] is `Send` but `!Sync`. The loop drains the
//! subscriber synchronously into a local `Vec<Envelope>` before awaiting the
//! dispatcher: that bounds how long the subscriber borrow is held and lets
//! backpressure on the dispatcher send propagate to iceoryx2 without keeping
//! the subscriber locked across the await. (`Send` is not actually required
//! here — `run` is awaited inline under `block_on`, never `tokio::spawn`'d, and
//! the later `serve_recording_state_queries` borrow is itself held across an
//! await.)

use std::path::{Path, PathBuf};
use std::time::Duration;

use data_daemon_shared::{
    DatasetSpec, Envelope, HealthReply, HealthRequest, RecordingStateQuery, RecordingStateReply,
    RegistryReply, RegistryRequest, RobotSpec, RowState, RowStatus, VersionReply, VersionRequest,
    DAEMON_OFFLINE_REASON,
};
use iceoryx2::port::server::Server;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::ipc;
use tokio::select;
use tokio::sync::{broadcast, mpsc};
use tokio::time::sleep;

use crate::ipc::node::IpcTransport;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::pipeline::dispatcher::RecordingState;
use crate::state::{
    DaemonEvent, EventBus, ResolutionReason, ResolutionStatus, SqliteStateStore, StateStore,
    StateStoreError,
};
use crate::storage::paths::is_in_robot_packages;

/// Poll cadence while envelopes are actively flowing.
///
/// 200 µs bounds the worst-case producer-block time on a full subscriber
/// buffer. At the integration matrix's heaviest fanout (8 multiprocess
/// workers × ~4 producer threads each = ~32 publishers competing for
/// LIFECYCLE_SUBSCRIBER_BUFFER_SIZE=64 slots), 10 ms left producer-side
/// `log_*` calls blocked for ~1 s at a stretch on 2-vCPU hosts when the
/// listener task was preempted off-CPU by ffmpeg / per-trace work.
const POLL_INTERVAL: Duration = Duration::from_micros(200);

/// Poll cadence once the subscriber has been empty for [`IDLE_POLL_AFTER_EMPTY`]
/// consecutive drains. iceoryx2 0.8 has no async waker, so the listener must
/// poll — but a *fixed* 200 µs tick wakes 5000×/s on a daemon with no producer
/// attached. Decaying to 25 ms when idle keeps an idle daemon near-quiescent
/// while the first arriving sample snaps the cadence straight back to 200 µs, so
/// active-load latency is unchanged.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Number of consecutive empty drains before the poll cadence relaxes to
/// [`IDLE_POLL_INTERVAL`]. A handful of empty ticks at 200 µs is a negligible
/// cost and avoids relaxing during a brief lull mid-recording.
const IDLE_POLL_AFTER_EMPTY: u32 = 64;

/// Drain the iceoryx2 subscriber until a shutdown signal arrives.
///
/// Each successfully decoded envelope is forwarded to the dispatcher's
/// [`mpsc::Sender`]. If the dispatcher's queue fills (the dispatcher has
/// stalled), the listener blocks instead of dropping samples — backpressure
/// propagates back to iceoryx2's publisher queue and the producer SDK.
///
/// The listener takes ownership of the [`IpcTransport`]; when this task
/// returns the transport's destructor releases the iceoryx2 node back to the
/// OS.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    transport: IpcTransport,
    dispatcher_tx: mpsc::Sender<Envelope>,
    recording_state: RecordingState,
    store: SqliteStateStore,
    packages_root: PathBuf,
    bus: EventBus,
    offline: bool,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    tracing::info!(
        commands = data_daemon_shared::service_name::COMMANDS,
        recording_state = data_daemon_shared::service_name::RECORDING_STATE,
        registry = data_daemon_shared::service_name::REGISTRY,
        health = data_daemon_shared::service_name::HEALTH,
        version = data_daemon_shared::service_name::VERSION,
        "ipc listener started"
    );

    let mut counters = LoopCounters::default();
    let mut batch: Vec<Envelope> = Vec::with_capacity(64);
    // Consecutive empty drains, used to relax the poll cadence on an idle
    // daemon. Reset to 0 the moment any envelope arrives.
    let mut empty_drains: u32 = 0;

    loop {
        // -- Synchronous drain --------------------------------------------------
        // The subscriber borrow MUST stay inside this block (no `.await` in
        // any of these calls). The local `batch` is `Send`, so it can survive
        // across the awaits below without infecting the task with !Send.
        drain_subscriber(transport.commands_subscriber(), &mut batch, &mut counters);

        empty_drains = if batch.is_empty() {
            empty_drains.saturating_add(1)
        } else {
            0
        };

        // -- Async forward ------------------------------------------------------
        for envelope in batch.drain(..) {
            let kind = envelope.kind();
            if dispatcher_tx.send(envelope).await.is_err() {
                tracing::debug!(
                    envelope = kind,
                    "ipc listener stopping: dispatcher receiver dropped"
                );
                return;
            }
        }

        // -- Answer readiness probes --------------------------------------------
        // A health reply is the SDK launcher's readiness contract: reaching this
        // point proves the daemon has opened IPC, spawned the dispatcher, and
        // entered the listener loop that drains commands.
        serve_health(transport.health_server());

        // -- Answer version queries ---------------------------------------------
        // The SDK compares this answer with its own installed version, so a
        // daemon left behind by an earlier install is reported to the user
        // instead of being adopted silently.
        serve_version(transport.version_server());

        // -- Answer recording-state queries --------------------------------------
        // Every process on this host reads its recording state from here
        // rather than from its own subscription to the backend. The reply
        // carries the cloud id too, so this is the only state service.
        serve_recording_state_queries(transport.recording_state_server(), &recording_state, &store)
            .await;

        // -- Answer registry requests --------------------------------------------
        // Connecting a robot or selecting a dataset only writes its row; the
        // server is asked elsewhere, so no answer here waits on the network.
        serve_registry_requests(
            transport.registry_server(),
            &store,
            &packages_root,
            &bus,
            offline,
        )
        .await;

        // -- Yield / shutdown ---------------------------------------------------
        // Poll fast while data is flowing; relax once the bus has been empty
        // for a while so an idle daemon isn't woken 5000×/s.
        let poll_interval = if empty_drains >= IDLE_POLL_AFTER_EMPTY {
            IDLE_POLL_INTERVAL
        } else {
            POLL_INTERVAL
        };
        select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "ipc listener shutting down");
                return;
            }
            _ = sleep(poll_interval) => {}
        }
    }
}

/// Drain every pending health query, answering each with this daemon's PID.
fn serve_health(server: &Server<ipc::Service, [u8], (), [u8], ()>) {
    loop {
        let active = match server.receive() {
            Ok(Some(active)) => active,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, "health server receive failed");
                return;
            }
        };

        let request = match HealthRequest::decode(active.payload()) {
            Ok(request) => request,
            Err(error) => {
                tracing::warn!(%error, "dropping malformed health query");
                continue;
            }
        };

        let reply = HealthReply {
            pid: std::process::id(),
            nonce: request.nonce,
        };
        match reply.encode() {
            Ok(bytes) => match active.loan_slice_uninit(bytes.len()) {
                Ok(response) => {
                    let response = response.write_from_slice(&bytes);
                    if let Err(error) = response.send() {
                        tracing::warn!(%error, "failed to send health reply");
                    }
                }
                Err(error) => tracing::warn!(%error, "failed to loan health reply sample"),
            },
            Err(error) => tracing::warn!(%error, "failed to encode health reply"),
        }
    }
}

/// Drain every pending version query, answering each with the neuracore
/// version this daemon was built from.
///
/// The rust workspace version is held equal to the neuracore version in
/// `pyproject.toml`, so `CARGO_PKG_VERSION` is that version.
fn serve_version(server: &Server<ipc::Service, [u8], (), [u8], ()>) {
    loop {
        let active = match server.receive() {
            Ok(Some(active)) => active,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, "version server receive failed");
                return;
            }
        };

        let request = match VersionRequest::decode(active.payload()) {
            Ok(request) => request,
            Err(error) => {
                tracing::warn!(%error, "dropping malformed version query");
                continue;
            }
        };

        let reply = VersionReply {
            nonce: request.nonce,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        match reply.encode() {
            Ok(bytes) => match active.loan_slice_uninit(bytes.len()) {
                Ok(response) => {
                    let response = response.write_from_slice(&bytes);
                    if let Err(error) = response.send() {
                        tracing::warn!(%error, "failed to send version reply");
                    }
                }
                Err(error) => tracing::warn!(%error, "failed to loan version reply sample"),
            },
            Err(error) => tracing::warn!(%error, "failed to encode version reply"),
        }
    }
}

/// Drain every pending recording-state query, answering each from the
/// dispatcher's own state.
///
/// The dispatcher owns what a source is recording: it holds the recordings
/// local producers bracket *and* the ones the backend announced over the
/// notification stream, which it alone subscribes to. So a process that
/// started nothing — on a node that has logged nothing — still gets the true
/// answer, and it never depends on a row existing yet. Only the cloud id of a
/// recording started on this host is read from its row.
///
/// Bounded per tick rather than unbounded: each client keeps at most one
/// request in flight (it awaits the reply before sending the next), so a single
/// drain serves at most one request per connected client
/// (≤ `MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE`) before `receive` returns
/// `None` and the loop hands the next tick back to the commands drain.
async fn serve_recording_state_queries(
    server: &Server<ipc::Service, [u8], (), [u8], ()>,
    recording_state: &RecordingState,
    store: &SqliteStateStore,
) {
    loop {
        let active = match server.receive() {
            Ok(Some(active)) => active,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, "recording-state server receive failed");
                return;
            }
        };

        let query = match RecordingStateQuery::decode(active.payload()) {
            Ok(query) => query,
            Err(error) => {
                tracing::warn!(%error, "dropping malformed recording-state query");
                continue;
            }
        };

        let reply = RecordingStateReply {
            recording: recording_state
                .live(store, &query.robot_id, query.robot_instance)
                .await,
        };

        let bytes = match reply.encode() {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(%error, "failed to encode recording-state reply");
                continue;
            }
        };

        let response = match active.loan_slice_uninit(bytes.len()) {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(%error, "failed to loan recording-state reply sample");
                continue;
            }
        };

        if let Err(error) = response.write_from_slice(&bytes).send() {
            tracing::warn!(%error, "failed to send recording-state reply");
        }
    }
}

/// Drain every pending registry request, answering each from the state store.
///
/// Bounded per tick like [`serve_recording_state_queries`]: each client keeps
/// one request in flight.
async fn serve_registry_requests(
    server: &Server<ipc::Service, [u8], (), [u8], ()>,
    store: &SqliteStateStore,
    packages_root: &Path,
    bus: &EventBus,
    offline: bool,
) {
    loop {
        let active = match server.receive() {
            Ok(Some(active)) => active,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, "registry server receive failed");
                return;
            }
        };
        let request = match RegistryRequest::decode(active.payload()) {
            Ok(request) => request,
            Err(error) => {
                tracing::warn!(%error, "dropping malformed registry request");
                continue;
            }
        };
        let mut reply = answer_registry_request(store, packages_root, bus, request).await;
        if offline {
            reply = mark_daemon_offline(reply);
        }
        let bytes = match reply.encode() {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(%error, "failed to encode registry reply");
                continue;
            }
        };
        let response = match active.loan_slice_uninit(bytes.len()) {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(%error, "failed to loan registry reply sample");
                continue;
            }
        };
        if let Err(error) = response.write_from_slice(&bytes).send() {
            tracing::warn!(%error, "failed to send registry reply");
        }
    }
}

/// Answer one registry request from the state store, waking the resolver
/// for a row it marks to be confirmed.
async fn answer_registry_request(
    store: &SqliteStateStore,
    packages_root: &Path,
    bus: &EventBus,
    request: RegistryRequest,
) -> RegistryReply {
    let answered = match request {
        RegistryRequest::ConnectRobot(spec) => {
            if let Some(refusal) = refuse_robot(&spec) {
                return RegistryReply::Refused(refusal.into());
            }
            store.connect_robot(&spec).await.map(|connected| {
                if let Some(replaced) = connected
                    .replaced_package
                    .filter(|replaced| is_in_robot_packages(packages_root, Path::new(replaced)))
                {
                    // The new package supersedes it before it was sent.
                    if let Err(error) = std::fs::remove_file(&replaced) {
                        tracing::debug!(%error, path = replaced, "failed to delete a replaced package");
                    }
                }
                tracing::info!(
                    robot_index = connected.robot_index,
                    robot = spec.name,
                    instance = spec.instance,
                    "robot connected"
                );
                bus.publish(DaemonEvent::RobotConnected {
                    robot_index: connected.robot_index,
                });
                RegistryReply::Index(connected.robot_index)
            })
        }
        RegistryRequest::SelectDataset(spec) => {
            if let Some(refusal) = refuse_dataset(&spec) {
                return RegistryReply::Refused(refusal.into());
            }
            store.select_dataset(&spec).await.map(|dataset_index| {
                tracing::info!(dataset_index, dataset = spec.name, "dataset selected");
                bus.publish(DaemonEvent::DatasetSelected { dataset_index });
                RegistryReply::Index(dataset_index)
            })
        }
        RegistryRequest::RobotState { robot_index } => {
            store.get_robot(robot_index).await.map(|robot| match robot {
                Some(robot) => RegistryReply::State(row_state(
                    robot.status,
                    robot.cloud_robot_id,
                    robot.reason,
                )),
                None => RegistryReply::Unknown,
            })
        }
        RegistryRequest::DatasetState { dataset_index } => store
            .get_dataset(dataset_index)
            .await
            .map(|dataset| match dataset {
                Some(dataset) => RegistryReply::State(row_state(
                    dataset.status,
                    dataset.cloud_dataset_id,
                    dataset.reason,
                )),
                None => RegistryReply::Unknown,
            }),
    };
    answered.unwrap_or_else(|error: StateStoreError| {
        tracing::warn!(%error, "failed to answer a registry request");
        RegistryReply::Refused(format!(
            "the daemon could not read or write its state: {error}"
        ))
    })
}

/// Why a robot connect is refused, if it is.
fn refuse_robot(spec: &RobotSpec) -> Option<&'static str> {
    if spec.org_id.is_empty() {
        Some("no organization is selected")
    } else if spec.name.is_empty() {
        Some("the robot has no name")
    } else {
        None
    }
}

/// Why a dataset select is refused, if it is.
fn refuse_dataset(spec: &DatasetSpec) -> Option<&'static str> {
    if spec.org_id.is_empty() {
        Some("no organization is selected")
    } else if spec.name.is_empty() {
        Some("the dataset has no name")
    } else {
        None
    }
}

/// A reply from a daemon running offline: a pending row that nothing else
/// holds gets [`DAEMON_OFFLINE_REASON`], since this daemon never asks the
/// server. The SDK then knows not to wait for the server.
fn mark_daemon_offline(reply: RegistryReply) -> RegistryReply {
    match reply {
        RegistryReply::State(RowState {
            status: RowStatus::Pending,
            cloud_id,
            reason: None,
        }) => RegistryReply::State(RowState {
            status: RowStatus::Pending,
            cloud_id,
            reason: Some(DAEMON_OFFLINE_REASON.to_string()),
        }),
        other => other,
    }
}

/// A row's state as the registry service reports it.
fn row_state(
    status: ResolutionStatus,
    cloud_id: Option<String>,
    reason: Option<ResolutionReason>,
) -> RowState {
    RowState {
        status: match status {
            ResolutionStatus::Pending => RowStatus::Pending,
            ResolutionStatus::Done => RowStatus::Done,
            ResolutionStatus::Failed => RowStatus::Failed,
        },
        cloud_id,
        reason: reason.map(|reason| reason.as_str().to_string()),
    }
}

/// Counters reported in the slow-path warning logs.
#[derive(Default)]
struct LoopCounters {
    decode_failures: u64,
    receive_failures: u64,
}

/// Synchronously drain every available sample on `subscriber`, appending
/// decoded envelopes to `batch`.
///
/// Receive errors and decode failures are logged with a saturating counter
/// rather than returned — both are recoverable (a malformed sample doesn't
/// invalidate the next one) and the only path that escalates to a listener
/// exit is the dispatcher-receiver-closed branch, which is handled in
/// [`run`].
fn drain_subscriber(
    subscriber: &Subscriber<ipc::Service, [u8], ()>,
    batch: &mut Vec<Envelope>,
    counters: &mut LoopCounters,
) {
    loop {
        match subscriber.receive() {
            Ok(Some(sample)) => match Envelope::decode(sample.payload()) {
                // Every envelope is forwarded whole. `BatchedData` is held and
                // released as a single unit by the dispatcher (all its items
                // share one timestamp, so they belong to one window) and
                // expanded into per-sensor routes only at release time.
                Ok(envelope) => batch.push(envelope),
                Err(error) => {
                    counters.decode_failures = counters.decode_failures.saturating_add(1);
                    tracing::warn!(
                        %error,
                        decode_failures = counters.decode_failures,
                        "ipc envelope decode failed; dropping sample"
                    );
                }
            },
            Ok(None) => return,
            Err(error) => {
                counters.receive_failures = counters.receive_failures.saturating_add(1);
                tracing::warn!(
                    error = %error,
                    receive_failures = counters.receive_failures,
                    "ipc subscriber receive failed"
                );
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (store, dir)
    }

    fn arm(package_path: Option<String>) -> RobotSpec {
        RobotSpec {
            org_id: "org-1".into(),
            name: "arm".into(),
            shared: false,
            instance: 7,
            overwrite_urdf: false,
            package_path,
        }
    }

    /// A package folder for the tests that save no package.
    const NO_PACKAGES: &str = "/nonexistent/.robot_packages";

    #[test]
    fn an_offline_daemon_marks_only_pending_rows_nothing_else_holds() {
        let state = |status, reason: Option<&str>| {
            RegistryReply::State(RowState {
                status,
                cloud_id: None,
                reason: reason.map(str::to_string),
            })
        };
        assert_eq!(
            mark_daemon_offline(state(RowStatus::Pending, None)),
            state(RowStatus::Pending, Some(DAEMON_OFFLINE_REASON))
        );
        // A reason the row already has says more than the daemon being offline.
        for unchanged in [
            state(RowStatus::Pending, Some("robot_archived")),
            state(RowStatus::Done, None),
            state(RowStatus::Failed, Some("rejected")),
            RegistryReply::Index(3),
            RegistryReply::Unknown,
        ] {
            assert_eq!(mark_daemon_offline(unchanged.clone()), unchanged);
        }
    }

    fn warehouse(name: &str) -> DatasetSpec {
        DatasetSpec {
            org_id: "org-1".into(),
            name: name.into(),
            shared: false,
            description: None,
            tags: Vec::new(),
        }
    }

    async fn index(store: &SqliteStateStore, request: RegistryRequest) -> i64 {
        match answer_registry_request(store, Path::new(NO_PACKAGES), &EventBus::new(), request)
            .await
        {
            RegistryReply::Index(index) => index,
            other => panic!("expected an index, got {other:?}"),
        }
    }

    async fn connect_with(store: &SqliteStateStore, packages: &Path, package: &Path) -> i64 {
        let request = RegistryRequest::ConnectRobot(arm(Some(package.display().to_string())));
        match answer_registry_request(store, packages, &EventBus::new(), request).await {
            RegistryReply::Index(index) => index,
            other => panic!("expected an index, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_replaced_package_outside_the_package_folder_is_never_deleted() {
        let (store, dir) = open_store().await;
        let packages = dir.path().join(".robot_packages");
        let outside = dir.path().join("arm.urdf");
        std::fs::write(&outside, b"urdf").unwrap();
        connect_with(&store, &packages, &outside).await;
        let climbing = packages.join("..").join("arm.urdf");
        connect_with(&store, &packages, &climbing).await;
        connect_with(&store, &packages, &packages.join("new.zip")).await;
        assert!(outside.exists(), "a file outside the package folder stays");
    }

    #[tokio::test]
    async fn every_process_connecting_a_robot_instance_gets_its_one_row() {
        let (store, dir) = open_store().await;
        let packages = dir.path().join(".robot_packages");
        std::fs::create_dir_all(&packages).unwrap();
        let first_package = packages.join("first.zip");
        std::fs::write(&first_package, b"zip").unwrap();
        let first = connect_with(&store, &packages, &first_package).await;

        // Another process, with a newer package for the same instance.
        let second_package = packages.join("second.zip");
        std::fs::write(&second_package, b"zip").unwrap();
        let again = connect_with(&store, &packages, &second_package).await;
        let other_instance = index(
            &store,
            RegistryRequest::ConnectRobot(RobotSpec {
                instance: 8,
                ..arm(None)
            }),
        )
        .await;

        assert_eq!(again, first);
        assert_ne!(other_instance, first);
        assert!(!first_package.exists(), "the replaced package is deleted");
        let robot = store.get_robot(first).await.unwrap().unwrap();
        assert_eq!(
            robot.package_path.as_deref(),
            Some(second_package.display().to_string().as_str())
        );
        assert_eq!(
            answer_registry_request(
                &store,
                Path::new(NO_PACKAGES),
                &EventBus::new(),
                RegistryRequest::RobotState { robot_index: first }
            )
            .await,
            RegistryReply::State(RowState {
                status: RowStatus::Pending,
                cloud_id: None,
                reason: None,
            })
        );
    }

    #[tokio::test]
    async fn a_dataset_selected_twice_has_one_row() {
        let (store, _dir) = open_store().await;
        let first = index(
            &store,
            RegistryRequest::SelectDataset(warehouse("warehouse")),
        )
        .await;
        let again = index(
            &store,
            RegistryRequest::SelectDataset(warehouse("warehouse")),
        )
        .await;
        let other = index(&store, RegistryRequest::SelectDataset(warehouse("yard"))).await;

        assert_eq!(again, first);
        assert_ne!(other, first);
        assert!(matches!(
            answer_registry_request(
                &store,
                Path::new(NO_PACKAGES),
                &EventBus::new(),
                RegistryRequest::DatasetState {
                    dataset_index: first
                }
            )
            .await,
            RegistryReply::State(RowState {
                status: RowStatus::Pending,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn requests_without_an_organization_are_refused() {
        let (store, _dir) = open_store().await;
        let robot = answer_registry_request(
            &store,
            Path::new(NO_PACKAGES),
            &EventBus::new(),
            RegistryRequest::ConnectRobot(RobotSpec {
                org_id: String::new(),
                ..arm(None)
            }),
        )
        .await;
        let dataset = answer_registry_request(
            &store,
            Path::new(NO_PACKAGES),
            &EventBus::new(),
            RegistryRequest::SelectDataset(DatasetSpec {
                org_id: String::new(),
                ..warehouse("warehouse")
            }),
        )
        .await;

        assert!(matches!(robot, RegistryReply::Refused(_)), "{robot:?}");
        assert!(matches!(dataset, RegistryReply::Refused(_)), "{dataset:?}");
    }

    #[tokio::test]
    async fn an_unknown_index_is_reported_as_unknown() {
        let (store, _dir) = open_store().await;
        assert_eq!(
            answer_registry_request(
                &store,
                Path::new(NO_PACKAGES),
                &EventBus::new(),
                RegistryRequest::RobotState { robot_index: 42 }
            )
            .await,
            RegistryReply::Unknown
        );
        assert_eq!(
            answer_registry_request(
                &store,
                Path::new(NO_PACKAGES),
                &EventBus::new(),
                RegistryRequest::DatasetState { dataset_index: 42 }
            )
            .await,
            RegistryReply::Unknown
        );
    }
}
