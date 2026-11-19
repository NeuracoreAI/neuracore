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
use std::sync::Arc;
use std::time::Duration;

use data_daemon_shared::{
    short_name, Connection, DatasetSpec, Envelope, HealthReply, HealthRequest, ReasonCount,
    RecordingStateQuery, RecordingStateReply, RegistryReply, RegistryRequest, RobotSpec, RowState,
    RowStatus, StatusReply, StatusRequest, VersionReply, VersionRequest,
};
use iceoryx2::active_request::ActiveRequest;
use iceoryx2::port::server::Server;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::ipc;
use tokio::select;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{sleep, Instant};

use crate::api::auth::api_key_saved;
use crate::api::models::OrgSummary;
use crate::api::ApiClient;
use crate::cloud::read_org_id_from_config_async;
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

/// What the registry service needs besides the store.
pub struct RegistrySources {
    /// The folder robot packages are saved in
    /// ([`crate::storage::paths::robot_packages_root`]).
    pub packages_root: PathBuf,
    /// Where connecting a robot or selecting a dataset is announced, for the
    /// resolvers to wake on.
    pub bus: EventBus,
}

/// What the status query reads besides the state store.
pub struct StatusSources {
    /// The SDK config holding the API key and the current org.
    pub config_path: PathBuf,
    /// The org the daemon uploads to when the config names none, as the org
    /// watcher has it.
    pub fallback_org_id: Option<String>,
    /// Whether the daemon can reach the backend, from its latest probe.
    pub reachable: watch::Receiver<Connection>,
    /// The backend client, to name the organisations a reply is about.
    /// `None` in the offline profile or when it could not be built.
    pub client: Option<Arc<ApiClient>>,
}

/// How long a status query waits for the backend to name organisations. The
/// query is answered within a few hundred milliseconds either way, so a slow
/// answer leaves the ids rather than holding the reply.
const ORG_NAME_LOOKUP_TIMEOUT: Duration = Duration::from_millis(150);

/// A status query the daemon has received.
type StatusQuery = ActiveRequest<ipc::Service, [u8], (), [u8], ()>;

/// A status query whose reply is ready but for the organisation names, which
/// the backend is being asked for meanwhile. The listener keeps draining
/// recordings while it waits.
struct PendingStatus {
    query: StatusQuery,
    reply: StatusReply,
    lookup: JoinHandle<Option<Vec<OrgSummary>>>,
    deadline: Instant,
}

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
pub async fn run(
    transport: IpcTransport,
    dispatcher_tx: mpsc::Sender<Envelope>,
    recording_state: RecordingState,
    store: SqliteStateStore,
    registry: RegistrySources,
    status: StatusSources,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    tracing::info!(
        commands = data_daemon_shared::service_name::COMMANDS,
        recording_state = data_daemon_shared::service_name::RECORDING_STATE,
        registry = data_daemon_shared::service_name::REGISTRY,
        status = data_daemon_shared::service_name::STATUS,
        health = data_daemon_shared::service_name::HEALTH,
        version = data_daemon_shared::service_name::VERSION,
        "ipc listener started"
    );

    let mut counters = LoopCounters::default();
    let mut batch: Vec<Envelope> = Vec::with_capacity(64);
    // Consecutive empty drains, used to relax the poll cadence on an idle
    // daemon. Reset to 0 the moment any envelope arrives.
    let mut empty_drains: u32 = 0;
    // Status queries waiting for organisation names, answered on later passes.
    let mut pending_status: Vec<PendingStatus> = Vec::new();

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
            &registry.packages_root,
            &registry.bus,
        )
        .await;

        // -- Answer status queries -----------------------------------------------
        // What the recordings on this machine are waiting on, for the SDK's
        // one-off summary and `data-daemon status`. Never waits on the
        // network: a query that needs organisation names is answered on a
        // later pass, once they are in or its time is up.
        serve_status(
            transport.status_server(),
            &store,
            &status,
            &mut pending_status,
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
                .live(store, &query.source_key, query.robot_instance)
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
        let reply = answer_registry_request(store, packages_root, bus, request).await;
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
        RegistryRequest::DatasetForCloudId {
            org_id,
            cloud_dataset_id,
        } => {
            if org_id.is_empty() || cloud_dataset_id.is_empty() {
                return RegistryReply::Refused(
                    "the dataset has no organization or server id".into(),
                );
            }
            store
                .dataset_index_for_cloud_id(&org_id, &cloud_dataset_id)
                .await
                .map(RegistryReply::Index)
        }
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

/// Drain every pending status query, answering each from the state store,
/// the SDK config and the latest backend probe.
///
/// A query is rare (once per SDK process, and `data-daemon status`), so the
/// store and config are read per query rather than cached. A query the store
/// cannot answer gets no reply: the caller times out and shows no status,
/// which is better than one showing wrong counts.
async fn serve_status(
    server: &Server<ipc::Service, [u8], (), [u8], ()>,
    store: &SqliteStateStore,
    sources: &StatusSources,
    pending: &mut Vec<PendingStatus>,
) {
    // Answer the queries whose organisation names are in, or whose time is up.
    let now = Instant::now();
    let mut waiting = Vec::with_capacity(pending.len());
    for mut status in pending.drain(..) {
        match names_if_ready(&mut status.lookup, status.deadline, now).await {
            Some(orgs) => {
                if let Some(orgs) = orgs {
                    apply_org_names(&mut status.reply.blocked, &orgs);
                }
                send_status(status.query, &status.reply);
            }
            None => waiting.push(status),
        }
    }
    *pending = waiting;

    loop {
        let query = match server.receive() {
            Ok(Some(query)) => query,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, "status server receive failed");
                return;
            }
        };

        let request = match StatusRequest::decode(query.payload()) {
            Ok(request) => request,
            Err(error) => {
                tracing::warn!(%error, "dropping malformed status query");
                continue;
            }
        };

        let reply = match status_reply(request, store, sources).await {
            Ok(reply) => reply,
            Err(error) => {
                tracing::warn!(%error, "failed to count recordings for a status query");
                continue;
            }
        };

        match start_org_lookup(&reply.blocked, sources) {
            Some(lookup) => pending.push(PendingStatus {
                query,
                reply,
                lookup,
                deadline: Instant::now() + ORG_NAME_LOOKUP_TIMEOUT,
            }),
            None => send_status(query, &reply),
        }
    }
}

/// Send `reply` as the answer to `query`.
fn send_status(query: StatusQuery, reply: &StatusReply) {
    let bytes = match reply.encode() {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "failed to encode status reply");
            return;
        }
    };
    let response = match query.loan_slice_uninit(bytes.len()) {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(%error, "failed to loan status reply sample");
            return;
        }
    };
    if let Err(error) = response.write_from_slice(&bytes).send() {
        tracing::warn!(%error, "failed to send status reply");
    }
}

/// The answer to `request`.
async fn status_reply(
    request: StatusRequest,
    store: &SqliteStateStore,
    sources: &StatusSources,
) -> Result<StatusReply, StateStoreError> {
    let counts = store.status_counts().await?;
    let org_id = read_org_id_from_config_async(&sources.config_path).await;
    Ok(StatusReply {
        nonce: request.nonce,
        connection: *sources.reachable.borrow(),
        api_key_saved: api_key_saved(&sources.config_path).await,
        org_selected: org_id.is_some() || sources.fallback_org_id.is_some(),
        waiting: counts.waiting,
        uploading: counts.uploading,
        blocked: counts.blocked,
        failed: counts.failed,
    })
}

/// Ask the backend, on a task of its own, for the organisations the account
/// is a member of, when `reasons` name an organisation and the daemon can
/// reach the server. `None` when there is nothing to ask: the reply then
/// shows organisations by id. Nothing is stored; each status query asks.
fn start_org_lookup(
    reasons: &[ReasonCount],
    sources: &StatusSources,
) -> Option<JoinHandle<Option<Vec<OrgSummary>>>> {
    if !reasons.iter().any(|reason| reason.org_id.is_some()) {
        return None;
    }
    if *sources.reachable.borrow() != Connection::Reachable {
        return None;
    }
    let client = Arc::clone(sources.client.as_ref()?);
    Some(tokio::spawn(async move {
        match client.my_orgs().await {
            Ok(orgs) => Some(orgs),
            Err(error) => {
                tracing::debug!(%error, "could not look up organisation names");
                None
            }
        }
    }))
}

/// The organisations `lookup` found once it is done, or `Some(None)` once
/// `deadline` has passed without an answer (the lookup is then dropped).
/// `None` while it may still answer in time.
async fn names_if_ready(
    lookup: &mut JoinHandle<Option<Vec<OrgSummary>>>,
    deadline: Instant,
    now: Instant,
) -> Option<Option<Vec<OrgSummary>>> {
    if lookup.is_finished() {
        return Some(lookup.await.ok().flatten());
    }
    if now >= deadline {
        tracing::debug!("looking up organisation names timed out");
        lookup.abort();
        return Some(None);
    }
    None
}

/// Name each organisation `reasons` are about that is in `orgs`, the ones
/// the account is a member of. The server names no other, so the rest show
/// by id.
fn apply_org_names(reasons: &mut [ReasonCount], orgs: &[OrgSummary]) {
    for reason in reasons {
        if let Some(org_id) = &reason.org_id {
            reason.org_name = orgs
                .iter()
                .find(|org| &org.id == org_id)
                .map(|org| short_name(&org.name));
        }
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

    use crate::state::{NewRecording, RecordingTimestamp};

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
    async fn a_dataset_read_from_the_server_gets_a_confirmed_row() {
        let (store, _dir) = open_store().await;
        let for_cloud_id = |org_id: &str| RegistryRequest::DatasetForCloudId {
            org_id: org_id.into(),
            cloud_dataset_id: "cloud-dataset-1".into(),
        };
        let first = index(&store, for_cloud_id("org-1")).await;
        let again = index(&store, for_cloud_id("org-1")).await;

        assert_eq!(again, first);
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
                status: RowStatus::Done,
                cloud_id: Some(ref cloud_id),
                ..
            }) if cloud_id == "cloud-dataset-1"
        ));
        assert!(matches!(
            answer_registry_request(
                &store,
                Path::new(NO_PACKAGES),
                &EventBus::new(),
                for_cloud_id("")
            )
            .await,
            RegistryReply::Refused(_)
        ));
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

    async fn status_with(
        config: Option<&str>,
        fallback_org_id: Option<&str>,
        reachable: Connection,
    ) -> StatusReply {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                dataset_id: Some("ds-1"),
                start: RecordingTimestamp::observed_at(1),
                ..NewRecording::default()
            })
            .await
            .unwrap();
        let config_path = dir.path().join("config.json");
        if let Some(config) = config {
            tokio::fs::write(&config_path, config).await.unwrap();
        }
        let (_reachable_tx, reachable) = watch::channel(reachable);
        let sources = StatusSources {
            config_path,
            fallback_org_id: fallback_org_id.map(str::to_string),
            reachable,
            client: None,
        };

        status_reply(StatusRequest { nonce: 9 }, &store, &sources)
            .await
            .unwrap()
    }

    mod org_names {
        use super::*;
        use crate::api::auth::StaticAuthProvider;
        use crate::api::ApiClientOptions;
        use crate::state::{NewRecording, SqliteStateStore};
        use data_daemon_shared::{resolution_detail, RobotSpec, MAX_SUBJECT_NAME_BYTES};
        use tempfile::TempDir;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        fn sources(server: &MockServer, connection: Connection) -> StatusSources {
            let options = ApiClientOptions {
                base_url: server.uri(),
                timeout: Duration::from_secs(5),
                max_retries: 1,
                max_backoff: Duration::from_secs(1),
            };
            let client = ApiClient::new(options, Arc::new(StaticAuthProvider::new("token-1")));
            let (reachable_tx, reachable) = watch::channel(connection);
            Box::leak(Box::new(reachable_tx));
            StatusSources {
                config_path: PathBuf::from("/nonexistent/config.json"),
                fallback_org_id: None,
                reachable,
                client: Some(Arc::new(client.unwrap())),
            }
        }

        fn not_in(org_id: &str) -> ReasonCount {
            ReasonCount {
                reason: resolution_detail::NOT_IN_ORG.into(),
                recordings: 1,
                subject: None,
                org_id: Some(org_id.into()),
                org_name: None,
            }
        }

        fn org(id: &str, name: &str) -> OrgSummary {
            OrgSummary {
                id: id.into(),
                name: name.into(),
            }
        }

        #[test]
        fn names_the_orgs_the_account_is_in_and_leaves_the_rest_by_id() {
            let long_name = "x".repeat(MAX_SUBJECT_NAME_BYTES + 10);
            let mut reasons = vec![not_in("org-1"), not_in("org-2"), not_in("org-3")];

            apply_org_names(
                &mut reasons,
                &[org("org-1", "Acme Robotics"), org("org-3", &long_name)],
            );

            let names: Vec<Option<&str>> = reasons
                .iter()
                .map(|reason| reason.org_name.as_deref())
                .collect();
            assert_eq!(
                names,
                vec![
                    Some("Acme Robotics"),
                    None,
                    Some(&long_name[..MAX_SUBJECT_NAME_BYTES]),
                ]
            );
        }

        #[tokio::test]
        async fn asks_nothing_offline_or_when_no_reason_names_an_org() {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/org-management/my-orgs"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
                .expect(0)
                .mount(&server)
                .await;

            assert!(start_org_lookup(
                &[not_in("org-1")],
                &sources(&server, Connection::Unreachable)
            )
            .is_none());
            let mut no_org = not_in("org-1");
            no_org.org_id = None;
            assert!(
                start_org_lookup(&[no_org], &sources(&server, Connection::Reachable)).is_none()
            );
        }

        #[tokio::test]
        async fn a_lookup_still_running_waits_until_its_time_is_up() {
            let mut lookup: JoinHandle<Option<Vec<OrgSummary>>> = tokio::spawn(async {
                sleep(Duration::from_secs(60)).await;
                None
            });
            let now = Instant::now();

            assert_eq!(
                names_if_ready(&mut lookup, now + Duration::from_secs(1), now).await,
                None
            );
            assert_eq!(names_if_ready(&mut lookup, now, now).await, Some(None));

            let mut done: JoinHandle<Option<Vec<OrgSummary>>> =
                tokio::spawn(async { Some(vec![org("org-1", "Acme Robotics")]) });
            while !done.is_finished() {
                sleep(Duration::from_millis(1)).await;
            }
            assert_eq!(
                names_if_ready(&mut done, now + Duration::from_secs(1), now).await,
                Some(Some(vec![org("org-1", "Acme Robotics")]))
            );
        }

        #[tokio::test]
        async fn a_slow_backend_never_holds_up_the_listener() {
            // Each pass of the status server returns at once; the query is
            // answered by id once its time is up.
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/org-management/my-orgs"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!([
                            { "org_membership": {}, "org": { "id": "org-1", "name": "Acme" } },
                        ]))
                        .set_delay(Duration::from_secs(5)),
                )
                .mount(&server)
                .await;
            let dir = TempDir::new().unwrap();
            let store = SqliteStateStore::open(&dir.path().join("state.db"))
                .await
                .unwrap();
            let robot_index = store
                .connect_robot(&RobotSpec {
                    org_id: "org-1".into(),
                    name: "arm".into(),
                    shared: false,
                    instance: 0,
                    overwrite_urdf: false,
                    package_path: None,
                })
                .await
                .unwrap()
                .robot_index;
            store
                .create_recording(NewRecording {
                    robot_index: Some(robot_index),
                    robot_instance: Some(0),
                    dataset_id: Some("cloud-dataset-1"),
                    ..NewRecording::default()
                })
                .await
                .unwrap();
            store
                .hold_robot(
                    robot_index,
                    Some(crate::state::ResolutionReason::NotInOrg),
                    chrono::Utc::now().naive_utc() + chrono::Duration::minutes(5),
                )
                .await
                .unwrap();
            let sources = sources(&server, Connection::Reachable);
            let service = format!(
                "neuracore/data_daemon/test/status-slow-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let node = iceoryx2::node::NodeBuilder::new()
                .create::<ipc::Service>()
                .expect("iceoryx2 node");
            let (_factory, status_server) = crate::ipc::node::open_query_server(
                &node,
                &service,
                data_daemon_shared::service_name::STATUS_MAX_PAYLOAD_BYTES,
            )
            .expect("status server");

            let asking = std::thread::spawn(move || {
                crate::ipc::status_query::query_status_on(&service, Duration::from_secs(5))
            });
            let mut pending = Vec::new();
            let mut longest_pass = Duration::ZERO;
            while !asking.is_finished() {
                let started = std::time::Instant::now();
                serve_status(&status_server, &store, &sources, &mut pending).await;
                longest_pass = longest_pass.max(started.elapsed());
                sleep(Duration::from_millis(5)).await;
            }
            let reply = asking
                .join()
                .unwrap()
                .expect("status service reachable")
                .expect("an answer in time");

            assert!(
                longest_pass < Duration::from_millis(200),
                "{longest_pass:?}"
            );
            assert_eq!(reply.blocked.len(), 1);
            assert_eq!(reply.blocked[0].org_id.as_deref(), Some("org-1"));
            assert_eq!(reply.blocked[0].org_name, None);
        }
    }

    #[tokio::test]
    async fn status_reply_reports_the_config_the_probe_and_the_recordings() {
        let reply = status_with(
            Some(r#"{"api_key":"nrc_abc","current_org_id":"org-1"}"#),
            None,
            Connection::Reachable,
        )
        .await;

        assert_eq!(reply.nonce, 9);
        assert_eq!(reply.connection, Connection::Reachable);
        assert!(reply.api_key_saved);
        assert!(reply.org_selected);
        assert_eq!(reply.waiting.recordings, 1);
    }

    #[tokio::test]
    async fn status_reply_before_login_and_before_the_first_probe() {
        let reply = status_with(None, None, Connection::NotChecked).await;

        assert_eq!(reply.connection, Connection::NotChecked);
        assert!(!reply.api_key_saved);
        assert!(!reply.org_selected);
    }

    #[tokio::test]
    async fn the_status_command_gets_its_answer_over_ipc() {
        // The client `data-daemon status` uses must open the service exactly
        // as the daemon's server does. A service name of the test's own keeps
        // it clear of a daemon running on the same machine.
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        let (_reachable_tx, reachable) = watch::channel(Connection::Reachable);
        let sources = StatusSources {
            config_path: dir.path().join("config.json"),
            fallback_org_id: Some("org-1".into()),
            reachable,
            client: None,
        };
        let service = format!(
            "neuracore/data_daemon/test/status-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let node = iceoryx2::node::NodeBuilder::new()
            .create::<ipc::Service>()
            .expect("iceoryx2 node");
        let (_factory, server) = crate::ipc::node::open_query_server(
            &node,
            &service,
            data_daemon_shared::service_name::STATUS_MAX_PAYLOAD_BYTES,
        )
        .expect("status server");

        let asking = std::thread::spawn(move || {
            crate::ipc::status_query::query_status_on(&service, Duration::from_secs(5))
        });
        let mut pending = Vec::new();
        while !asking.is_finished() {
            serve_status(&server, &store, &sources, &mut pending).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let reply = asking
            .join()
            .unwrap()
            .expect("status service reachable")
            .expect("an answer in time");

        assert_eq!(reply.connection, Connection::Reachable);
        assert!(reply.org_selected);
        assert!(!reply.api_key_saved);
    }

    #[tokio::test]
    async fn status_reply_counts_the_fallback_org() {
        // As the org watcher does: NEURACORE_ORG_ID or the profile's org
        // stands in when the config names none.
        let reply = status_with(
            Some(r#"{"api_key":"nrc_abc"}"#),
            Some("org-1"),
            Connection::NotChecked,
        )
        .await;

        assert!(reply.org_selected);
    }
}
