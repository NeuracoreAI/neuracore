//! iceoryx2 node and per-stream service bring-up.
//!
//! The daemon owns a single iceoryx2 [`Node`] for the duration of the process.
//! At startup it opens one long-lived subscriber on the `commands` service:
//! it carries lifecycle envelopes, non-video `Frame`s, and the
//! [`Envelope::VideoChunkReady`] notifications the producer emits when a NUT
//! chunk lands on disk. There is no dedicated video bus — pixel buffers are
//! spooled to disk by the producer, so the IPC bus only ever carries
//! metadata-sized payloads.
//!
//! [`Envelope::VideoChunkReady`]: data_daemon_shared::Envelope::VideoChunkReady

use data_daemon_shared::service_name::{
    COMMANDS, HEALTH, HEALTH_MAX_PAYLOAD_BYTES, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE,
    MAX_NODES_PER_SERVICE, MAX_PUBLISHERS_PER_SERVICE, MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE,
    MAX_REQUEST_RESPONSE_SERVERS_PER_SERVICE, MAX_SUBSCRIBERS_PER_SERVICE, RECORDING_IDS,
    RECORDING_ID_MAX_PAYLOAD_BYTES,
};
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::server::Server;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::{ipc, NodeName};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use iceoryx2::service::port_factory::request_response::PortFactory as QueryPortFactory;
use thiserror::Error;

/// Errors raised while bringing up the daemon's iceoryx2 transport.
///
/// The inner cause fields are named `detail` rather than `source` so
/// `thiserror` doesn't try to wrap them in `dyn StdError`; the iceoryx2 error
/// types only implement `Display`, so we stringify at the boundary.
#[derive(Debug, Error)]
pub enum IpcSetupError {
    /// Constructing the iceoryx2 node failed.
    #[error("failed to create iceoryx2 node: {0}")]
    NodeCreate(String),
    /// The configured node name is not a valid iceoryx2 semantic string.
    #[error("invalid node name '{name}': {detail}")]
    InvalidNodeName {
        /// Offending name.
        name: String,
        /// Underlying iceoryx2 error message.
        detail: String,
    },
    /// The configured service name is not a valid iceoryx2 semantic string.
    #[error("invalid service name '{name}': {detail}")]
    InvalidServiceName {
        /// Offending name.
        name: String,
        /// Underlying iceoryx2 error message.
        detail: String,
    },
    /// Opening or creating an iceoryx2 service failed.
    #[error("failed to open service '{name}': {detail}")]
    ServiceOpen {
        /// Offending service.
        name: String,
        /// Underlying iceoryx2 error message.
        detail: String,
    },
    /// Building a subscriber port failed.
    #[error("failed to create subscriber on '{name}': {detail}")]
    SubscriberCreate {
        /// Owning service.
        name: String,
        /// Underlying iceoryx2 error message.
        detail: String,
    },
    /// Building a request-response server port failed.
    #[error("failed to create server on '{name}': {detail}")]
    ServerCreate {
        /// Owning service.
        name: String,
        /// Underlying iceoryx2 error message.
        detail: String,
    },
}

/// Daemon-side iceoryx2 transport.
///
/// Holds the node and the long-lived `commands` subscriber. The struct is
/// `Send` (so it can move into the tokio main task) but not `Sync` because
/// iceoryx2's subscriber ports own shared-memory descriptors that must be
/// advanced from a single thread.
pub struct IpcTransport {
    /// Backing iceoryx2 node. Holding it alive keeps every service this
    /// daemon created visible to discovery.
    _node: Node<ipc::Service>,
    /// Subscriber on `neuracore/data_daemon/commands`.
    commands_subscriber: Subscriber<ipc::Service, [u8], ()>,
    /// Service handle held alongside the subscriber so port discovery doesn't
    /// race the service handle going out of scope.
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    /// Request-response server on `neuracore/data_daemon/recording_ids` that answers
    /// SDK recording-id lookups.
    recording_id_server: Server<ipc::Service, [u8], (), [u8], ()>,
    /// Service handle held alongside the server, as for the commands service.
    _recording_id_service: QueryPortFactory<ipc::Service, [u8], (), [u8], ()>,
    /// Request-response server on `neuracore/data_daemon/health` that answers
    /// side-effect-free readiness probes.
    health_server: Server<ipc::Service, [u8], (), [u8], ()>,
    /// Service handle held alongside the health server.
    _health_service: QueryPortFactory<ipc::Service, [u8], (), [u8], ()>,
}

impl IpcTransport {
    /// Bring up the daemon's iceoryx2 transport.
    ///
    /// Creates a node named after this daemon's PID
    /// (`neuracore-data-daemon-{pid}`), opens the `commands` service, and
    /// builds a subscriber on it.
    pub fn bring_up() -> Result<Self, IpcSetupError> {
        let node_name = format!("neuracore-data-daemon-{}", std::process::id());
        let parsed_name =
            NodeName::new(&node_name).map_err(|error| IpcSetupError::InvalidNodeName {
                name: node_name.clone(),
                detail: error.to_string(),
            })?;
        let node = NodeBuilder::new()
            .name(&parsed_name)
            .create::<ipc::Service>()
            .map_err(|error| IpcSetupError::NodeCreate(error.to_string()))?;

        let (commands_service, commands_subscriber) =
            open_subscriber(&node, COMMANDS, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE)?;

        let (recording_id_service, recording_id_server) =
            open_query_server(&node, RECORDING_IDS, RECORDING_ID_MAX_PAYLOAD_BYTES)?;
        let (health_service, health_server) =
            open_query_server(&node, HEALTH, HEALTH_MAX_PAYLOAD_BYTES)?;

        Ok(IpcTransport {
            _node: node,
            commands_subscriber,
            _commands_service: commands_service,
            recording_id_server,
            _recording_id_service: recording_id_service,
            health_server,
            _health_service: health_service,
        })
    }

    /// Borrow the `commands` subscriber port.
    pub fn commands_subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.commands_subscriber
    }

    /// Borrow the `recording_ids` request-response server port.
    pub fn recording_id_server(&self) -> &Server<ipc::Service, [u8], (), [u8], ()> {
        &self.recording_id_server
    }

    /// Borrow the `health` request-response server port.
    pub fn health_server(&self) -> &Server<ipc::Service, [u8], (), [u8], ()> {
        &self.health_server
    }
}

/// Convenience alias for the `[u8]` pub/sub factory + subscriber pair
/// [`open_subscriber`] returns.
type ByteSliceFactory = PortFactory<ipc::Service, [u8], ()>;
type ByteSliceSubscriber = Subscriber<ipc::Service, [u8], ()>;

/// Open or attach to a `[u8]` pub/sub service and build a subscriber on it.
///
/// Centralised so the error annotations carry the offending service name in
/// one place. Subscriber slice lengths are negotiated from the publisher's
/// `initial_max_slice_len` — no per-service budget is applied here.
fn open_subscriber(
    node: &Node<ipc::Service>,
    name: &str,
    subscriber_buffer_size: usize,
) -> Result<(ByteSliceFactory, ByteSliceSubscriber), IpcSetupError> {
    let service_name = name
        .try_into()
        .map_err(|error| IpcSetupError::InvalidServiceName {
            name: name.to_string(),
            detail: format!("{error}"),
        })?;
    // `enable_safe_overflow(false)` is load-bearing: iceoryx2 defaults a
    // service to safe-overflow *on*, where a full subscriber buffer silently
    // evicts the oldest sample — which, for the `commands` service, is
    // typically a `StartTrace`. With overflow disabled a full buffer instead
    // makes the producer's `UnableToDeliverStrategy::Block` take effect, so
    // delivery is lossless and in-order regardless of how shallow the buffer
    // is. Dropping a lifecycle envelope strands the per-trace actor.
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
        .enable_safe_overflow(false)
        .subscriber_max_buffer_size(subscriber_buffer_size)
        .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
        .max_subscribers(MAX_SUBSCRIBERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| IpcSetupError::ServiceOpen {
            name: name.to_string(),
            detail: error.to_string(),
        })?;
    let subscriber =
        service
            .subscriber_builder()
            .create()
            .map_err(|error| IpcSetupError::SubscriberCreate {
                name: name.to_string(),
                detail: error.to_string(),
            })?;
    Ok((service, subscriber))
}

/// Convenience aliases for the `[u8]` request-response factory + server pair.
type ByteSliceQueryFactory = QueryPortFactory<ipc::Service, [u8], (), [u8], ()>;
type ByteSliceServer = Server<ipc::Service, [u8], (), [u8], ()>;

/// Open or attach to the `[u8]` request-response `queries` service and build the
/// daemon's single server on it.
///
/// The SDK opens client ports on the same service (one per OS thread, like the
/// `commands` publisher), so the caps mirror the publisher topology. Requests
/// and responses are both small postcard blobs ([`RECORDING_ID_MAX_PAYLOAD_BYTES`]).
fn open_query_server(
    node: &Node<ipc::Service>,
    name: &str,
    max_slice_len: usize,
) -> Result<(ByteSliceQueryFactory, ByteSliceServer), IpcSetupError> {
    let service_name = name
        .try_into()
        .map_err(|error| IpcSetupError::InvalidServiceName {
            name: name.to_string(),
            detail: format!("{error}"),
        })?;
    let service = node
        .service_builder(&service_name)
        .request_response::<[u8], [u8]>()
        .max_clients(MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE)
        .max_servers(MAX_REQUEST_RESPONSE_SERVERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| IpcSetupError::ServiceOpen {
            name: name.to_string(),
            detail: error.to_string(),
        })?;
    let server = service
        .server_builder()
        .initial_max_slice_len(max_slice_len)
        .create()
        .map_err(|error| IpcSetupError::ServerCreate {
            name: name.to_string(),
            detail: error.to_string(),
        })?;
    Ok((service, server))
}
