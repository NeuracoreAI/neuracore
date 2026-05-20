//! iceoryx2 node and per-stream service bring-up.
//!
//! The daemon owns a single iceoryx2 [`Node`] for the duration of the process.
//! At startup it opens two long-lived subscribers: `commands` carries
//! lifecycle envelopes plus non-video `Frame`s, and `frames` carries the
//! pixel-bearing traffic of video traces. The split keeps the deep lifecycle
//! buffer away from multi-MiB payloads — see
//! [`data_daemon_ipc::service_name::FRAMES`] for the rationale.

use data_daemon_ipc::service_name::{
    COMMANDS, FRAMES, FRAMES_SUBSCRIBER_BUFFER_SIZE, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE,
    MAX_NODES_PER_SERVICE, MAX_PUBLISHERS_PER_SERVICE, MAX_SUBSCRIBERS_PER_SERVICE,
};
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::{ipc, NodeName};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
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
}

/// Daemon-side iceoryx2 transport.
///
/// Holds the node and the long-lived `commands` and `frames` subscribers. The
/// struct is `Send` (so it can move into the tokio main task) but not `Sync`
/// because iceoryx2's subscriber ports own shared-memory descriptors that must
/// be advanced from a single thread.
pub struct IpcTransport {
    /// Backing iceoryx2 node. Holding it alive keeps every service this
    /// daemon created visible to discovery.
    _node: Node<ipc::Service>,
    /// Subscriber on `neuracore/data_daemon/commands`.
    commands_subscriber: Subscriber<ipc::Service, [u8], ()>,
    /// Service handle held alongside the subscriber so port discovery doesn't
    /// race the service handle going out of scope.
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    /// Subscriber on `neuracore/data_daemon/frames` — video-trace traffic.
    frames_subscriber: Subscriber<ipc::Service, [u8], ()>,
    /// Service handle for the `frames` service, held for the same reason as
    /// `_commands_service`.
    _frames_service: PortFactory<ipc::Service, [u8], ()>,
}

impl IpcTransport {
    /// Bring up the daemon's iceoryx2 transport.
    ///
    /// Creates a node named after this daemon's PID
    /// (`neuracore-data-daemon-{pid}`), opens the `commands` and `frames`
    /// services, and builds a subscriber on each.
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
        let (frames_service, frames_subscriber) =
            open_subscriber(&node, FRAMES, FRAMES_SUBSCRIBER_BUFFER_SIZE)?;

        Ok(IpcTransport {
            _node: node,
            commands_subscriber,
            _commands_service: commands_service,
            frames_subscriber,
            _frames_service: frames_service,
        })
    }

    /// Borrow the `commands` subscriber port.
    pub fn commands_subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.commands_subscriber
    }

    /// Borrow the `frames` subscriber port.
    pub fn frames_subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.frames_subscriber
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
    // `subscriber_buffer_size` differs per service: the `commands` service
    // takes `LIFECYCLE_SUBSCRIBER_BUFFER_SIZE`, while `frames` takes the
    // small `FRAMES_SUBSCRIBER_BUFFER_SIZE` so the publisher data segment
    // that backs multi-MiB video frames stays bounded. The producer reuses
    // the same service config via `open_or_create`, so both sides must agree
    // on every attribute here.
    //
    // `enable_safe_overflow(false)` is load-bearing: iceoryx2 defaults a
    // service to safe-overflow *on*, where a full subscriber buffer silently
    // evicts the oldest sample — which, for the `commands` service, is
    // typically a `StartTrace`. With overflow disabled a full buffer instead
    // makes the producer's `UnableToDeliverStrategy::Block` take effect, so
    // delivery is lossless and in-order regardless of how shallow the buffer
    // is. Dropping a lifecycle envelope strands the per-trace actor; dropping
    // a video frame corrupts the trace — neither service can tolerate it.
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
