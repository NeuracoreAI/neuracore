//! iceoryx2 node and per-stream service bring-up.
//!
//! The daemon owns a single iceoryx2 [`Node`] for the duration of the process.
//! At startup it opens the long-lived `commands` subscriber, which in phase 4
//! carries every envelope variant (lifecycle commands and frame payloads).
//! Sub-phase 4h splits the high-throughput frame traffic onto dedicated
//! per-resolution services; until then there is only one subscriber to drain.

use data_daemon_ipc::service_name::{
    COMMANDS, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE, MAX_PUBLISHERS_PER_SERVICE,
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

        let (commands_service, commands_subscriber) = open_subscriber(&node, COMMANDS)?;

        Ok(IpcTransport {
            _node: node,
            commands_subscriber,
            _commands_service: commands_service,
        })
    }

    /// Borrow the `commands` subscriber port.
    pub fn commands_subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.commands_subscriber
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
) -> Result<(ByteSliceFactory, ByteSliceSubscriber), IpcSetupError> {
    let service_name = name
        .try_into()
        .map_err(|error| IpcSetupError::InvalidServiceName {
            name: name.to_string(),
            detail: format!("{error}"),
        })?;
    // `subscriber_max_buffer_size` is sized for the worst-case burst of
    // lifecycle envelopes a single SDK call can publish (see
    // `LIFECYCLE_SUBSCRIBER_BUFFER_SIZE` for the failure mode at the
    // iceoryx2 default of 2). The publisher side reuses the same service
    // config via `open_or_create` so producers honour this without having to
    // declare it explicitly themselves.
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
        .subscriber_max_buffer_size(LIFECYCLE_SUBSCRIBER_BUFFER_SIZE)
        .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
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
