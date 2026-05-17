//! iceoryx2 node and per-stream service bring-up.
//!
//! The daemon owns a single iceoryx2 [`Node`] for the duration of the process.
//! At startup it opens two long-lived subscribers — `commands` for lifecycle
//! envelopes and `scalars` for low-rate sensor data — and exposes
//! [`IpcTransport::open_frame_subscriber`] so the listener can lazily open the
//! per-resolution `frames/<WxH>` services on first use. Holding everything off
//! a single [`Node`] keeps the dead-node sweep in
//! `lifecycle::recovery::cleanup_stale_ipc` able to reap every artefact this
//! daemon left behind in one pass.

use data_daemon_ipc::service_name::{self, COMMANDS, SCALARS};
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::{ipc, NodeName};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use thiserror::Error;

/// Errors raised while bringing up or extending the daemon's iceoryx2
/// transport.
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

/// A subscriber port on a per-resolution `frames/<WxH>` service.
///
/// Holds both the [`Subscriber`] and the [`PortFactory`] handle so the
/// service stays alive even after the listener drops its reference.
pub struct FrameSubscription {
    /// iceoryx2 service handle kept alive alongside the subscriber.
    _service: PortFactory<ipc::Service, [u8], ()>,
    /// Subscriber the listener polls.
    subscriber: Subscriber<ipc::Service, [u8], ()>,
}

impl FrameSubscription {
    /// Borrow the subscriber port for polling.
    pub fn subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.subscriber
    }
}

/// Daemon-side iceoryx2 transport.
///
/// Holds the node and the long-lived `commands` + `scalars` subscribers. The
/// struct is `Send` (so it can move into the tokio main task) but not `Sync`
/// because iceoryx2's subscriber ports own shared-memory descriptors that
/// must be advanced from a single thread.
pub struct IpcTransport {
    /// Backing iceoryx2 node. Holding it alive keeps every service this
    /// daemon created visible to discovery.
    node: Node<ipc::Service>,
    /// Subscriber on `neuracore/data_daemon/commands`.
    commands_subscriber: Subscriber<ipc::Service, [u8], ()>,
    /// Subscriber on `neuracore/data_daemon/scalars`.
    scalars_subscriber: Subscriber<ipc::Service, [u8], ()>,
    /// Service handles held alongside the subscribers so port discovery
    /// doesn't race the service handle going out of scope.
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    _scalars_service: PortFactory<ipc::Service, [u8], ()>,
}

impl IpcTransport {
    /// Bring up the daemon's iceoryx2 transport.
    ///
    /// Creates a node named after this daemon's PID
    /// (`neuracore-data-daemon-{pid}`), opens the `commands` and `scalars`
    /// services, and builds subscribers on both. Per-resolution
    /// `frames/<WxH>` services are opened lazily via
    /// [`open_frame_subscriber`](Self::open_frame_subscriber) when an
    /// [`OpenFrameStream`](data_daemon_ipc::Envelope::OpenFrameStream)
    /// envelope arrives.
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
        let (scalars_service, scalars_subscriber) = open_subscriber(&node, SCALARS)?;

        Ok(IpcTransport {
            node,
            commands_subscriber,
            scalars_subscriber,
            _commands_service: commands_service,
            _scalars_service: scalars_service,
        })
    }

    /// Borrow the `commands` subscriber port.
    pub fn commands_subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.commands_subscriber
    }

    /// Borrow the `scalars` subscriber port.
    pub fn scalars_subscriber(&self) -> &Subscriber<ipc::Service, [u8], ()> {
        &self.scalars_subscriber
    }

    /// Open (or attach to) the per-resolution `frames/<WxH>` service.
    ///
    /// Idempotent on the iceoryx2 side — `open_or_create()` reattaches if the
    /// producer side has already created the service. The returned
    /// [`FrameSubscription`] owns the subscriber and the service handle; the
    /// listener stores it in its local registry and polls it in the same
    /// drain loop as the long-lived subscribers.
    pub fn open_frame_subscriber(
        &self,
        width: u32,
        height: u32,
    ) -> Result<FrameSubscription, IpcSetupError> {
        // The per-resolution payload budget
        // (`service_name::frames_max_payload_bytes`) is enforced on the
        // *publisher* side via `initial_max_slice_len`; the subscriber
        // honours whatever the publisher negotiated, so there is nothing to
        // pass through here. Phase 4h wires that publisher up.
        let name = service_name::frames(width, height);
        let (service, subscriber) = open_subscriber(&self.node, &name)?;
        Ok(FrameSubscription {
            _service: service,
            subscriber,
        })
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
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
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
