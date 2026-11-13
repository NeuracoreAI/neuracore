//! The asking side of the status service, for `data-daemon status`.
//!
//! The running daemon serves [`service_name::STATUS`] (see
//! [`listener`](super::listener)); the `status` command is a separate process
//! of the same binary, so it asks over IPC like the SDK does. It only opens the
//! service, never creates it, so asking a daemon that has just exited leaves
//! nothing behind.

use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use data_daemon_shared::service_name::{
    MAX_NODES_PER_SERVICE, MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE,
    MAX_REQUEST_RESPONSE_SERVERS_PER_SERVICE, STATUS, STATUS_MAX_PAYLOAD_BYTES,
};
use data_daemon_shared::{StatusReply, StatusRequest};
use iceoryx2::node::NodeBuilder;
use iceoryx2::prelude::{ipc, NodeName};

/// How often to look for the reply.
const RECEIVE_POLL: Duration = Duration::from_millis(5);

/// Ask the running daemon what its recordings are waiting on.
///
/// `Ok(None)` means it did not answer within `timeout`. An `Err` means the
/// service could not be reached at all.
pub fn query_status(timeout: Duration) -> Result<Option<StatusReply>> {
    query_status_on(STATUS, timeout)
}

/// [`query_status`] on the service called `service`, so tests can serve one of
/// their own without meeting a daemon running on the same machine.
pub(crate) fn query_status_on(service: &str, timeout: Duration) -> Result<Option<StatusReply>> {
    let node_name = format!("neuracore-data-daemon-status-{}", std::process::id());
    let node = NodeBuilder::new()
        .name(&NodeName::new(&node_name).map_err(|error| anyhow!("{error}"))?)
        .create::<ipc::Service>()
        .map_err(|error| anyhow!("{error}"))
        .context("failed to create iceoryx2 node")?;
    let service_name = service
        .try_into()
        .map_err(|error| anyhow!("invalid service name {service}: {error}"))?;
    let port_factory = node
        .service_builder(&service_name)
        .request_response::<[u8], [u8]>()
        .max_clients(MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE)
        .max_servers(MAX_REQUEST_RESPONSE_SERVERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open()
        .map_err(|error| anyhow!("{error}"))
        .with_context(|| format!("failed to open {service}"))?;
    let client = port_factory
        .client_builder()
        .initial_max_slice_len(STATUS_MAX_PAYLOAD_BYTES)
        .create()
        .map_err(|error| anyhow!("{error}"))
        .context("failed to create the status client")?;

    let nonce = u64::from(std::process::id());
    let request = StatusRequest { nonce }.encode()?;
    let pending = client
        .loan_slice_uninit(request.len())
        .map_err(|error| anyhow!("{error}"))?
        .write_from_slice(&request)
        .send()
        .map_err(|error| anyhow!("{error}"))
        .context("failed to send the status query")?;

    let deadline = Instant::now() + timeout;
    loop {
        if let Some(response) = pending
            .receive()
            .map_err(|error| anyhow!("{error}"))
            .context("failed to receive the status reply")?
        {
            let reply = StatusReply::decode(response.payload())?;
            if reply.nonce == nonce {
                return Ok(Some(reply));
            }
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(RECEIVE_POLL);
    }
}
