//! What the robot and dataset resolvers share: the task loop that decides
//! when to ask the server, sorting a server error into what it means for a
//! row, and when to ask again.
//!
//! A resolver runs a pass at startup, when a row is connected or selected, on
//! reconnect and after it missed events. While online it also runs when the
//! earliest wait set after a failure runs out; with nothing pending, or
//! offline, nothing else wakes it. Each resolver supplies only what differs
//! ([`RowResolver`]): which event asks for a pass, how to ask the server about
//! the rows due now, and when the next row is due.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{NaiveDateTime, Utc};
use reqwest::StatusCode;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{sleep_until, Instant};

use crate::api::auth::AuthError;
use crate::api::ApiClientError;
use crate::intervals::{BLOCKED_ID_RETRY, PASSING_ID_RETRY};
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{ConnectionState, DaemonEvent, EventBus, ResolutionReason, StateStoreError};

/// One kind of row a resolver has the server confirm.
#[async_trait]
pub(crate) trait RowResolver: Send + Sync + 'static {
    /// Short label used in the resolver's log lines.
    fn label(&self) -> &'static str;

    /// Whether `event` is a row being connected or selected, which asks for
    /// a pass.
    fn wakes_on(&self, event: &DaemonEvent) -> bool;

    /// Ask the server about every pending row that may be asked about now.
    async fn resolve_due(&self);

    /// The earliest time a pending row may be asked about again: `now` for
    /// one due already, `None` when no row is pending.
    async fn next_attempt_at(
        &self,
        now: NaiveDateTime,
    ) -> Result<Option<NaiveDateTime>, StateStoreError>;
}

/// Handle to a resolver task.
pub struct ResolverHandle {
    join: JoinHandle<()>,
    label: &'static str,
}

impl ResolverHandle {
    /// Wait for the task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, resolver = self.label, "resolver task join failed");
        }
    }
}

/// Spawn `resolver`'s task on the current Tokio runtime.
pub(crate) fn spawn_resolver<R: RowResolver>(
    resolver: R,
    bus: &EventBus,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> ResolverHandle {
    let label = resolver.label();
    let mut subscriber = bus.subscribe();
    let join = tokio::spawn(async move {
        // Assume online until the connection monitor says otherwise, as the
        // recording notifiers do: its first `Down` may predate this task.
        let mut connected = true;
        // The startup pass.
        let mut next_pass = pass(&resolver, connected).await;
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, resolver = label, "resolver shutting down");
                    break;
                }
                event = subscriber.recv() => match event {
                    Ok(DaemonEvent::ConnectionStateChanged(state)) => {
                        let reconnected = !connected && state == ConnectionState::Up;
                        connected = state == ConnectionState::Up;
                        if reconnected {
                            next_pass = pass(&resolver, connected).await;
                        } else if !connected {
                            // Nothing can be asked until the next `Up`.
                            next_pass = None;
                        }
                    }
                    Ok(event) if resolver.wakes_on(&event) => {
                        next_pass = pass(&resolver, connected).await;
                    }
                    // A missed event may be a row being connected or selected:
                    // a pass reads what is pending from the state store.
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        next_pass = pass(&resolver, connected).await;
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                () = sleep_until_due(next_pass) => next_pass = pass(&resolver, connected).await,
            }
        }
    });
    ResolverHandle { join, label }
}

/// Ask the server about every pending row that may be asked about now, when
/// `online`.
///
/// Returns when the next pass is due: once the earliest wait set after a
/// failure runs out, or `None` when offline or nothing is pending.
pub(crate) async fn pass<R: RowResolver>(resolver: &R, online: bool) -> Option<Instant> {
    if !online {
        return None;
    }
    resolver.resolve_due().await;
    let now = Utc::now().naive_utc();
    let wait = match resolver.next_attempt_at(now).await {
        Ok(next_attempt_at) => wait_until(next_attempt_at?, now),
        Err(error) => {
            tracing::warn!(
                %error,
                resolver = resolver.label(),
                "failed to find when to ask the server again"
            );
            PASSING_ID_RETRY
        }
    };
    Some(Instant::now() + wait)
}

/// Sleep until `due`, or for ever when no pass is due.
async fn sleep_until_due(due: Option<Instant>) {
    match due {
        Some(due) => sleep_until(due).await,
        None => std::future::pending().await,
    }
}

/// How long to sleep from `now` until `next_attempt_at`.
///
/// One already due right after a pass is left over from a write that failed
/// or a pass slower than its wait; it waits [`PASSING_ID_RETRY`], so a failing
/// state store cannot make the resolver spin. No sleep is longer than
/// [`BLOCKED_ID_RETRY`], so the resolver keeps waking if the clock jumps;
/// each wait is still measured on the clock, so one set back by an hour
/// delays the next attempt by an hour.
pub(crate) fn wait_until(next_attempt_at: NaiveDateTime, now: NaiveDateTime) -> Duration {
    match (next_attempt_at - now).to_std() {
        Ok(wait) if !wait.is_zero() => wait.min(BLOCKED_ID_RETRY),
        _ => PASSING_ID_RETRY,
    }
}

/// When the server may next be asked about a row held for `wait`.
pub(crate) fn retry_at(wait: Duration) -> NaiveDateTime {
    Utc::now().naive_utc() + wait
}

/// What a failed call means for the row it was about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Failure {
    /// Final: retrying the request cannot help.
    Refused(ResolutionReason),
    /// Stuck until the user fixes something; retried meanwhile.
    Blocked(ResolutionReason),
    /// Passing (an outage, a 5xx): retried soon.
    Retry,
}

/// Sort a server error into [`Failure`].
///
/// A 403 is final only when the server says the account may not create
/// shared datasets (`SHARED_UPLOAD_NOT_ALLOWED`); any other 403 is the
/// organization and role check, which logging in with an account that can
/// record there fixes. An archived robot (409 `ROBOT_ARCHIVED`) can be
/// unarchived. A rejected or missing API key can be fixed by logging in
/// again; any other failure to exchange the key is the server's and passes.
pub(crate) fn classify(error: &ApiClientError) -> Failure {
    if error.is_shared_upload_refused() {
        return Failure::Refused(ResolutionReason::SharedNotAllowed);
    }
    if error.is_robot_archived() {
        return Failure::Blocked(ResolutionReason::RobotArchived);
    }
    match error {
        ApiClientError::Status { status, .. } => match *status {
            StatusCode::UNAUTHORIZED => Failure::Blocked(ResolutionReason::Credentials),
            StatusCode::FORBIDDEN => Failure::Blocked(ResolutionReason::NotInOrg),
            StatusCode::BAD_REQUEST | StatusCode::CONFLICT | StatusCode::UNPROCESSABLE_ENTITY => {
                Failure::Refused(ResolutionReason::Rejected)
            }
            _ => Failure::Retry,
        },
        ApiClientError::Auth(AuthError::Missing { .. }) => {
            Failure::Blocked(ResolutionReason::Credentials)
        }
        ApiClientError::Auth(AuthError::ExchangeStatus { status, .. })
            if matches!(*status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) =>
        {
            Failure::Blocked(ResolutionReason::Credentials)
        }
        _ => Failure::Retry,
    }
}

/// What the user can do about a [`Failure::Blocked`] reason, for the log.
pub(crate) fn blocked_message(reason: ResolutionReason) -> &'static str {
    match reason {
        ResolutionReason::NotInOrg => {
            "the account cannot record in the organization the robot or dataset was made \
             in; log in with `neuracore login` as an account that can record there"
        }
        ResolutionReason::NameTaken => {
            "a dataset of the other kind, shared or private, already has this name; \
             rename or delete it on the Neuracore dashboard so the recordings can upload"
        }
        ResolutionReason::Credentials => {
            "the API key was rejected or is missing; run `neuracore login` so the \
             recordings can upload"
        }
        ResolutionReason::RobotArchived => {
            "the robot is archived; unarchive it on the Neuracore dashboard so the \
             recordings can upload"
        }
        ResolutionReason::SharedNotAllowed | ResolutionReason::Rejected => {
            "recordings are waiting; retrying later"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wait_is_until_the_next_attempt_within_bounds() {
        let now = Utc::now().naive_utc();
        let after = |seconds: i64| now + chrono::Duration::seconds(seconds);
        assert_eq!(wait_until(after(7), now), Duration::from_secs(7));
        // Due already: left over from a failed write, so not at once.
        assert_eq!(wait_until(now, now), PASSING_ID_RETRY);
        assert_eq!(wait_until(after(-60), now), PASSING_ID_RETRY);
        // Never longer than the longest wait the resolver sets.
        assert_eq!(wait_until(after(24 * 3600), now), BLOCKED_ID_RETRY);
    }

    #[test]
    fn errors_sort_into_refused_blocked_and_passing() {
        let status = |code: u16| ApiClientError::Status {
            status: StatusCode::from_u16(code).unwrap(),
            body: String::new(),
        };
        assert_eq!(
            classify(&status(401)),
            Failure::Blocked(ResolutionReason::Credentials)
        );
        // A 403 is the organization and role check, unless the server says
        // the account may not create shared datasets.
        assert_eq!(
            classify(&status(403)),
            Failure::Blocked(ResolutionReason::NotInOrg)
        );
        let shared_refused = ApiClientError::Status {
            status: StatusCode::FORBIDDEN,
            body: serde_json::json!({
                "detail": { "error_type": "SHARED_UPLOAD_NOT_ALLOWED" }
            })
            .to_string(),
        };
        assert_eq!(
            classify(&shared_refused),
            Failure::Refused(ResolutionReason::SharedNotAllowed)
        );
        let archived = ApiClientError::Status {
            status: StatusCode::CONFLICT,
            body: serde_json::json!({ "detail": { "error_type": "ROBOT_ARCHIVED" } }).to_string(),
        };
        assert_eq!(
            classify(&archived),
            Failure::Blocked(ResolutionReason::RobotArchived)
        );
        for code in [400, 409, 422] {
            assert_eq!(
                classify(&status(code)),
                Failure::Refused(ResolutionReason::Rejected)
            );
        }
        for code in [404, 429, 500, 503] {
            assert_eq!(classify(&status(code)), Failure::Retry);
        }
    }
}
