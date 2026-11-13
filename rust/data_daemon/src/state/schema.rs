//! Strongly-typed rows and lifecycle enums for the daemon's SQLite tables.
//!
//! The enum string values are part of the integration-test contract: the
//! constants in `tests/integration/platform/data_daemon/shared/db_constants.py`
//! pin the string columns, so the stored spellings must stay stable.

use std::str::FromStr;

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use thiserror::Error;

macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $value:literal,
            )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            /// Wire-format string used in the SQLite column.
            pub fn as_str(self) -> &'static str {
                match self {
                    $(
                        Self::$variant => $value,
                    )+
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = ParseEnumError;
            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match value {
                    $(
                        $value => Ok(Self::$variant),
                    )+
                    other => Err(ParseEnumError {
                        kind: stringify!($name),
                        value: other.to_string(),
                    }),
                }
            }
        }
    };
}

/// Returned when a status string read from SQLite does not match any known
/// variant. Surfaced as a `StateStoreError::Decode` by the store
/// implementation.
#[derive(Debug, Clone, Error)]
#[error("invalid {kind} value: '{value}'")]
pub struct ParseEnumError {
    /// Enum type name that failed to parse.
    pub kind: &'static str,
    /// Offending column value.
    pub value: String,
}

string_enum! {
    /// Write/persistence lifecycle for a trace.
    ///
    /// Matches `TraceWriteStatus` in `neuracore/data_daemon/models.py`.
    pub enum TraceWriteStatus {
        Pending         => "pending",
        Initializing    => "initializing",
        Writing         => "writing",
        PendingMetadata => "pending_metadata",
        Written         => "written",
        Failed          => "failed",
    }
}

string_enum! {
    /// Backend registration lifecycle for a trace.
    pub enum TraceRegistrationStatus {
        Pending     => "pending",
        Registering => "registering",
        Registered  => "registered",
        Retrying    => "retrying",
        Failed      => "failed",
    }
}

string_enum! {
    /// Upload lifecycle for a trace.
    pub enum TraceUploadStatus {
        Pending   => "pending",
        Queued    => "queued",
        Uploading => "uploading",
        Paused    => "paused",
        Uploaded  => "uploaded",
        Retrying  => "retrying",
        Failed    => "failed",
    }
}

string_enum! {
    /// Standardised error codes for trace failures.
    pub enum TraceErrorCode {
        Unknown             => "unknown",
        WriteFailed         => "write_failed",
        EncodeFailed        => "encode_failed",
        UploadFailed        => "upload_failed",
        DiskFull            => "disk_full",
        NetworkError        => "network_error",
        ProgressReportError => "progress_report_error",
        RecordingCancelled  => "recording_cancelled",
        RecordingNotPending => "recording_not_pending",
    }
}

string_enum! {
    /// Status of progress report for a recording.
    pub enum ProgressReportStatus {
        Pending   => "pending",
        Reporting => "reporting",
        Reported  => "reported",
    }
}

/// Why a robot or dataset made offline is stuck `pending`, or why it
/// `failed`, as stored in `local_robots.detail` and `local_datasets.detail`.
/// The codes are shared with the SDK, which shows them in the daemon's status.
pub use data_daemon_shared::resolution_detail;

string_enum! {
    /// Where a recording started offline is in getting the backend's ids for
    /// its robot and dataset (`recordings.offline`). A recording started
    /// online has none.
    pub enum OfflineState {
        /// Waiting for its robot and/or dataset.
        Resolving => "resolving",
        /// Has every backend id; the normal pipeline takes it from here.
        Resolved  => "resolved",
        /// A robot or dataset was refused for good; it never uploads.
        Failed    => "failed",
    }
}

string_enum! {
    /// Where a robot or dataset made offline is in being found or created on
    /// the backend (`local_robots.status`, `local_datasets.status`).
    pub enum LocalIdStatus {
        /// The backend's id is not known yet.
        Pending => "pending",
        /// The backend's id is known.
        Done    => "done",
        /// The backend refused to find or create it.
        Failed  => "failed",
    }
}

/// A row from the `recordings` table.
///
/// The daemon owns recording identity: `recording_index` is the local primary
/// key (AUTOINCREMENT), allocated when the `StartRecording` envelope is first
/// seen; `recording_id` is the cloud handle, filled asynchronously by the
/// recording-start notifier once `/recording/start` lands. The two are
/// independent — never aliased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingRow {
    /// Local primary key (AUTOINCREMENT). The daemon keys every internal
    /// structure and the `traces` foreign key on this.
    pub recording_index: i64,
    /// Cloud handle. `None` until `/recording/start` is notified. Every cloud
    /// URL reads this straight from the row; downstream coordinators wait for
    /// it, so an offline recording stays pending until the daemon is online.
    pub recording_id: Option<String>,
    /// Robot identifier — first half of the source key.
    pub robot_id: Option<String>,
    /// Robot instance — second half of the source key.
    pub robot_instance: Option<i64>,
    /// Dataset identifier, when supplied.
    pub dataset_id: Option<String>,
    /// Caller's start of the recording in microseconds on its data clock.
    pub start_timestamp_us: Option<i64>,
    /// Wall-clock time of the start (Unix nanoseconds): the producer's publish time, or the daemon's clock when the daemon
    /// observed or received it.
    pub start_publish_timestamp_ns: Option<i64>,
    /// Caller's stop or cancel of the recording in microseconds on its data
    /// clock.
    pub stop_timestamp_us: Option<i64>,
    /// Wall-clock time of the stop or cancel (Unix nanoseconds): the producer's publish time, or the daemon's clock when the daemon
    /// observed or received it.
    pub stop_publish_timestamp_ns: Option<i64>,
    /// Expected number of traces, set when the producer declares it.
    pub expected_trace_count: Option<i64>,
    /// `1` once the expected trace count has been reported to the backend.
    pub expected_trace_count_reported: i64,
    /// Progress-report lifecycle for this recording.
    pub progress_reported: ProgressReportStatus,
    /// Set when the producer issues a stop command.
    pub stopped_at: Option<NaiveDateTime>,
    /// Set when the producer issues a cancel command. Cancelled recordings
    /// are ignored by the cloud coordinators and skipped by the progress
    /// reporter.
    pub cancelled_at: Option<NaiveDateTime>,
    /// Set when the recording-start notifier successfully POSTed
    /// `/recording/start` and persisted the cloud `recording_id`.
    pub backend_start_notified_at: Option<NaiveDateTime>,
    /// Set when the recording-stop notifier successfully POSTed
    /// `/recording/stop` to the backend. `None` means the backend has not
    /// yet been notified — typically a recording stopped while the daemon
    /// was offline; the notifier sweeps these on startup.
    pub backend_stop_notified_at: Option<NaiveDateTime>,
    /// Set when the recording-cancel notifier successfully POSTed
    /// `/recording/cancel` to the backend. `None` means either the recording
    /// was never cancelled, or cancellation has not yet been notified.
    pub backend_cancel_notified_at: Option<NaiveDateTime>,
    /// Set when the backend answered 404 for the recording's uploads or
    /// progress. Its remaining uploads are failed and its files kept until
    /// the reaper deletes recordings that can never upload.
    pub backend_not_found_at: Option<NaiveDateTime>,
    /// Where a recording started offline is in getting its backend ids;
    /// `None` for a recording started online.
    pub offline: Option<OfflineState>,
    /// When the reaper first saw that the recording can never upload. It
    /// deletes the recording a fixed time later.
    pub failed_at: Option<NaiveDateTime>,
    /// First-seen timestamp.
    pub created_at: NaiveDateTime,
    /// Last write timestamp; bumped on every row mutation.
    pub last_updated: NaiveDateTime,
}

impl RecordingRow {
    /// Decode a SQLite row into a [`RecordingRow`].
    pub(crate) fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        let progress_reported = parse_column::<ProgressReportStatus>(row, "progress_reported")?;
        let offline = row
            .try_get::<Option<String>, _>("offline")?
            .map(|raw| OfflineState::from_str(&raw).map_err(|error| decode_error("offline", error)))
            .transpose()?;
        Ok(RecordingRow {
            recording_index: row.try_get("recording_index")?,
            recording_id: row.try_get("recording_id")?,
            robot_id: row.try_get("robot_id")?,
            robot_instance: row.try_get("robot_instance")?,
            dataset_id: row.try_get("dataset_id")?,
            start_timestamp_us: row.try_get("start_timestamp_us")?,
            start_publish_timestamp_ns: row.try_get("start_publish_timestamp_ns")?,
            stop_timestamp_us: row.try_get("stop_timestamp_us")?,
            stop_publish_timestamp_ns: row.try_get("stop_publish_timestamp_ns")?,
            expected_trace_count: row.try_get("expected_trace_count")?,
            expected_trace_count_reported: row.try_get("expected_trace_count_reported")?,
            progress_reported,
            stopped_at: row.try_get("stopped_at")?,
            cancelled_at: row.try_get("cancelled_at")?,
            backend_start_notified_at: row.try_get("backend_start_notified_at")?,
            backend_stop_notified_at: row.try_get("backend_stop_notified_at")?,
            backend_cancel_notified_at: row.try_get("backend_cancel_notified_at")?,
            backend_not_found_at: row.try_get("backend_not_found_at")?,
            offline,
            failed_at: row.try_get("failed_at")?,
            created_at: row.try_get("created_at")?,
            last_updated: row.try_get("last_updated")?,
        })
    }
}

/// A row from the `local_robots` table: one instance of a robot connected
/// offline that a recording has used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRobotRow {
    /// The local robot id, which recordings hold as their `robot_id`.
    pub robot_id: String,
    /// The robot instance, which recordings hold as their `robot_instance`.
    pub instance: i64,
    /// The robot name it is registered by.
    pub name: String,
    /// Whether the robot is shared.
    pub shared: bool,
    /// Whether its URDF package replaces one the backend already has.
    pub overwrite_urdf: bool,
    /// The backend's id, once registered.
    pub cloud_robot_id: Option<String>,
    /// How far registering it has got.
    pub status: LocalIdStatus,
    /// Why it is stuck `pending` or `failed`.
    pub detail: Option<String>,
    /// The backend is not asked about it again before this.
    pub next_attempt_at: Option<NaiveDateTime>,
}

impl LocalRobotRow {
    /// Decode a SQLite row into a [`LocalRobotRow`].
    pub(crate) fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        Ok(LocalRobotRow {
            robot_id: row.try_get("robot_id")?,
            instance: row.try_get("instance")?,
            name: row.try_get("name")?,
            shared: row.try_get("shared")?,
            overwrite_urdf: row.try_get("overwrite_urdf")?,
            cloud_robot_id: row.try_get("cloud_robot_id")?,
            status: parse_column(row, "status")?,
            detail: row.try_get("detail")?,
            next_attempt_at: row.try_get("next_attempt_at")?,
        })
    }
}

/// A row from the `local_datasets` table: a dataset selected offline that a
/// recording has used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDatasetRow {
    /// The local dataset id, which recordings hold as their `dataset_id`.
    pub dataset_id: String,
    /// The dataset name it is found or created by.
    pub name: String,
    /// Whether the dataset is shared.
    pub shared: bool,
    /// Description, used if the dataset is created.
    pub description: Option<String>,
    /// Tags, used if the dataset is created.
    pub tags: Vec<String>,
    /// The backend's id, once found or created.
    pub cloud_dataset_id: Option<String>,
    /// How far finding or creating it has got.
    pub status: LocalIdStatus,
    /// Why it is stuck `pending` or `failed`.
    pub detail: Option<String>,
    /// The backend is not asked about it again before this.
    pub next_attempt_at: Option<NaiveDateTime>,
}

impl LocalDatasetRow {
    /// Decode a SQLite row into a [`LocalDatasetRow`].
    pub(crate) fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        let tags: String = row.try_get("tags")?;
        let tags = serde_json::from_str(&tags).map_err(|error| sqlx::Error::ColumnDecode {
            index: "tags".to_string(),
            source: Box::new(error),
        })?;
        Ok(LocalDatasetRow {
            dataset_id: row.try_get("dataset_id")?,
            name: row.try_get("name")?,
            shared: row.try_get("shared")?,
            description: row.try_get("description")?,
            tags,
            cloud_dataset_id: row.try_get("cloud_dataset_id")?,
            status: parse_column(row, "status")?,
            detail: row.try_get("detail")?,
            next_attempt_at: row.try_get("next_attempt_at")?,
        })
    }
}

/// A row from the `recording_offline` table: the backend's ids for a
/// recording started offline once it has them, or why it failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingOfflineRow {
    /// The recording's local index.
    pub recording_index: i64,
    /// The backend's robot id, once resolved.
    pub cloud_robot_id: Option<String>,
    /// The backend's dataset id, once resolved.
    pub cloud_dataset_id: Option<String>,
    /// Why it failed, once a robot or dataset it used was refused: the
    /// refused one's detail, kept here as that row may be forgotten.
    pub failure: Option<String>,
    /// What was refused: `robot` or `dataset`.
    pub failure_kind: Option<String>,
    /// The refused robot's or dataset's name.
    pub failure_name: Option<String>,
}

impl RecordingOfflineRow {
    /// Decode a SQLite row into a [`RecordingOfflineRow`].
    pub(crate) fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        Ok(RecordingOfflineRow {
            recording_index: row.try_get("recording_index")?,
            cloud_robot_id: row.try_get("cloud_robot_id")?,
            cloud_dataset_id: row.try_get("cloud_dataset_id")?,
            failure: row.try_get("failure")?,
            failure_kind: row.try_get("failure_kind")?,
            failure_name: row.try_get("failure_name")?,
        })
    }
}

/// A row from the `traces` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRecord {
    /// Primary key (daemon-minted UUID).
    pub trace_id: String,
    /// Parent recording (local `recording_index`).
    pub recording_index: i64,
    /// Write lifecycle.
    pub write_status: TraceWriteStatus,
    /// Registration lifecycle.
    pub registration_status: TraceRegistrationStatus,
    /// Upload lifecycle.
    pub upload_status: TraceUploadStatus,
    /// Wire data-type label (e.g. `"video"`); free-form string carried verbatim.
    pub data_type: Option<String>,
    /// Producer-supplied data-type name.
    pub data_type_name: Option<String>,
    /// Filesystem path to the on-disk artefact.
    pub path: Option<String>,
    /// Bytes written so far.
    pub bytes_written: i64,
    /// Total bytes once finalised (`0` while in progress).
    pub total_bytes: i64,
    /// Bytes uploaded so far.
    pub bytes_uploaded: i64,
    /// Latest error code, if any.
    pub error_code: Option<TraceErrorCode>,
    /// Latest error message, if any.
    pub error_message: Option<String>,
    /// JSON-encoded `{filepath: session_uri}` map populated by the
    /// registration coordinator. The uploader reads this back on
    /// `ReadyForUpload` and dispatches one resumable upload per entry.
    pub upload_session_uris: Option<String>,
    /// When the status coordinator's `UPLOAD_COMPLETE` PUT for this trace
    /// was acknowledged by the backend. `NULL` until then; the reaper only
    /// reclaims recordings whose traces are all stamped (or have exhausted
    /// [`completion_report_attempts`](Self::completion_report_attempts)).
    /// Note: failed traces emit a terminal status through the same path, so
    /// a stamp means "final status acked", NOT "uploaded successfully" —
    /// `upload_status` stays the authority on upload success.
    pub completion_reported_at: Option<NaiveDateTime>,
    /// How many times the reconcile pass has re-driven this trace's
    /// completion PUT; reclaim opens once the attempts are exhausted.
    pub completion_report_attempts: i64,
    /// First-seen timestamp.
    pub created_at: NaiveDateTime,
    /// Last write timestamp; bumped on every row mutation.
    pub last_updated: NaiveDateTime,
}

impl TraceRecord {
    /// Decode a SQLite row into a [`TraceRecord`].
    pub(crate) fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        let write_status = parse_column::<TraceWriteStatus>(row, "write_status")?;
        let registration_status =
            parse_column::<TraceRegistrationStatus>(row, "registration_status")?;
        let upload_status = parse_column::<TraceUploadStatus>(row, "upload_status")?;
        let error_code = row
            .try_get::<Option<String>, _>("error_code")?
            .map(|raw| {
                TraceErrorCode::from_str(&raw).map_err(|error| decode_error("error_code", error))
            })
            .transpose()?;
        Ok(TraceRecord {
            trace_id: row.try_get("trace_id")?,
            recording_index: row.try_get("recording_index")?,
            write_status,
            registration_status,
            upload_status,
            data_type: row.try_get("data_type")?,
            data_type_name: row.try_get("data_type_name")?,
            path: row.try_get("path")?,
            bytes_written: row.try_get("bytes_written")?,
            total_bytes: row.try_get("total_bytes")?,
            bytes_uploaded: row.try_get("bytes_uploaded")?,
            error_code,
            error_message: row.try_get("error_message")?,
            upload_session_uris: row.try_get("upload_session_uris")?,
            completion_reported_at: row.try_get("completion_reported_at")?,
            completion_report_attempts: row.try_get("completion_report_attempts")?,
            created_at: row.try_get("created_at")?,
            last_updated: row.try_get("last_updated")?,
        })
    }
}

fn parse_column<T>(row: &sqlx::sqlite::SqliteRow, column: &'static str) -> Result<T, sqlx::Error>
where
    T: FromStr<Err = ParseEnumError>,
{
    let raw: String = row.try_get(column)?;
    T::from_str(&raw).map_err(|error| decode_error(column, error))
}

fn decode_error(column: &'static str, error: ParseEnumError) -> sqlx::Error {
    sqlx::Error::ColumnDecode {
        index: column.to_string(),
        source: Box::new(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_round_trip_through_str() {
        for status in [
            TraceWriteStatus::Pending,
            TraceWriteStatus::Initializing,
            TraceWriteStatus::Writing,
            TraceWriteStatus::PendingMetadata,
            TraceWriteStatus::Written,
            TraceWriteStatus::Failed,
        ] {
            assert_eq!(TraceWriteStatus::from_str(status.as_str()).unwrap(), status);
        }
    }

    #[test]
    fn enum_string_values_match_on_disk_contract() {
        // Spot-check a few values that integration tests assert on.
        assert_eq!(
            TraceWriteStatus::PendingMetadata.as_str(),
            "pending_metadata"
        );
        assert_eq!(TraceUploadStatus::Uploaded.as_str(), "uploaded");
        assert_eq!(ProgressReportStatus::Reported.as_str(), "reported");
        assert_eq!(TraceErrorCode::DiskFull.as_str(), "disk_full");
    }

    #[test]
    fn parse_rejects_unknown_value() {
        let error = TraceUploadStatus::from_str("bogus").unwrap_err();
        assert_eq!(error.value, "bogus");
        assert_eq!(error.kind, "TraceUploadStatus");
    }
}
