//! Strongly-typed rows and lifecycle enums for the daemon's SQLite tables.
//!
//! The enum string values are part of the integration-test contract: helpers
//! under `tests/integration/platform/data_daemon/shared/db_helpers.py` read
//! the string columns directly, so the stored spellings must stay stable.

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

/// A row from the `recordings` table.
///
/// The daemon owns recording identity: `recording_index` is the local primary
/// key (AUTOINCREMENT), allocated when the `StartRecording` envelope is first
/// seen; `recording_id` is the cloud handle, filled asynchronously by the
/// recording-start notifier or minted on demand by the registration
/// coordinator. The two are independent — never aliased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingRow {
    /// Local primary key (AUTOINCREMENT). The daemon keys every internal
    /// structure and the `traces` foreign key on this.
    pub recording_index: i64,
    /// Cloud handle. `None` until `/recording/start` is notified (or a UUID is
    /// minted on demand for an offline recording). Every cloud URL reads this
    /// straight from the row.
    pub recording_id: Option<String>,
    /// Organisation that owns the recording (backfilled when known).
    pub org_id: Option<String>,
    /// Robot identifier — first half of the source key.
    pub robot_id: Option<String>,
    /// Robot instance — second half of the source key.
    pub robot_instance: Option<i64>,
    /// Robot human-readable name, when supplied.
    pub robot_name: Option<String>,
    /// Dataset identifier, when supplied.
    pub dataset_id: Option<String>,
    /// Dataset human-readable name, when supplied.
    pub dataset_name: Option<String>,
    /// Producer capture-clock window lower bound (Unix nanoseconds).
    pub start_timestamp_ns: Option<i64>,
    /// Producer capture-clock window upper bound (Unix nanoseconds).
    pub stop_timestamp_ns: Option<i64>,
    /// Expected number of traces, set when the producer declares it.
    pub expected_trace_count: Option<i64>,
    /// `1` once the expected trace count has been reported to the backend.
    pub expected_trace_count_reported: i64,
    /// Traces that have reached the `uploaded` terminal state.
    pub uploaded_trace_count: i64,
    /// Progress-report lifecycle for this recording.
    pub progress_reported: ProgressReportStatus,
    /// Daemon wall-clock time the recording was opened.
    pub started_at: Option<NaiveDateTime>,
    /// Set when the producer issues a stop command.
    pub stopped_at: Option<NaiveDateTime>,
    /// Set when the producer issues a cancel command. Cancelled recordings
    /// are ignored by the cloud coordinators and skipped by the progress
    /// reporter.
    pub cancelled_at: Option<NaiveDateTime>,
    /// Set when the recording-start notifier successfully POSTed
    /// `/recording/start` and persisted the cloud `recording_id`.
    pub backend_start_notified_at: Option<NaiveDateTime>,
    /// Set when the recording-start notifier permanently skipped
    /// `/recording/start` because the recording is older than the recency
    /// bound (the registration coordinator mints a cloud id on demand
    /// instead). The recording row is **not** cancelled — it still uploads
    /// best-effort.
    pub backend_start_failed_at: Option<NaiveDateTime>,
    /// Set when the recording-stop notifier successfully POSTed
    /// `/recording/stop` to the backend. `None` means the backend has not
    /// yet been notified — typically a recording stopped while the daemon
    /// was offline; the notifier sweeps these on startup.
    pub backend_stop_notified_at: Option<NaiveDateTime>,
    /// Set when the recording-cancel notifier successfully POSTed
    /// `/recording/cancel` to the backend. `None` means either the recording
    /// was never cancelled, or cancellation has not yet been notified.
    pub backend_cancel_notified_at: Option<NaiveDateTime>,
    /// First-seen timestamp.
    pub created_at: NaiveDateTime,
    /// Last write timestamp; bumped on every row mutation.
    pub last_updated: NaiveDateTime,
}

impl RecordingRow {
    /// Decode a SQLite row into a [`RecordingRow`].
    pub(crate) fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        let progress_reported = parse_column::<ProgressReportStatus>(row, "progress_reported")?;
        Ok(RecordingRow {
            recording_index: row.try_get("recording_index")?,
            recording_id: row.try_get("recording_id")?,
            org_id: row.try_get("org_id")?,
            robot_id: row.try_get("robot_id")?,
            robot_instance: row.try_get("robot_instance")?,
            robot_name: row.try_get("robot_name")?,
            dataset_id: row.try_get("dataset_id")?,
            dataset_name: row.try_get("dataset_name")?,
            start_timestamp_ns: row.try_get("start_timestamp_ns")?,
            stop_timestamp_ns: row.try_get("stop_timestamp_ns")?,
            expected_trace_count: row.try_get("expected_trace_count")?,
            expected_trace_count_reported: row.try_get("expected_trace_count_reported")?,
            uploaded_trace_count: row.try_get("uploaded_trace_count")?,
            progress_reported,
            started_at: row.try_get("started_at")?,
            stopped_at: row.try_get("stopped_at")?,
            cancelled_at: row.try_get("cancelled_at")?,
            backend_start_notified_at: row.try_get("backend_start_notified_at")?,
            backend_start_failed_at: row.try_get("backend_start_failed_at")?,
            backend_stop_notified_at: row.try_get("backend_stop_notified_at")?,
            backend_cancel_notified_at: row.try_get("backend_cancel_notified_at")?,
            created_at: row.try_get("created_at")?,
            last_updated: row.try_get("last_updated")?,
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
    /// Dataset ID, when supplied.
    pub dataset_id: Option<String>,
    /// Dataset name, when supplied.
    pub dataset_name: Option<String>,
    /// Robot name, when supplied.
    pub robot_name: Option<String>,
    /// Robot ID, when supplied.
    pub robot_id: Option<String>,
    /// Robot instance number, when supplied.
    pub robot_instance: Option<i64>,
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
    /// Number of upload attempts made so far.
    pub num_upload_attempts: i64,
    /// Next scheduled retry time, when in backoff.
    pub next_retry_at: Option<NaiveDateTime>,
    /// JSON-encoded `{filepath: session_uri}` map populated by the
    /// registration coordinator. The uploader reads this back on
    /// `ReadyForUpload` and dispatches one resumable upload per entry.
    pub upload_session_uris: Option<String>,
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
            dataset_id: row.try_get("dataset_id")?,
            dataset_name: row.try_get("dataset_name")?,
            robot_name: row.try_get("robot_name")?,
            robot_id: row.try_get("robot_id")?,
            robot_instance: row.try_get("robot_instance")?,
            path: row.try_get("path")?,
            bytes_written: row.try_get("bytes_written")?,
            total_bytes: row.try_get("total_bytes")?,
            bytes_uploaded: row.try_get("bytes_uploaded")?,
            error_code,
            error_message: row.try_get("error_message")?,
            num_upload_attempts: row.try_get("num_upload_attempts")?,
            next_retry_at: row.try_get("next_retry_at")?,
            upload_session_uris: row.try_get("upload_session_uris")?,
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
    fn enum_string_values_match_python() {
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
