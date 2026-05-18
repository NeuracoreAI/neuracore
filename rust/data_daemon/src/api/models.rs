//! Request and response shapes for the Neuracore backend.
//!
//! Mirrors the seven endpoints called out in §8 of the rewrite plan. The
//! daemon's serde types are kept thin: only fields the daemon writes or
//! reads are modelled, so a schema change on a field the daemon ignores does
//! not break the client.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One file the backend should expect for a trace registration request.
///
/// Matches the body of `POST /org/{org}/recording/traces/batch-register`,
/// `traces[].cloud_files[]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CloudFile {
    /// Path inside the trace's cloud directory, e.g. `"video/cam_0/lossy.mp4"`.
    pub filepath: String,
    /// MIME type, e.g. `"video/mp4"`.
    pub content_type: String,
}

/// One trace inside a batch-register request body.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RegisterTraceRequest {
    /// Recording the trace belongs to.
    pub recording_id: String,
    /// Wire data-type label (e.g. `"RGB_IMAGES"`).
    pub data_type: String,
    /// Trace identifier.
    pub trace_id: String,
    /// Files to register for this trace.
    pub cloud_files: Vec<CloudFile>,
}

/// Backend response payload for `POST /traces/batch-register`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BatchRegisterResponse {
    /// Traces the backend accepted, with the GCS resumable session URIs the
    /// daemon should PUT chunks to.
    #[serde(default)]
    pub registered_traces: Vec<RegisteredTrace>,
    /// Traces the backend rejected, with a per-trace error message.
    #[serde(default)]
    pub failed_traces: Vec<FailedTrace>,
}

/// One successful entry in the batch-register response.
#[derive(Debug, Clone, Deserialize)]
pub struct RegisteredTrace {
    /// Trace identifier accepted by the backend.
    pub trace_id: String,
    /// Map of `cloud_file.filepath → resumable session URI`. Optional because
    /// the backend may omit the field when the trace has no upload targets
    /// (e.g. a metadata-only trace).
    #[serde(default)]
    pub upload_session_uris: BTreeMap<String, String>,
}

/// One failed entry in the batch-register response.
#[derive(Debug, Clone, Deserialize)]
pub struct FailedTrace {
    /// Trace identifier the backend rejected.
    pub trace_id: String,
    /// Optional human-readable error message.
    #[serde(default)]
    pub error: Option<String>,
}

/// Response for `GET /recording/{rec}/resumable_upload_url`.
#[derive(Debug, Clone, Deserialize)]
pub struct ResumableUploadUrlResponse {
    /// Fresh resumable session URI.
    pub url: String,
}

/// Status value for the per-trace batch update API. Matches the wire enum
/// `RecordingDataTraceStatus` in `neuracore_types`.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum TraceStatusValue {
    /// Trace has been queued for upload.
    #[serde(rename = "QUEUED")]
    Queued,
    /// Upload of this trace has started.
    #[serde(rename = "UPLOAD_STARTED")]
    UploadStarted,
    /// Upload of this trace has completed.
    #[serde(rename = "UPLOAD_COMPLETE")]
    UploadComplete,
}

/// One per-trace update inside a batch-update request body.
///
/// Modelled to match the `TraceStatusUpdates` pydantic class.  Fields are
/// `Option`-wrapped and skip when null so the wire body only carries fields
/// the caller actually wants to change — mirroring the Python
/// `model_dump(mode="json", exclude_defaults=True)` call site.
#[derive(Debug, Clone, Serialize, Default)]
pub struct TraceStatusUpdate {
    /// Lifecycle status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<TraceStatusValue>,
    /// Bytes uploaded so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uploaded_bytes: Option<i64>,
    /// Total bytes once finalised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_register_body_matches_python_layout() {
        let request = RegisterTraceRequest {
            recording_id: "rec-1".to_string(),
            data_type: "RGB_IMAGES".to_string(),
            trace_id: "trace-1".to_string(),
            cloud_files: vec![CloudFile {
                filepath: "rgb/cam_0/lossy.mp4".to_string(),
                content_type: "video/mp4".to_string(),
            }],
        };
        let body = serde_json::to_value(serde_json::json!({"traces": [request]})).unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "traces": [{
                    "recording_id": "rec-1",
                    "data_type": "RGB_IMAGES",
                    "trace_id": "trace-1",
                    "cloud_files": [{
                        "filepath": "rgb/cam_0/lossy.mp4",
                        "content_type": "video/mp4"
                    }]
                }]
            })
        );
    }

    #[test]
    fn trace_status_update_strips_unset_fields() {
        let update = TraceStatusUpdate {
            status: Some(TraceStatusValue::UploadComplete),
            uploaded_bytes: Some(42),
            total_bytes: None,
        };
        let json = serde_json::to_value(&update).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"status": "UPLOAD_COMPLETE", "uploaded_bytes": 42})
        );
    }

    #[test]
    fn batch_register_response_round_trips() {
        let body = serde_json::json!({
            "registered_traces": [{
                "trace_id": "trace-1",
                "upload_session_uris": {"rgb/cam_0/lossy.mp4": "https://upload.example/1"}
            }],
            "failed_traces": [{
                "trace_id": "trace-2",
                "error": "bad cloud file"
            }]
        });
        let response: BatchRegisterResponse = serde_json::from_value(body).unwrap();
        assert_eq!(response.registered_traces.len(), 1);
        assert_eq!(response.registered_traces[0].trace_id, "trace-1");
        assert_eq!(
            response.registered_traces[0]
                .upload_session_uris
                .get("rgb/cam_0/lossy.mp4")
                .map(String::as_str),
            Some("https://upload.example/1")
        );
        assert_eq!(response.failed_traces[0].trace_id, "trace-2");
    }
}
