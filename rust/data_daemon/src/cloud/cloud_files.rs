//! Derive the cloud-file list for a trace from its data-type label.
//!
//! The data type partitions into `JSON` vs `RGB` content. The classification
//! works directly off the wire string so an unrecognised data type still
//! registers (as JSON) rather than being refused.

use crate::api::models::CloudFile;

/// Filename of the lossy MP4 emitted for video traces.
pub const LOSSY_VIDEO_NAME: &str = "lossy.mp4";
/// Filename of the lossless MP4 emitted for video traces.
pub const LOSSLESS_VIDEO_NAME: &str = "lossless.mp4";
/// Filename of the JSON trace artefact (or sidecar metadata for video).
pub const TRACE_FILE: &str = "trace.json";

/// Wire-side classification used to build the cloud-file list. The set of
/// data types that produce video is small and stable, so a hard-coded match
/// on the wire string is good enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    /// JSON-only payload (scalar / sensor / event traces).
    Json,
    /// RGB video payload: lossy + lossless mp4 plus a JSON sidecar.
    Rgb,
}

/// Classify a data-type wire label.
///
/// Anything not recognised is treated as JSON.
///
/// `DEPTH_IMAGES` is intentionally mapped to `Rgb`: the upload pipeline uses
/// the same `lossy.mp4` + `lossless.mp4` artefact pair to carry depth frames
/// packed into RGB channels. Diverging here would register a different
/// cloud-file set than the backend expects and break wire compatibility.
pub fn content_type_for(data_type: &str) -> ContentKind {
    match data_type {
        "RGB_IMAGES" | "DEPTH_IMAGES" => ContentKind::Rgb,
        _ => ContentKind::Json,
    }
}

/// Build the cloud-file list for a trace.
///
/// `data_type` is the wire label. `data_type_name` is the producer-supplied
/// alias (e.g. camera name); when missing we fall back to a single underscore
/// so the path is well-formed.
pub fn cloud_file_list(data_type: &str, data_type_name: Option<&str>) -> Vec<CloudFile> {
    let prefix = format!("{data_type}/{}", data_type_name.unwrap_or("_"));
    let mut files = Vec::with_capacity(3);
    if matches!(content_type_for(data_type), ContentKind::Rgb) {
        files.push(CloudFile {
            filepath: format!("{prefix}/{LOSSY_VIDEO_NAME}"),
            content_type: "video/mp4".to_string(),
        });
        files.push(CloudFile {
            filepath: format!("{prefix}/{LOSSLESS_VIDEO_NAME}"),
            content_type: "video/mp4".to_string(),
        });
    }
    files.push(CloudFile {
        filepath: format!("{prefix}/{TRACE_FILE}"),
        content_type: "application/json".to_string(),
    });
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_trace_lists_only_trace_json() {
        let files = cloud_file_list("JOINT_POSITIONS", Some("arm0"));
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filepath, "JOINT_POSITIONS/arm0/trace.json");
        assert_eq!(files[0].content_type, "application/json");
    }

    #[test]
    fn rgb_trace_lists_both_mp4_outputs_plus_sidecar() {
        let files = cloud_file_list("RGB_IMAGES", Some("cam_0"));
        let paths: Vec<&str> = files.iter().map(|f| f.filepath.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "RGB_IMAGES/cam_0/lossy.mp4",
                "RGB_IMAGES/cam_0/lossless.mp4",
                "RGB_IMAGES/cam_0/trace.json",
            ]
        );
        assert_eq!(files[0].content_type, "video/mp4");
        assert_eq!(files[1].content_type, "video/mp4");
        assert_eq!(files[2].content_type, "application/json");
    }

    #[test]
    fn missing_data_type_name_falls_back_to_underscore() {
        let files = cloud_file_list("JOINT_POSITIONS", None);
        assert_eq!(files[0].filepath, "JOINT_POSITIONS/_/trace.json");
    }
}
