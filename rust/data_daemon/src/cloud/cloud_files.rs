//! Derive the cloud-file list for a trace from its data-type label.
//!
//! The data type partitions into `JSON` vs `RGB` content. The classification
//! works directly off the wire string so an unrecognised data type still
//! registers (as JSON) rather than being refused.

use crate::api::models::CloudFile;

// The artefact filenames are wire-critical and owned by `storage::paths`, where
// the on-disk writers stamp them. Re-export rather than redefine so there is a
// single source of truth — two copies could silently drift and break the
// upload↔disk filename contract.
pub use crate::storage::paths::{
    LOSSLESS_VIDEO_FILENAME as LOSSLESS_VIDEO_NAME, LOSSY_VIDEO_FILENAME as LOSSY_VIDEO_NAME,
    TRACE_JSON_FILENAME as TRACE_FILE,
};

/// Wire-side classification used to build the cloud-file list. The set of
/// data types that produce video is small and stable, so a hard-coded match
/// on the wire string is good enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentKind {
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
///
/// This is the video-family artefact predicate; it is deliberately broader than
/// the lossy-codec predicate ([`crate::encoding::video_encoder::LossyVideoCodec::for_trace`]),
/// which is RGB-only (depth is video-family but never lossy-eligible). Any new
/// video-family type added here must be considered there too, or it could lose
/// its lossless archive.
fn content_type_for(data_type: &str) -> ContentKind {
    match data_type {
        "RGB_IMAGES" | "DEPTH_IMAGES" => ContentKind::Rgb,
        _ => ContentKind::Json,
    }
}

/// The MIME content-type the daemon registers (and re-acquires session URIs)
/// for an artefact, keyed off its filename suffix. The single source of truth
/// for the mapping, shared by [`cloud_file_list`] and the uploader's session
/// refresh so the two can't disagree. Only the `.mp4` video artefacts are
/// `video/mp4`; everything else (the JSON trace / sidecar) is `application/json`.
pub fn content_type_for_filename(filename: &str) -> &'static str {
    if filename.ends_with(".mp4") {
        "video/mp4"
    } else {
        "application/json"
    }
}

/// Build the cloud-file list for a trace.
///
/// `data_type` is the wire label. `data_type_name` is the producer-supplied
/// alias (e.g. camera name); when missing we fall back to a single underscore
/// so the path is well-formed.
///
/// For RGB content the lossless archive is registered unless `lossy_only` is set
/// (`nc.Codec.H264_MEDIUM`), where the daemon writes just `lossy.mp4`. The caller
/// derives `lossy_only` from the resolved codec — the same source and predicate
/// the encoder uses (see [`crate::encoding::video_encoder::LossyVideoCodec::for_trace`]);
/// because the codec is fixed for a recording (set before start), the file list
/// matches what the encoder produces. `lossy_only` is ignored for non-RGB content.
///
/// The decision is NOT taken from disk: registration runs *while the recording
/// is still writing* ("pre-registration", see the module docs on the
/// registration coordinator), so the video files do not exist yet when this is
/// called.
pub fn cloud_file_list(
    data_type: &str,
    data_type_name: Option<&str>,
    lossy_only: bool,
) -> Vec<CloudFile> {
    let prefix = format!("{data_type}/{}", data_type_name.unwrap_or("_"));
    let mut filenames = Vec::with_capacity(3);
    if matches!(content_type_for(data_type), ContentKind::Rgb) {
        filenames.push(LOSSY_VIDEO_NAME);
        if !lossy_only {
            filenames.push(LOSSLESS_VIDEO_NAME);
        }
    }
    filenames.push(TRACE_FILE);
    filenames
        .into_iter()
        .map(|filename| CloudFile {
            filepath: format!("{prefix}/{filename}"),
            content_type: content_type_for_filename(filename).to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_trace_lists_only_trace_json() {
        let files = cloud_file_list("JOINT_POSITIONS", Some("arm0"), false);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filepath, "JOINT_POSITIONS/arm0/trace.json");
        assert_eq!(files[0].content_type, "application/json");
    }

    #[test]
    fn rgb_trace_lists_both_mp4_outputs_plus_sidecar() {
        let files = cloud_file_list("RGB_IMAGES", Some("cam_0"), false);
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
    fn rgb_trace_lossy_only_omits_lossless() {
        // Lossy-only (nc.Codec.H264_MEDIUM): register just the lossy video plus
        // its sidecar. Regression guard for the phantom-lossless bug — the
        // daemon must not register a lossless.mp4 it never writes, or the
        // uploader fails the RGB trace on the missing artefact.
        let files = cloud_file_list("RGB_IMAGES", Some("cam_0"), true);
        let paths: Vec<&str> = files.iter().map(|f| f.filepath.as_str()).collect();
        assert_eq!(
            paths,
            vec!["RGB_IMAGES/cam_0/lossy.mp4", "RGB_IMAGES/cam_0/trace.json"]
        );
    }

    #[test]
    fn depth_trace_ignores_lossy_only_and_keeps_lossless() {
        // Depth always keeps its lossless storage even when a lossy codec is
        // selected: the RGB-only gate lives in `LossyVideoCodec::for_trace`, so
        // a DEPTH_IMAGES trace never reaches here with `lossy_only == true`, but
        // pin the both-files layout regardless as a belt-and-braces guard.
        let files = cloud_file_list("DEPTH_IMAGES", Some("cam_0"), false);
        let paths: Vec<&str> = files.iter().map(|f| f.filepath.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "DEPTH_IMAGES/cam_0/lossy.mp4",
                "DEPTH_IMAGES/cam_0/lossless.mp4",
                "DEPTH_IMAGES/cam_0/trace.json",
            ]
        );
    }

    #[test]
    fn missing_data_type_name_falls_back_to_underscore() {
        let files = cloud_file_list("JOINT_POSITIONS", None, false);
        assert_eq!(files[0].filepath, "JOINT_POSITIONS/_/trace.json");
    }

    #[test]
    fn registration_decision_matches_encoder_output() {
        use crate::encoding::video_encoder::LossyVideoCodec;

        // Reproduce the registration coordinator's per-trace decision end to end
        // (`for_trace(...).is_lossy_only()` feeding `cloud_file_list`) so the
        // wiring can't regress into the phantom-lossless bug: the registered
        // file set must match exactly what the encoder writes for that codec.
        let registered = |data_type: &str, codec: Option<&str>| {
            let lossy_only = LossyVideoCodec::for_trace(data_type, codec).is_lossy_only();
            cloud_file_list(data_type, Some("cam"), lossy_only)
                .into_iter()
                .map(|file| file.filepath)
                .collect::<Vec<_>>()
        };

        // RGB + h264_medium: lossy-only, so NO lossless is registered (C1).
        assert_eq!(
            registered("RGB_IMAGES", Some("h264_medium")),
            vec!["RGB_IMAGES/cam/lossy.mp4", "RGB_IMAGES/cam/trace.json"]
        );
        // RGB default and depth-under-lossy both still register the lossless.
        assert!(registered("RGB_IMAGES", None)
            .iter()
            .any(|path| path.ends_with("lossless.mp4")));
        assert!(registered("DEPTH_IMAGES", Some("h264_medium"))
            .iter()
            .any(|path| path.ends_with("lossless.mp4")));
    }
}
