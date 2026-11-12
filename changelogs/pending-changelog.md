# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Breaking change: `recording.set_status` and `RecordingStatus` are removed, and recording metadata has no `status` or `qa_flag_reasons`. Use `recording.add_tag("flagged")` and `recording.remove_tag("flagged")` to flag and unflag a recording, and `recording.tags` to read its user and QA tags. This release needs `neuracore_types` 17.

The `StopRecording` message to the data daemon carries a list of user tag names. This is an internal change: no public function uses it yet, and the SDK and the daemon must be the same version.
