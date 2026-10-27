# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Timestamps are now kept as integer microseconds. Every `timestamp` argument still takes float seconds, and the SDK rounds it once to microseconds. Logged samples, stored traces and synchronized data carry `timestamp_us` beside `timestamp`, and `timestamp` is the same instant rounded to the microsecond. The SDK now needs pydantic 2.12 or later. Two timestamps that round to the same microsecond on one stream are rejected as not increasing, and a timestamp that is not finite or is out of range raises `ValueError`.

Recordings that several machines record into are now saved with every machine's traces, and `stop_recording(wait=True)` returns once all of them have uploaded. A machine that misses a recording's stop while disconnected from the backend now finishes its part when it reconnects.

Tag recordings in a dataset. `recording.add_tag(name)` and `recording.remove_tag(name)` add and remove user tags, and `recording.tags` lists the user and QA tags of a recording. `recording.set_notes` no longer sends the status.

Breaking change: `recording.set_status` and `RecordingStatus` are removed, and recording metadata has no `status` or `qa_flag_reasons`. Use `recording.add_tag("flagged")` and `recording.remove_tag("flagged")` to flag and unflag a recording, and `recording.tags` to read its user and QA tags. This release needs `neuracore_types` 16.
