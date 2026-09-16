# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Every `timestamp` argument still takes float seconds, also accepts integer microsecond ticks, and defaults to the monotonic clock instead of the wall clock. Synchronized data and the live stream carry integer ticks in `timestamp`, and episodes carry `ticks_per_second`. Importers pass source clocks through as exact ticks. `get_cloud_recording_id` takes the start marker in ticks as `start_timestamp` instead of `timestamp_ns`.
