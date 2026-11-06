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

Training can now learn a done flag on datasets that never logged one. Set `synthesize_done_tail_s` (for example `0.5`) and each episode is extended after synchronization by that many seconds of its last frame, labelled done, with every real frame labelled not done. The setting is ignored, with a warning, when the dataset already logs DONE, and the run fails before fetching data if the algorithm cannot output DONE.
