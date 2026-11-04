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

Tag recordings in a dataset. `recording.add_tag(name)` and `recording.remove_tag(name)` add and remove user tags, and `recording.tags` lists the user and QA tags of a recording. `recording.set_status` is deprecated: `FLAGGED` adds the tag `flagged`, `NORMAL` removes it, and `QA_FLAGGED` raises `ValueError`. `recording.set_notes` no longer sends the status.

`nc.connect_robot` now raises `RobotError` for an archived robot instead of a warning, and starts no streaming or data daemon for it. The error tells you to unarchive the robot on the Archived Robots page of the Neuracore dashboard. Older SDK versions show a generic 409 error when the backend refuses an archived robot, and cannot rename an archived robot.

`nc.update_robot_name` now renames through the rename route, so it works for archived robots and registers no instance. The `instance` argument is removed, so a call that passes `instance` now raises `TypeError`. A key that is neither a robot ID nor the name of a robot connected in this session now fails with "Robot not found" instead of creating a new robot.
