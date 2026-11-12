# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

The `StopRecording` message to the data daemon carries a list of user tag names. This is an internal change: no public function uses it yet, and the SDK and the daemon must be the same version.

The data daemon keeps the tag names of a stop in its state store and sends them in the `/recording/stop` request, also after an offline period or a restart. A stop without tags sends the same request as before.

`nc.stop_recording` takes an optional `tags` list of user tag names, so a script tags a recording when it stops it. The names are normalized like `recording.add_tag`, and a name given twice gives one tag. An invalid name raises `ValueError` and the recording does not stop. The tags also reach a recording that was made offline.
