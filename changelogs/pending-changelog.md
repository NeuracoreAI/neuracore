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

The data daemon keeps the tag names of a stop in its state store and sends them in the `/recording/stop` request, also after an offline period or a restart. A stop without tags sends the same request as before.

`nc.stop_recording` takes an optional `tags` list of user tag names, so a script tags a recording when it stops it. The names are normalized like `recording.add_tag`, and a name given twice gives one tag. An invalid name raises `ValueError` and the recording does not stop. The tags also reach a recording that was made offline.

Breaking change: `nc.connect_robot` raises `ValidationError` if the URDF has a visual mesh that is not STL or DAE, because the web viewer shows only these formats. The error lists each unsupported mesh file. Convert these meshes to STL or DAE before you connect the robot.

Resuming a multi-GPU training job downloads the checkpoint on rank 0 only, then the other ranks load that local file, so they cannot truncate it while another rank is reading it.

`nc.login()` no longer fails when the Neuracore server cannot be reached: it warns and continues offline, using the saved API key without checking it, and saves the given key only when none is saved. Authenticated requests to the server then raise `ServerUnreachableError`, a subclass of `AuthenticationError`. `nc.stop_recording(wait=True)` while offline stops the recording and raises `ServerUnreachableError` straight away instead of waiting.
