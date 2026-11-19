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

`nc.create_dataset` now goes through the data daemon, online or offline, and returns straight away: the daemon has the server find or create the dataset in the background. `nc.get_dataset` still asks the server directly, so after an offline login it raises `DatasetError`; use `nc.create_dataset` to record into a dataset offline. The server id of a dataset selected through the daemon is fetched when it is first read, which waits for the server, as passing `wait=True` does: that raises the server's refusal (`DatasetError`), or `ServerUnreachableError` while offline. Offline, this needs an organization saved by an earlier `neuracore select-org`.

`nc.connect_robot` now goes through the data daemon, online or offline, and returns straight away: the daemon has the server register the robot instance, and upload its description if the server needs it, in the background. Several processes connecting the same robot instance share it, and record it with no network connection. The robot's server id is fetched when it is first read, which waits for the server, as passing `wait=True` does: that raises `RobotArchivedError` for an archived robot, `RobotError` for another refusal, or `ServerUnreachableError` while offline. Live data streaming starts once the server has registered the robot, if it does within 10 minutes of connecting. Offline, this needs an organization saved by an earlier `neuracore select-org`. `nc.start_recording` refuses a robot and a dataset in different organizations.

The first `nc.connect_robot`, `nc.create_dataset` or `nc.start_recording` in each process now warns about anything that keeps recordings from uploading: no API key saved, no organization selected, an archived robot, a rejected API key, an account that cannot record in the organization a robot or dataset is in, a dataset name taken, robots, datasets and recordings the server refused, and recordings kept on the machine because the server cannot be reached. Nothing is shown when nothing is wrong. `neuracore data-daemon status` now also shows whether the daemon can reach the server, the recordings still to upload and the same problems.
