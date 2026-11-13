# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->
`nc.login()` no longer fails when the Neuracore server cannot be reached: it warns and continues offline, using the saved API key without checking it, and saves the given key only when none is saved. Calls that need the server then raise `ServerUnreachableError`, a subclass of `AuthenticationError`. `nc.stop_recording(wait=True)` while offline stops the recording and raises `ServerUnreachableError` straight away instead of waiting.

After an offline login, `nc.create_dataset` and `nc.get_dataset` by name return a dataset kept on this machine under a local id that every process works out from the name. The data daemon finds or creates it on the server, by name and with its description and tags, before uploading recordings made into it. `nc.get_dataset(id=...)` raises `ServerUnreachableError` while offline.

After an offline login, `nc.connect_robot` sets the robot up on this machine under a local id that every process works out from the name, and builds and saves its URDF package. Several processes can then record it together with no network connection. Once the server can be reached, the data daemon uploads the recordings, registering the robot and uploading its description before the first one. A robot never recorded with is not registered, and its saved files are deleted a day after it was last connected. A robot connected offline streams no live data.

The first `nc.connect_robot` or `nc.start_recording` in each process now warns about anything that keeps recordings from uploading: no API key saved, no organization selected, an archived robot, a rejected API key, a dataset name taken, recordings the server refused, and recordings kept on the machine because the server cannot be reached. Nothing is shown when nothing is wrong. `neuracore data-daemon status` now also shows whether the daemon can reach the server, the recordings still to upload and the same problems.
