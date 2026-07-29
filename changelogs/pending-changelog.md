# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->
The Rust data daemon is now the default. Installs that ship the bundled
`data-daemon` binary — the published Linux x86_64 and Apple-Silicon macOS
wheels — use it without any opt-in; `NCD_RUST_DAEMON=1` is no longer needed.
Set `NCD_RUST_DAEMON=0` to pin a process back to the legacy Python daemon.
Where the bundled binary is absent (a source checkout that has not run
`rust/scripts/build_wheel_artefacts.sh`, or a platform with no published
wheel), the Python daemon is still used automatically.
