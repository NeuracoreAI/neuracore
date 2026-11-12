# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->
`nc.login()` no longer fails when the Neuracore server cannot be reached: it warns and continues offline, using the saved API key without checking it, and saves the given key only when none is saved. Calls that need the server then raise `ServerUnreachableError`, a subclass of `AuthenticationError`. `nc.stop_recording(wait=True)` while offline stops the recording and raises `ServerUnreachableError` straight away instead of waiting.
