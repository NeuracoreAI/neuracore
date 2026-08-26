# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

The data daemon now reports the neuracore version it was built from, and the SDK checks it before it uses the daemon. A daemon left over from an earlier install is reported with the steps to fix it instead of being used silently.

The data daemon encodes video chunks faster on slow machines with a lighter preview scaling filter.

The data daemon now works with ffmpeg 8 and later. It selects the frame timing option that the installed ffmpeg accepts, so new and old ffmpeg builds are both supported.

Dataset video decoding now works with ffmpeg 8 and later too. It selects the frame timing option the installed ffmpeg accepts, instead of falling back to the slower PyAV decoder.

Synchronizing a recording is now asynchronous: the SDK starts the synchronization, waits for it to finish, and then downloads the episode directly from storage through a short-lived signed URL instead of receiving it inline from the API. Already-synchronized recordings are ready on the first check, so opening them no longer waits on multi-megabyte responses travelling through the API.

Datasets now support backward pagination: `reversed(dataset)` walks a dataset's recordings from newest to oldest directly against the backend, instead of loading every recording forward first. This is new — backward traversal has never previously been available in the SDK. Forward iteration, indexing, and slicing are unchanged. The SDK also now detects a stalled pagination cursor (for example, a backend that repeats the same page instead of advancing) and raises a clear error instead of silently loading duplicate recordings.
