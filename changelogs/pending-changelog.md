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

The data daemon batches video chunk encodes under backlog, so a machine that falls behind recovers with fewer ffmpeg invocations.

Final videos now carry the exact capture timing of every frame, so chunk boundaries no longer drift and real capture gaps are preserved.
