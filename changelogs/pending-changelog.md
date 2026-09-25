# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Datasets now support backward pagination: `reversed(dataset)` walks a dataset's recordings from oldest to newest directly against the backend, instead of loading every recording forward first. This is new — backward traversal has never previously been available in the SDK. Forward iteration, indexing, and slicing are unchanged and continue to show the newest recordings first. The SDK also now detects a stalled pagination cursor (for example, a backend that repeats the same page instead of advancing) and raises a clear error instead of silently loading duplicate recordings.

Adds the `h264_fast` video codec: the same single lossy-only RGB video as `h264_medium`, encoded at libx264's `veryfast` preset instead of `medium`. On measured camera footage it encodes 2.4-3x quicker and uploads 11-22% fewer bytes, for about 2 dB of PSNR — pick it when getting data to the cloud quickly matters more than the last of the image fidelity. Select it with `nc.set_video_encoding_options(codec=nc.Codec.H264_FAST)`, `--video-codec h264_fast`, or `NCD_VIDEO_CODEC=h264_fast`.
