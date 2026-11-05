# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Timestamps are now kept as integer microseconds. Every `timestamp` argument still takes float seconds, and the SDK rounds it once to microseconds. Logged samples, stored traces and synchronized data carry `timestamp_us` beside `timestamp`, and `timestamp` is the same instant rounded to the microsecond. The SDK now needs pydantic 2.12 or later. Two timestamps that round to the same microsecond on one stream are rejected as not increasing, and a timestamp that is not finite or is out of range raises `ValueError`.

GR00T N1.6 can now predict the episode done flag logged with `nc.log_done()`. Add `DONE` to the output data types (RGB images are required) and GR00T trains a small classifier on attention-pooled VLM image-token features, the same observation context the action head attends to, with a class-balanced BCE loss next to its flow matching loss; `done_loss_weight`, `done_max_pos_weight`, `detach_done_encoder` and `done_use_proprio` control it. At inference the model also returns the done probability for the current step, computed once per call rather than per denoising step. Without `DONE` GR00T is unchanged and existing checkpoints still load.
