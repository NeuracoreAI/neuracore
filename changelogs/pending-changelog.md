# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Timestamps are now kept as integer microseconds. Every `timestamp` argument still takes float seconds, and the SDK rounds it once to microseconds. Logged samples, stored traces and synchronized data carry `timestamp_us` beside `timestamp`, and `timestamp` is the same instant rounded to the microsecond. The SDK now needs pydantic 2.12 or later. Two timestamps that round to the same microsecond on one stream are rejected as not increasing, and a timestamp that is not finite or is out of range raises `ValueError`.

Diffusion Policy can now predict the episode done flag logged with `nc.log_done()`. Add `DONE` to the output data types (RGB images are required) and Diffusion Policy trains a small classifier on attention-pooled image backbone features with a class-balanced BCE loss next to its diffusion loss; `done_loss_weight`, `done_max_pos_weight`, `detach_done_encoder` and `done_use_proprio` control it. The classifier reads only the observations, so it is computed once per inference call rather than at every denoising step, and the model also returns the done probability for the current step. This works with both the diffusion and flow matching processes. Without `DONE` Diffusion Policy is unchanged and existing checkpoints still load.
