# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Download checkpoints from cloud training runs. `neuracore training list-checkpoints -t <run>` lists a run's checkpoints and `neuracore training download-checkpoint -t <run> [--epoch N] [-o DIR] [--force]` downloads one (latest by default). The same is available in Python as `nc.list_training_checkpoints(job_id)` and `nc.download_training_checkpoint(job_id, epoch=None, output_dir=".", overwrite=False)`.

A recording started from the web no longer takes the frames logged before its start.

Fixed-choice hyperparameters of the built-in algorithms are now string enums, so the training wizard can offer them as dropdowns: Diffusion Policy's `process_type`, `noise_scheduler_type`, `prediction_type`, `beta_schedule` and `lr_scheduler_type`, CNNMLP's `image_backbone`, Pi0 and Pi05's `paligemma_variant`, `action_expert_variant` (including the test-scale `gemma_tiny`) and `compile_mode`, and Groot's `torch_compile_mode`. Defaults are unchanged and plain string values still work. Custom algorithms can declare their own; see the contribution guide.

Timestamps are now kept as integer microseconds. Every `timestamp` argument still takes float seconds, and the SDK rounds it once to microseconds. Logged samples, stored traces and synchronized data carry `timestamp_us` beside `timestamp`, and `timestamp` is the same instant rounded to the microsecond. The SDK now needs pydantic 2.12 or later. Two timestamps that round to the same microsecond on one stream are rejected as not increasing, and a timestamp that is not finite or is out of range raises `ValueError`.
