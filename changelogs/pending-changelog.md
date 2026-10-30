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

The built-in CNNMLP, DiffusionPolicy, Groot, Pi0 and Pi05 algorithms no longer list the H100 GPU, which is no longer offered. Groot, Pi0 and Pi05 now train on the A100 80GB only.

Requires neuracore_types 13, which no longer has `GPUType.NVIDIA_H100_80GB`.

Recordings that several machines record into are now saved with every machine's traces, and `stop_recording(wait=True)` returns once all of them have uploaded. A machine that misses a recording's stop while disconnected from the backend now finishes its part when it reconnects.
