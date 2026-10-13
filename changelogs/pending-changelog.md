# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Download checkpoints from cloud training runs. `neuracore training list-checkpoints -t <run>` lists a run's checkpoints and `neuracore training download-checkpoint -t <run> [--epoch N] [-o DIR] [--force]` downloads one (latest by default). The same is available in Python as `nc.list_training_checkpoints(job_id)` and `nc.download_training_checkpoint(job_id, epoch=None, output_dir=".", overwrite=False)`.
