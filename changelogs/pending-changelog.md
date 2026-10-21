# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

- A training run started with the `training_id` of a local training job reports how it ended: COMPLETED when training finishes, FAILED on an error, CANCELLED when stopped with Ctrl+C. Its progress, checkpoints and model then show on the Training page like a cloud job's.
