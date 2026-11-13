# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

CNNMLP can now predict the episode done flag logged with `nc.log_done()`. Add `DONE` to the output data types (RGB images are required) and CNNMLP trains a small classifier on attention-pooled image features with a class-balanced BCE loss next to its L1 action loss; `done_loss_weight`, `done_max_pos_weight`, `detach_done_encoder` and `done_use_proprio` control it. At inference the model also returns the done probability for the current step. Without `DONE` CNNMLP is unchanged and existing checkpoints still load.
