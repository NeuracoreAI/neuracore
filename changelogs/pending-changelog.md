# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

The requests-based HTTP session now retries requests that fail because the
connection was dropped before a response arrived (TCP resets, stale keep-alive
reuse), matching the retry behaviour of the aiohttp stack.

Custom algorithms can now declare the GPU types they support with a static
`get_supported_gpus()` method. Neuracore checks this metadata before starting a
cloud training run, so users get an immediate error instead of launching an
algorithm on incompatible hardware.

Existing custom algorithms must add `get_supported_gpus()` and be uploaded again
before they can start new cloud training runs.
