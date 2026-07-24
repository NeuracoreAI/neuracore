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
