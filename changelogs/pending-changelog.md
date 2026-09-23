# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Datasets now support backward pagination: `reversed(dataset)` walks a dataset's recordings from oldest to newest directly against the backend, instead of loading every recording forward first. This is new — backward traversal has never previously been available in the SDK. Forward iteration, indexing, and slicing are unchanged and continue to show the newest recordings first. The SDK also now detects a stalled pagination cursor (for example, a backend that repeats the same page instead of advancing) and raises a clear error instead of silently loading duplicate recordings.
