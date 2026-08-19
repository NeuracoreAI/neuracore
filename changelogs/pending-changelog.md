# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

The data daemon now reports the neuracore version it was built from, and the SDK checks it before it uses the daemon. A daemon left over from an earlier install is reported with the steps to fix it instead of being used silently.

The data daemon encodes video chunks faster on slow machines with a lighter preview scaling filter.
