# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

`neuracore` now publishes Linux aarch64 wheels (`manylinux_2_28`) alongside the
existing x86_64 and macOS ones, so pinning an exact version
(`pip install neuracore==<version>`) works on 64-bit Arm Linux instead of
resolving back to the last pure-Python release.

Custom algorithms can now declare the GPU types they support with a static
`get_supported_gpus()` method. Neuracore checks this metadata before starting a
cloud training run, so users get an immediate error instead of launching an
algorithm on incompatible hardware.

Existing custom algorithms must add `get_supported_gpus()` and be uploaded again
before they can start new cloud training runs.
