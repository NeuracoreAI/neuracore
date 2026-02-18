# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

### Dataset & Recording Editing:

Adds Recording & Dataset metadata editing methods:
    Datasets:
     - `dataset.set_name(name)`
     - `dataset.set_description(description)`
     - `dataset.set_tag(tag)`
     - `dataset.add_tag(status)`
    Recordings:
     - `recording.set_status(status)`
     - `recording.set_notes(notes)`

Please see the new examples for how to use these methods