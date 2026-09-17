# Dataset export

Export a Neuracore dataset locally to **generic MCAP**, with one MCAP file per
recording. The CLI runs in the foreground; keep it running until it finishes.
There is no remote job or microservice in this first version.

```bash
pip install 'neuracore[export]'
neuracore login
neuracore select-org
neuracore export mcap --dataset 'My dataset' --output ./exports/my-dataset
# Or select an exact dataset ID:
neuracore export mcap --dataset-id DATASET_ID --output ./exports/by-id
```

Use exactly one dataset selector. Authentication also supports the existing
`NEURACORE_API_KEY` environment variable. The output directory must not exist.

```text
my-dataset/
  manifest.json
  nc_Pick and place.mcap
  nc_Stack blocks.mcap
```

Filenames use `nc_RECORDING_NAME.mcap`. Filesystem-unsafe characters are replaced
with underscores; duplicate names receive numeric suffixes such as `_2`. Empty
names use `nc_recording.mcap`. Terminal progress and validation errors use recording
names. Recording IDs remain in the manifest and embedded metadata for traceability.

The manifest maps filenames to recording IDs and records the dataset name, ID,
exporter version, selected recording IDs and completion status. Selection is
collected before conversion starts; concurrent dataset edits during pagination
are not a transactional snapshot.

## Contents and compatibility

- Each sensor has a JSON channel at `/neuracore/{DATA_TYPE}/{escaped_sensor_name}`.
  Sensor names are percent-encoded in topics and preserved in channel metadata.
- Messages retain the original trace JSON, sample order within each sensor and
  original timestamps. MCAP log/publish timestamps are the trace's Unix seconds
  converted to integer nanoseconds. No synchronisation or resampling is applied.
- JSON schemas describe the common timestamp field and allow the original
  Neuracore-specific fields. These are **not ROS message schemas**.
- Original camera videos are embedded as MCAP attachments. RGB prefers
  `lossless.mp4`, falling back to `lossy.mp4` only when the former is absent;
  depth requires `lossless.mp4`. Frame indices remain in the sensor messages.
- Point-cloud `trace.bin` files are embedded unchanged, with their original
  frame indices/offsets in the trace messages.
- A channel's `attachment` metadata identifies its binary payload. Attachment
  names are recording-relative storage paths, not local extraction instructions.
- Original recording metadata is included as the `neuracore.recording` MCAP
  metadata record, under its `json` key.

These files are readable with the standard MCAP reader. Viewing camera images or
point clouds requires interpreting the Neuracore trace and corresponding media
attachment; this version does not provide automatic ROS/Foxglove visualisation.
It does not export LeRobot datasets.

## Limits and failures

Recordings must be completed and have a stored sensor manifest covering all
recorded data types. Older recordings without that manifest are rejected rather
than silently omitting sensors. Missing media, invalid timestamps and invalid
traces fail the export. No additional lossy conversion is performed.

Recordings and sensors are processed sequentially. The current implementation
loads one sensor trace and one media file into memory, and the MCAP writer can
make additional copies of the media. Allow several times the largest sensor
file size in available memory. Output chunks are uncompressed in this version;
allow disk space for the original data plus MCAP overhead.

Files are written with a `.partial` suffix and renamed only after finalisation.
On a caught failure or Ctrl+C, the current partial MCAP is removed and the
manifest is marked `incomplete`; already completed recordings remain available.
A forced process kill can leave partial files and a `running` manifest. Only a
`succeeded` manifest indicates a complete dataset export. There is no automatic
resume yet; retry into a new output directory.

## Code structure

- `neuracore/exporter/cli.py`: arguments, authentication and terminal messages.
- `neuracore/exporter/export.py`: format-independent selection, progress, lifecycle
  and manifest management, plus the `DatasetExporter` abstract base class.
- `neuracore/exporter/source.py`: original recording metadata and file access.
- `neuracore/exporter/mcap.py`: MCAP dependencies, validation, filenames and writing.

The shared entry point can be called by a CLI or a future background worker:

```python
from pathlib import Path
from neuracore.exporter.export import export_dataset
from neuracore.exporter.mcap import McapExporter

export_dataset(dataset, Path("./exports/new-export"), McapExporter())
```

The original `export_dataset_mcap(dataset, output, progress=None)` entry point
remains as a convenience wrapper around the shared workflow.

A new format implements `check_dependencies`, `validate_recording`, `prepare`,
`write_recording`, `finalize` and `abort`, plus a `format_name`. Writers receive
both the dataset-scoped SDK recording and the base metadata and source reader.
Format validation runs before output creation. Each recording can produce zero,
one or multiple `ExportFile` entries; finalisation can publish shared dataset
files without a recording ID. This supports formats with dataset-wide output
without imposing MCAP's file layout or sensor-manifest validation.

The manifest tracks `completed_recording_ids` separately from published files.
An export passes through `finalising` and only becomes `succeeded` after the
writer finishes. A failure invokes `abort` and marks the export `incomplete`.
