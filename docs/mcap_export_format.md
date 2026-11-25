# Neuracore MCAP Export Format

`neuracore export mcap` writes one [MCAP](https://mcap.dev) file per recording. Each file holds every stored trace entry of the recording as a JSON message, and the stored camera and point cloud files as attachments. Any program that reads MCAP can read the file. `examples/example_read_mcap_export.py` reads one with only the `mcap`, `av`, `imagecodecs` and `numpy` Python packages.

## Export a dataset

```bash
neuracore export mcap --dataset "My Dataset" --output ./my_dataset_mcap
```

The output directory holds one `nc_<recording name>.mcap` per recording and a `manifest.json`. The manifest names the format (`neuracore-mcap-json-v2`), the dataset, the exported recording ids and the file of each recording.

## File layout

| Record | Content |
|---|---|
| Header | Library `neuracore/<version>` |
| Metadata `neuracore.recording` | Key `json` holds a JSON object with the recording `id`, `robot_id`, `instance`, `start_time`, `end_time`, `metadata`, `data_types` and `sensor_manifest` |
| Schema, one per data type | Name `neuracore.<DATA_TYPE>.v1`, encoding `jsonschema` |
| Channel, one per sensor | Topic `/neuracore/<DATA_TYPE>/<sensor name>`, message encoding `json` |
| Message, one per trace entry | The stored trace entry as JSON |
| Attachment, one per stored camera or point cloud file | The stored file, byte for byte |

The writer compresses message chunks with zstd.

### Schemas

Each schema comes from the `neuracore_types` model of its data type (for example `JointData` for `JOINT_POSITIONS`). Fields with a default value are optional, and every schema sets `additionalProperties` to `true`, because stored entries carry extra keys such as the byte range of a depth frame.

### Channels

The sensor name in the topic is URL-encoded, so `cameras/0` becomes `/neuracore/RGB_IMAGES/cameras%2F0`. Each channel carries this metadata:

| Key | Value |
|---|---|
| `data_type` | The Neuracore data type, for example `RGB_IMAGES` |
| `sensor_name` | The sensor name as the recording stores it |
| `attachment` | The name of the attachment that holds the sensor's frames. Present on camera channels and on point cloud channels with a `trace.bin` file. |

### Messages

Each message holds one trace entry exactly as the recording stores it. Every entry has `timestamp` (Unix seconds) and usually `timestamp_us` (Unix microseconds). The message `log_time` and `publish_time` are the capture time in nanoseconds, and `sequence` is the entry's position in its trace.

### Attachments

An attachment name is the file's path inside the recording, `<DATA_TYPE>/<sensor name>/<file>`. Its `log_time` is the recording start. Videos have media type `video/mp4`, and other files have `application/octet-stream`.

## Reading each data type

### RGB images

The attachment is `lossless.mp4` (H.264, lossless RGB) when the recording has one, and otherwise `lossy.mp4` (H.264, YUV 4:2:0). Decode the video in display order: the Nth decoded frame belongs to the entry with `frame_idx` N. The exporter checks that the video holds exactly one frame per entry and that each frame time is within 2 µs of its entry's time, measured from the first frame and the first entry.

Entries also hold `width`, `height`, `intrinsics` (3×3) and `extrinsics` (4×4).

### Depth images

The attachment is `lossless.bin`, a concatenation of JPEG XL images. Bytes `offset` to `offset + length` of the attachment hold the entry's frame. The frame decodes to a uint16 image of sensor units, where 0 means no return. Multiply by the entry's `depth_scale_m` to get metres.

```python
start = entry["offset"]
units = imagecodecs.jpegxl_decode(attachment[start : start + entry["length"]])
depth_m = units.astype(np.float32) * np.float32(entry["depth_scale_m"])
```

### Point clouds

Recordings store point clouds in one of two ways, and the channel metadata says which.

A channel with no `attachment` key holds each cloud inside its entry. `points` is base64 of little-endian float16 values, shape (N, 3) as x, y, z. `rgb_points` is base64 of uint8 values, shape (N, 3) as red, green, blue.

A channel with an `attachment` key points to `trace.bin`. Bytes `offset` to `offset + length` hold one frame:

| Part | Content |
|---|---|
| 4 bytes | Little-endian uint32 length of the metadata |
| Metadata | JSON with `num_points`, `points_nbytes` and `rgb_nbytes` |
| Points | `points_nbytes` bytes of little-endian float16, shape (N, 3) |
| Colours | `rgb_nbytes` bytes of uint8, shape (N, 3), when `rgb_nbytes` is above 0 |

Entries also hold `intrinsics` (3×3) and `extrinsics` (4×4).

### Other data types

The entry holds the value directly:

| Data type | Field |
|---|---|
| `JOINT_POSITIONS`, `JOINT_VELOCITIES`, `JOINT_TORQUES`, `JOINT_TARGET_POSITIONS`, `VISUAL_JOINT_POSITIONS` | `value`, one number per joint channel |
| `END_EFFECTOR_POSES`, `POSES` | `pose`, `[x, y, z, qx, qy, qz, qw]` |
| `PARALLEL_GRIPPER_OPEN_AMOUNTS`, `PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS` | `open_amount`, 0 closed to 1 open |
| `LANGUAGE` | `text` |
| `CUSTOM_1D` | `data`, a list of numbers |
| `DONE` | `done`, a boolean |

## Import an export back into Neuracore

The Neuracore MCAP importer reads the attachments and adds each frame to its message, under `frame` for cameras and `points` and `rgb_points` for point clouds. Map those fields in the import configuration:

```yaml
RGB_IMAGES:
  mapping:
    - name: front
      source_name: /neuracore/RGB_IMAGES/front.frame
DEPTH_IMAGES:
  format:
    depth_scale_m: 0.001  # the depth_scale_m of the exported entries
  mapping:
    - name: wrist
      source_name: /neuracore/DEPTH_IMAGES/wrist.frame
POINT_CLOUDS:
  mapping:
    - name: lidar
      source_name: /neuracore/POINT_CLOUDS/lidar.points
```
