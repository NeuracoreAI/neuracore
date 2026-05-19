<!-- cspell:disable -->
# Design: Import `libero_10_subtasks` with `SUBTASK_LANGUAGE` support

## Goal

Upload the LeRobot dataset `KeWangRobotics/libero_10_subtasks` into Neuracore.
The dataset carries fine-grained `subtask` labels in addition to the high-level
`overall_task` instruction. `SUBTASK_LANGUAGE` already exists as a `DataType` in
`neuracore_types`, but the `neuracore` importer does not yet handle it. This work
wires `SUBTASK_LANGUAGE` through the importer and adds a config file for the
dataset.

## Dataset facts (verified from `meta/info.json`)

- LeRobot `codebase_version` `v3.0`, `fps` `10.0`, `total_episodes` `500`.
- Flat feature keys:
  - `state` — `float32`, shape `[8]` (end-effector pose + gripper).
  - `actions` — `float32`, shape `[7]` (delta EE pose + gripper command).
  - `images.agentview_rgb` — `image`, shape `[3, 256, 256]` (channels-first).
  - `images.wrist_rgb` — `image`, shape `[3, 256, 256]` (channels-first).
  - `subtask` — `string`, shape `[1]` (7 distinct values).
  - `overall_task` — `string`, shape `[1]` (4 distinct values).
- Robot: Franka Panda — same URDF as `austin_sailor` (`panda.urdf`).

## Part A — Importer wiring for `SUBTASK_LANGUAGE`

`SUBTASK_LANGUAGE` is present in `neuracore_types` (`DataType.SUBTASK_LANGUAGE`,
mapped to `LanguageData` / `LanguageDataImportConfig`). Four spots in the
`neuracore` importer must be updated. All changes are additive and
backward-compatible.

1. **`neuracore/api/logging.py` — `log_language` + `log_subtask_language`**
   Generalize `log_language` with parameter `data_type: DataType =
   DataType.LANGUAGE`, used for:
   - the stream id: `f"{data_type.value}:{name}"`
   - the `JsonDataStream(data_type=data_type, ...)` constructor
   - the `_publish_json_to_p2p(robot, str_id, data_type, language_data)` call
   `LanguageData` is the payload class for both `LANGUAGE` and `SUBTASK_LANGUAGE`
   (`DATA_TYPE_TO_NC_DATA_CLASS`); the *stream* carries the type, so no payload
   change is needed. Existing callers that omit `data_type` keep current
   behavior.

   Then add a thin public wrapper `log_subtask_language(name, language,
   robot_name=None, instance=0, timestamp=None, dry_run=False)` that simply
   delegates to `log_language(..., data_type=DataType.SUBTASK_LANGUAGE)`. This
   keeps a single implementation while giving the API a discoverable,
   self-documenting entry point for subtask logging. Export it alongside
   `log_language` (e.g. in `neuracore/__init__.py`).

2. **`neuracore/importer/core/base.py` — `_log_data`**
   Change `elif data_type == DataType.LANGUAGE:` to match both `LANGUAGE` and
   `SUBTASK_LANGUAGE`. Dispatch to `nc.log_subtask_language` when the data type
   is `SUBTASK_LANGUAGE` and `nc.log_language` otherwise.

3. **`neuracore/importer/core/base.py` — `_validate_input_data`**
   Include `SUBTASK_LANGUAGE` in the branch that calls `validate_language`.

4. **`neuracore/importer/lerobot_importer.py` — `_record_step`**
   The numpy-conversion skip currently fires only for `LANGUAGE` +
   `LanguageConfig.STRING`. Extend it so `SUBTASK_LANGUAGE` + `STRING` also skips
   `_convert_source_data` (the value is a Python string, not a tensor).

Config parsing needs no change: `data_import_config` keys are resolved through
`DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS`, which already routes
`SUBTASK_LANGUAGE` to `LanguageDataImportConfig`. The plan confirms this with a
parse test.

The RLDS importer (`rlds_tfds_importer.py`) is out of scope — this dataset is
LeRobot. The `base.py` changes benefit RLDS as well.

## Part B — Config file `neuracore/importer/config/libero_10_subtasks.yaml`

Modeled on `austin_sailor.yaml`: Franka Panda, `panda.urdf`, EE-based with IK to
derive joint positions.

- **Header**: `input_dataset_name: KeWangRobotics/libero_10_subtasks`,
  `dataset_type: LEROBOT`, `frequency: 10.0`, robot `franka_panda` with
  `urdf_path: "panda.urdf"`, `overwrite_existing: true`.
- **RGB_IMAGES**: `source: images`, `image_convention: CHANNELS_FIRST`,
  `order_of_channels: RGB`; mapping `agentview_rgb`, `wrist_rgb`.
- **END_EFFECTOR_POSES**: `source: state`, `index_range` `0:6`, `panda_hand_tcp`,
  `pose_type: POSITION_ORIENTATION`, orientation EULER `XYZ` radians.
- **JOINT_POSITIONS**: `joint_position_input_type: END_EFFECTOR` — IK from the EE
  pose using `panda.urdf`. Processed after `END_EFFECTOR_POSES`
  (`_get_ordered_import_configs` handles the ordering).
- **PARALLEL_GRIPPER_OPEN_AMOUNTS**: from the `state` gripper index, normalized.
- **VISUAL_JOINT_POSITIONS**: `visual_joint_input_type: GRIPPER` for
  `panda_finger_joint1` / `panda_finger_joint2`.
- **JOINT_TARGET_POSITIONS**: `source: actions`, `index_range` `0:6`,
  `action_type: RELATIVE`, `action_space: END_EFFECTOR`.
- **LANGUAGE**: `source: overall_task`, `language_type: STRING`,
  mapping name `instruction`.
- **SUBTASK_LANGUAGE**: `source: subtask`, `language_type: STRING`,
  mapping name `subtask`.

## Open items resolved during implementation

These are confirmed against the real dataset / config models rather than guessed
in this spec:

1. Exact layout of the 8-dim `state` — gripper at index `6` only, or `6` and `7`
   (two finger values). Determined by inspecting a real sample.
2. Encoding of the delta rotation in the 7-dim `actions` (EULER vs AXIS_ANGLE)
   and whether a `scale_position` is needed.
3. The mapping schema for IK-based `JOINT_POSITIONS` — specifically the
   `name` / `source_name` fields that reference the logged EE pose
   (`panda_hand_tcp`) — confirmed against the importer config models.

## Verification

1. Run the importer with `--dry-run` on a few episodes: streams are validated
   without uploading, surfacing any `state` / `actions` mapping errors.
2. Run a small real import (a handful of episodes) and confirm `LANGUAGE` and
   `SUBTASK_LANGUAGE` streams appear in the resulting Neuracore dataset.

## Assumptions and risks

- The installed `lerobot` package must support `codebase_version v3.0`. If not,
  the dataset must be loaded with a compatible version — flagged as a risk, not
  handled by this work.
- The Franka `panda.urdf` already used by `austin_sailor` is reused as-is.
