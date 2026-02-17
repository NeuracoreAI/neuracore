# Model Construction

In robotics, the primary bottleneck is rarely model optimization. It is representation alignment across embodiments.

A Franka Emika Panda and a KUKA arm differ in joint count, kinematic structure, sensor topology, and naming conventions. Cameras may vary in count and placement, grippers may expose different control signals, and joints may not have one-to-one semantic matches.

Without a principled construction layer, each new robot requires:
- Custom tensor wiring
- Ad-hoc preprocessing pipelines
- Model-specific reshaping logic
- Fragile, robot-dependent inference code

This does not scale. It creates brittle systems where architecture assumptions leak into application logic.

Neuracore eliminates per-robot model wiring by auto-constructing model input and output dimensions from the training data specification.

Instead of hard-coding tensor shapes into model definitions, Neuracore derives them from a declarative robot data spec. That spec becomes the source of truth for:
- Input dimensionality
- Output dimensionality
- Cross-embodiment alignment
- Padding and masking behavior

At a high level, this enables easy cross-embodiment training: one model pipeline that can ingest data from heterogeneous robot morphologies without bespoke architectural changes.

This document explains how model construction works and why this spec-driven approach is required for stable multi-robot training and inference.

> [!IMPORTANT]
> **Model shape is spec-driven, not robot-driven.** The training spec defines the fixed tensor layout used for both training and inference.

## Starting a Training Job

When launching a training job in Neuracore, you define:
- Input Robot Data Spec
- Output Robot Data Spec

Other parameters exist (optimizer, architecture, dataset, etc.), but these two specs determine the model input and output dimensionality.

The model shape is not hard-coded. It is constructed deterministically from the specification.

> [!NOTE]
> **Input/Output Robot Data Specs are the dimensionality contract** for the model.

## What Is a Robot Data Spec?

```python
EmbodimentDescription = dict[DataType, dict[int, str]]
CrossEmbodimentDescription = dict[str, EmbodimentDescription]
```

We use explicit terminology to clarify intent.

`EmbodimentDescription` describes one robot's data structure.

`CrossEmbodimentDescription` maps robot names to their embodiment descriptions.

This distinction matters because model construction operates across embodiments, not just within one robot.

> [!TIP]
> Keep this distinction clear:
> - `EmbodimentDescription` = one robot
> - `CrossEmbodimentDescription` = many robots

### EmbodimentDescription

An `EmbodimentDescription` defines what signals exist for one robot and where each signal lives in index space.

Each `DataType` (for example `JOINTS`, `RGB_IMAGES`, `GRIPPER_OPEN_AMOUNT`) maps to:

`dict[int, str]`

Where:
- `int`: deterministic feature index
- `str`: logged key (joint name, camera name, etc.)

Conceptual example:

```python
{
    JOINTS: {
        0: "panda_joint_1",
        1: "panda_joint_2",
        2: "panda_joint_3",
        3: "panda_joint_4",
        5: "franka_gripper",
    },
    RGB_IMAGES: {
        0: "wrist_camera",
        1: "overhead_camera"
    }
}
```

Why explicit indices:
- **Index defines tensor position.**
- **Tensor position defines semantic meaning.**
- If two robots place an equivalent joint at index 2, the model can treat that feature consistently.
- Allows for aligning similar joints across robots.

### CrossEmbodimentDescription

`CrossEmbodimentDescription = dict[str, EmbodimentDescription]`

This maps:
- Robot name -> embodiment structure

Example:

```python
{
    "franka": { ... },
    "kuka": { ... }
}
```

Why this is necessary:
- Multi-robot training needs a shared tensor space.
- Robots differ in joint counts, camera counts, naming conventions, and morphology.

The `CrossEmbodimentDescription` lets Neuracore:
- Compute maximum dimensionality per `DataType`
- Define canonical index ranges
- Apply deterministic zero-padding
- Guarantee fixed input/output tensor shapes

Without this cross-embodiment layer, model dimensions are undefined.

> [!IMPORTANT]
> A shared cross-embodiment index space is what allows one model to train across heterogeneous robots.

## Example

The example below uses the explicit indexed structure described above.
Each robot is an `EmbodimentDescription`, and the full mapping is a `CrossEmbodimentDescription`.

For clarity, input and output specs are identical here. In practice, output specs often represent target data (for example target joints, torques, or end-effector pose) and may differ from input specs.

```python
input_cross_embodiment_spec: CrossEmbodimentDescription = {
    "franka": {
        JOINTS: {
            0: "panda_joint_1",
            1: "panda_joint_2",
            2: "panda_joint_3",
            3: "panda_joint_4",
            5: "franka_gripper",
        },
        RGB_IMAGES: {
            0: "franka_wrist_camera",
            1: "franka_overhead_camera",
        },
    },
    "kuka": {
        JOINTS: {
            0: "iiwa_joint_1",
            1: "iiwa_joint_2",
            2: "iiwa_joint_3",
            3: "iiwa_joint_4",
            4: "iiwa_joint_5",
            5: "kuka_gripper",
        },
        RGB_IMAGES: {
            0: "kuka_front_camera",
        },
    },
}

output_cross_embodiment_spec: CrossEmbodimentDescription = input_cross_embodiment_spec
```
In this example, the Franka gripper is assigned to **index 5** in the model’s joint vector. However, the Franka robot does not have a joint corresponding to **index 4**, meaning that position in the vector is unused for this robot and therefore padded (typically with zeros).

We did this to keep a consistent joint vector size and ordering across all robots used during training.

The model expects a fixed-length input and output vector. That structure is defined by the global training specification, not by any single robot. Some robots may have more joints, some fewer — but the model architecture cannot change shape depending on which robot is being used.

In this example:

- Index 4 exists in the global joint layout because another robot in the dataset uses it.
- The Franka does not have a joint that maps to index 4.
- Therefore, we leave that index empty and pad it with zero.

This ensures:

- The model always receives the same input dimension.
- Joint meanings stay aligned across robots.

Padding is simply a structural requirement to preserve a shared representation across multiple robots.

> [!NOTE]
> **Empty index slots are expected** when a robot does not populate every position in the global layout.

## Data Padding

Different robots can have different numbers of joints, but training requires a **consistent tensor shape**.
In this example, the gripper value stays at its globally assigned index, even when some intermediate indices are empty for a given robot.

If one robot has fewer joints, Neuracore fills missing dimensions with `0` so all records align to the same width.

> [!TIP]
> Think of zero-padding as a **layout-preserving operation**, not a semantic value.

Reading the table:
- Each row is one recording
- Each column is one joint-position slot
- Missing joints are zero-padded

| Recording ID | 0             | 1             | 2             | 3             | 4              | 5             |
| ------------ | ------------- | ------------- | ------------- | ------------- | -------------- | ------------- |
| 1            | panda_joint_1 | panda_joint_2 | panda_joint_3 | panda_joint_4 | **0**          | franka_gripper |
| 2            | iiwa_joint_1  | iiwa_joint_2  | iiwa_joint_3  | iiwa_joint_4  | iiwa_joint_5   | kuka_gripper  |
| 3            | panda_joint_1 | panda_joint_2 | panda_joint_3 | panda_joint_4 | **0**          | franka_gripper |
| 4            | panda_joint_1 | panda_joint_2 | panda_joint_3 | panda_joint_4 | **0**          | franka_gripper |
| 5            | panda_joint_1 | panda_joint_2 | panda_joint_3 | panda_joint_4 | **0**          | franka_gripper |
| 6            | iiwa_joint_1  | iiwa_joint_2  | iiwa_joint_3  | iiwa_joint_4  | iiwa_joint_5   | kuka_gripper  |
| 7            | iiwa_joint_1  | iiwa_joint_2  | iiwa_joint_3  | iiwa_joint_4  | iiwa_joint_5   | kuka_gripper  |
| 8            | iiwa_joint_1  | iiwa_joint_2  | iiwa_joint_3  | iiwa_joint_4  | iiwa_joint_5   | kuka_gripper  |

The same strategy applies to other data types, including image channels.

When training on cross-embodiment data, it is important to match joints by meaning, not just by name or position in a list.

Joints that perform the same role — for example base rotation, elbow bend, or gripper open/close — should be placed in the same index of the model’s input and output vectors across all robots.

If this is not done:

- The same model index may represent different motions on different robots.
- The model becomes confused during training.
- Generalisation between robots becomes much weaker.

For cross-robot training to work well, each dimension in the model must represent a consistent physical meaning across all embodiments.

> [!WARNING]
> Misaligned indices create **label noise across robots** and can significantly reduce generalization.

## Training a Neuracore Model

Once padding enforces consistent shapes across recordings and robots, the model can be constructed as shown below.

![Model input diagram](assets/model_input.png)

In this example, input and output specs are the same, so model heads have identical shape.

## Running Inference

For inference, define model input/output order using an `EmbodimentDescription` that directly matches the robot being controlled.

> [!IMPORTANT]
> Inference must use the **same global positional conventions** learned at training time.

For example, if you trained on `franka` and `kuka` data and want inference on `kuka`:

```python
model_input_data_spec: EmbodimentDescription = {
    JOINTS: {
        0: "iiwa_joint_1",
        1: "iiwa_joint_2",
        2: "iiwa_joint_3",
        3: "iiwa_joint_4",
        4: "iiwa_joint_5",
        5: "kuka_gripper",
    },
    RGB_IMAGES: {
        0: "kuka_front_camera",
    },
}

# model_input_data_spec follows the JOINTS/RGB_IMAGES order from input_cross_embodiment_spec["kuka"]
```

For inference on `franka`, keep the same shared index space and skip index `4` (it will be zero-padded):

```python
model_input_data_spec: EmbodimentDescription = {
    JOINTS: {
        0: "panda_joint_1",
        1: "panda_joint_2",
        2: "panda_joint_3",
        3: "panda_joint_4",
        5: "franka_gripper",
    },
    RGB_IMAGES: {
        0: "franka_wrist_camera",
        1: "franka_overhead_camera",
    },
}
```

### Inference on a New Robot

Suppose a new inference-time robot `ur5` has:

`[ur5_shoulder_pan, ur5_shoulder_lift, ur5_gripper]`

Since this robot only has two joints plus a gripper, we should semantically align those signals with the trained index layout and keep the gripper at the same index used during training:

| Robot  | 0                 | 1                  | 2             | 3             | 4            | 5               |
| ------ | ----------------- | ------------------ | ------------- | ------------- | ------------ | --------------- |
| franka | panda_joint_1     | panda_joint_2      | panda_joint_3 | panda_joint_4 | **0**        | franka_gripper  |
| kuka   | iiwa_joint_1      | iiwa_joint_2       | iiwa_joint_3  | iiwa_joint_4  | iiwa_joint_5 | kuka_gripper    |
| ur5    | ur5_shoulder_pan  | ur5_shoulder_lift  | **0**         | **0**         | **0**        | ur5_gripper     |

```python
ur5_embodiment_spec: EmbodimentDescription = {
    JOINTS: {
        0: "ur5_shoulder_pan",
        1: "ur5_shoulder_lift",
        5: "ur5_gripper",
    },
    RGB_IMAGES: {
        1: "ur5_overhead_camera",
    },
}
```

Likewise, RGB inputs must follow the same positional convention used during training.

For example, if during training the model always received:

`[wrist_camera, overhead_camera]`

where:
- Index 0 = wrist camera
- Index 1 = overhead camera

then that ordering becomes part of the model’s learned structure.

If, at inference time, a new robot only has a single RGB camera mounted overhead, we must place that image in index 1, not index 0, because the model has learned that index 1 corresponds to the overhead viewpoint.

> [!WARNING]
> Do not remap camera indices opportunistically at inference time. Positional drift changes feature meaning.


![Model ordering diagram](assets/model_ordering.png)



## More Complex Robots

If the model was trained with two RGB inputs, inference expects that same camera-channel structure.

If a new robot provides eight RGB cameras, the model cannot directly consume all eight streams. You must select two cameras to feed the model, and ignore the rest.

This mirrors padding behavior: smaller inputs can be padded, but larger inputs cannot exceed trained dimensionality.

> [!NOTE]
> **You can pad missing channels, but you cannot exceed trained dimensionality** without changing the model architecture and retraining.
