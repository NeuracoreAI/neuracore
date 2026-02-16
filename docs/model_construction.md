# Model Construction 

In Neuracore we auto-construct models input and output dimensions based on what the model is trained on.

This document aim to help how we do this. 

## Starting a Training Job

When training a model on Neuracore you need to provide: 
- Input Robot Data Spec
- Output Robot Data Spec

There are other things but this is the only parameters that determine the input and output dimensions of the model. 

### What is a Robot Data Spec? 
```python
RobotDataSpec:  dict[str, DataSpec]
DataSpec: dict[DataType, list[str]]
```
In a robot data spec, the dictionary keys are strings corresponding to robot names.  
If the dataset contains data from a single robot, the dictionary will have exactly one key.  
If the dataset contains data from multiple robots, the `RobotDataSpec` will include multiple keys—one for each robot represented in the dataset.

A `DataSpec` is a dictionary where the keys are `DataType` values.  
Each `DataType` corresponds to a Neuracore log type, such as `joint_positions`, `rgb_images`, or `gripper_open_amounts`.

Each `DataType` maps to a list of strings that specify the keys used when the data was logged.  
For example, for image data, the list might contain camera identifiers such as `WristCamera` and `HeadCamera`.  
For joint data, the list would typically contain joint or motor names such as `motor_1`, `motor_2`, and so on.

### Example
In the example below we have a RobotDataSpec where I am training a model on two robots, that have different joint lengths and have different cameras attached. 

Note: For simplicity in this explanation we will set the input and output to be the same, normally you would want the output to be the target_joint positions or different depending on your application. 
```python

input_robot_data_spec: RobotDataSpec = {
	"snake": {
		JOINTS:[
			spine_1,
			spine_2,
			spine_3,
			head
		],
		RGB_IMAGES:[
			left_eye, 
			right_eye
		]
	},
	"cyclops": {
		JOINTS:[
			leg_1,
			leg_2,
			arm_1,
			arm_2,
			head
		]
		RGB_IMAGES:[
			eye
		]
	}
}

output_robot_data_spec: RobotDataSpec = input_robot_data_spec
```

## Data Padding

Different robots can have different numbers of joints.  
However, when we train a model, all data needs to have the same shape.

In this example, one robot has fewer joints than the other. To fix this, we **add extra zeros** to the robot with fewer joints so that both robots end up with the same number of joint values.

Looking at the table:

- Each **row** is one recording.
- Each **column** represents one joint position.
- Some robots do not have all of these joints, so their missing joints are filled with **0**.

| Recording ID | 1       | 2       | 3       | 4     | 5     |
| ------------ | ------- | ------- | ------- | ----- | ----- |
| 1            | spine_1 | spine_2 | spine_3 | head  | **0** |
| 2            | leg_1   | leg_2   | arm_1   | arm_2 | head  |
| 3            | spine_1 | spine_2 | spine_3 | head  | **0** |
| 4            | spine_1 | spine_2 | spine_3 | head  | **0** |
| 5            | spine_1 | spine_2 | spine_3 | head  | **0** |
| 6            | leg_1   | leg_2   | arm_1   | arm_2 | head  |
| 7            | leg_1   | leg_2   | arm_1   | arm_2 | head  |
| 8            | leg_1   | leg_2   | arm_1   | arm_2 | head  |

This way, every recording has the same number of joint values, which makes it possible to combine all recordings into a single dataset and train a model without issues.

Similarly this is done for all other DataTypes such as RGB_Images in this case. 

Now we have essentially forced each robots joint lengths to be of length 5 in this case. 
## Training a Neuracore Model 
Now that we have out padded data ensuring every data point no matter which recording/robot it comes from is the same length, we construct a model like the diagram below: 
![Model input diagram](assets/model_input.png)


Note: that in this case since the output is the same as the input the model heads would of identical shape. 

## Running Inference
When running inference on a model you need to define the model input and output order which is of type DataSpec. This should be directly referencing the robot you are trying to run inference on. 

So in the case you train the model on data from snake and cyclopes and you want to run inference on cyclopes. Your model input spec can be: 
 
```python
model_input_data_spec: DataSpec = {
	JOINTS:[
		leg_1,
		leg_2,
		arm_1,
		arm_2,
		head
	]
	RGB_IMAGES:[
		eye
	]
}

# Note: in this case the model_input_data_spec is the same as input_robot_data_spec["cyclops"]
```

## Input/Output order matters.  
The order of elements in the joint list defines the exact order in which values are passed into the model.

For example, suppose we introduce a new robot at inference time called **“bat”**, with joints:

`[wing_1, wing_2, head]`

The model input vector will follow this order exactly:

`[wing_1, wing_2, head, 0, 0, …]`


![Model ordering diagram](assets/model_ordering.png)
Because the bat has fewer joints than the maximum joint count used during training, the remaining dimensions are **zero-padded** to match the expected input size.  

This ensures the input/output tensor has a consistent shape.

## Common Gotcha's

### Alignment of Values
In the example above, the joint name **`head`** exists on all three robots. However, because each robot defines a different joint ordering, the `head` value may be placed at **different indices** in the model input vector.

This means that, although the joint has the same name, it is interpreted by the model as **different input features** across robots.

For this reason, it is critical during both training and inference to **align semantically similar joints to the same input indices** wherever possible. Consistent alignment ensures that each dimension of the input tensor retains a stable semantic meaning, allowing the model to generalize correctly across different robot morphologies.
### More Complex Robots
In the examples above, the model was trained using **two RGB camera inputs**. This fixes both the **number** and **structure** of visual inputs the model expects at inference time.

If a new robot—such as **`spider`**—is introduced with **eight RGB cameras**, the model cannot directly consume all eight streams. The input interface only supports the two camera channels seen during training.

As a result, the user must **select two RGB cameras** to pass into the model at inference time. The remaining camera feeds are ignored. This constraint mirrors the joint-padding behavior: while smaller input sets can be padded, larger input sets cannot be expanded beyond the model’s trained input dimensionality.
