# Neuracore Examples

This contains examples for using Neuracore with simulated robot environments. We provide [ALOHA](https://tonyzhaozh.github.io/aloha/) simulation environment as a manipulation focused scenario and [bigym](https://chernyadev.github.io/bigym/) environment as a humanoid focused scenario. You'll learn how to:
- Collect and record robot demonstrations
- Deploy trained models locally
- Visualize robot behavior

## Installation

**NOTE:** You will need Git LFS to run examples.

```bash
conda create -n neuracore_examples python=3.10
conda activate neuracore_examples
pip install "neuracore[examples]"
```

Make sure you have an account on [neuracore.com](https://neuracore.com).

### Installing Bi Gym
To run Bi Gym examples please install it using the following commands: 
```bash
git clone https://github.com/chernyadev/bigym.git
pip install . 
```

## Examples

### Data Collection
The data collection example demonstrates how to:
- Connect to the Neuracore platform
- Record robot demonstrations
- Visualize the robot in real-time
- Save demonstrations for future use

**NOTE:** You might need to set the `MUJOCO_GL` environment to `egl` using `export MUJOCO_GL=egl` if your RGB Camera Feed visualizations are glitchy on the Robot Data Visualiser Console.

1. Run the ALOHA example and record the demos:
```bash
python example_data_collection_vx300s.py --record --num_episodes=1
```

run the bigym example and record the demos:

```bash 
python example_data_collection_bigym.py --record --num_episodes=1
```
2. Navigate to the "Robots" tab in the Neuracore Dashboard
3. You should see a live view of your robot running!
4. The script will automatically start and stop recordings for each demonstration. You can see this process happening in the "Robots" tab in the Neuracore Dashboard
5. Navigate to the "Data" tab in the Neuracore Dashboard to see your dataset


### Launching Training
Launch training example show how to:
- Launch a training run from the UI
- Launch a training run using python API on the server.

**NOTE: Before running this example:**
- Collect a dataset following the example: [Data Collection](#data-collection)

Now that you have some data, navigate to "Data" tab in the Neuracore Dashboard to launch a training run on your newly collected data.

Alternatively, you can launch training runs from the python API:

```
python example_launch_training.py \
   --name 'My Training Job' \
   --algorithm_name 'CNNMLP' \
   --dataset_name 'Example Dataset'
```
For more available arguments run `python example_launch_training.py --help`


### Local Model Deployment
The local deployment example shows how to:
- Deploy and run a model locally
- Visualize the model's performance

**NOTE: Before running this example:**
- Collect a dataset following the example: [Data Collection](#data-collection)
- Start a training run by:
   - Go to the "Training" tab on the Neuracore Dashboard and start a training run
   - Or follow [Launching Training](#launching-training) to start a training run
- Wait for the training run to finish

For local model deployment, you'll need additional packages:
```bash
pip install "neuracore[ml]"
```


Run the local model with ALOHA example:
```bash
python example_local_endpoint.py
```
or if your running the Bi Gym example:
```bash 
python example_local_endpoint_bigym.py
```

### Server Model Deployment
The server deployment example shows how to:
- Start a model endpoint
- Visualize the model's performance using that active endpoint

**NOTE: Before running this example:**
- Collect a dataset following the example: [Data Collection](#data-collection)
- Start a training run by:
   - Go to the "Training" tab on your Neuracore Dashboard and start a training run
   - Or follow [Launching Training](#launching-training) to start a training run
- Wait for the training run to finish
- Go to the "Endpoint" tab on your Neuracore Dashboard and start an endpoint. Call it __"MyExampleEndpoint"__
- Wait for the status to be active

Once you have completed the steps above:
```bash
python example_server_endpoint.py
```

Unlike the previous example ([Local Model Deployment](#local-model-deployment)), this endpoint runs on our servers. 


### Retrieve and Visualize Dataset
This example shows you how to:
- Stream data from neuracore to your python application (for saving or training)
- Pull a dataset from Neuracore and **synchronize** joint positions and camera streams at a chosen frequency
- **Play back** a single episode locally

Run it to replay the first episode of the ASU Table Top shared dataset:

```bash
python example_retrieve_dataset.py
```

If you want to just view your data, then the best way is via the [web interface](https://www.neuracore.com/dashboard/datasets).


### Edit Dataset metadata

This example shows you how to:
- Edit Dataset metadata:
  - Its name
  - Description
  - Tags 
- Edit Recording metadata
  - Its Notes
  - Its Status e.g. flagged

```bash
python example_flag_data.py
```

### Work with your own robot
If you want to get your own robot working with neuracore, please refer to and also [tutorial](../docs/tutorial.md) and use the example files as a reference.
