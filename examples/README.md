# Neuracore Examples

This contains examples for using Neuracore with a simulated robot environment. You'll learn how to:
- Collect and record robot demonstrations
- Deploy trained models locally
- Visualize robot behavior

## Installation

```bash
conda create -n neuracore_examples python=3.10
conda activate neuracore_examples
pip install "neuracore[examples]"
```

Make sure you have an account on [neuracore.app](https://neuracore.app).

## Examples

### Data Collection
The data collection example demonstrates how to:
- Connect to the Neuracore platform
- Record robot demonstrations
- Visualize the robot in real-time
- Save demonstrations for future use

1. Run the example:
```bash
python example_data_collection_vx300s.py
```
2. Navigate to the [robots](https://neuracore.app/dashboard/robots) tab in the app
3. You should see a live view of your robot running!

To record demonstrations:
1. Run with the record flag:
```bash
python example_data_collection_vx300s.py --record
```
2. The script will automatically start and stop recordings for each demonstration. You can see this process happening in the [robots](https://neuracore.app/dashboard/robots) tab in the app
3. Navigate to the [data](https://neuracore.app/dashboard/data) tab in the app to see your dataset


### Launching Training
Launch training example show how to:
- Launch a training run from the UI
- Launch a training run using python API on the server.

**NOTE: Before running this example:**
- Collect a dataset following the example: [Data Collection](#data-collection)

Now that you have some data, navigate to [https://neuracore.app/dashboard/training](https://neuracore.app/dashboard/training) to launch a training run on your newly collected data.

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
   - Go to your [training dashboard ](https://www.neuracore.app/dashboard/training) and start a training run
   - Or follow [Launching Training](#launching-training) to start a training run
- Wait for the training run to finish

For local model deployment, you'll need additional packages:
```bash
pip install "neuracore[ml]"
```

Local model deployment also requires a Java JRE.

<details>
<summary>Install Java JRE</summary>

Mac:
```bash
brew install temurin java
```

Linux:
```bash
sudo apt install default-jre
```
</details>


Run the local model:
```bash
python example_local_endpoint.py
```


### Server Model Deployment
The server deployment example shows how to:
- Start a model endpoint
- Visualize the model's performance using that active endpoint

**NOTE: Before running this example:**
- Collect a dataset following the example: [Data Collection](#data-collection)
- Start a training run by:
   - Go to your [training dashboard ](https://www.neuracore.app/dashboard/training) and start a training run
   - Or follow [Launching Training](#launching-training) to start a training run
- Wait for the training run to finish
- Go to your [endpoint dashboard ](https://www.neuracore.app/dashboard/endpoints) and start an endpoint. Call it __"MyExampleEndpoint"__
- Wait for the status to be active

One you have completed the steps above:
```bash
python example_server_endpoint.py
```

Unlike the previous example ([Local Model Deployment](#local-model-deployment)), this endpoint runs on our servers. 


### View Dataset
This example shows you how to:
- Stream data from neuracore to your python application (for saving or training)

```bash
python example_view_dataset.py
```

If you want to just view your data, then the best way is via the [web interface](https://www.neuracore.app/dashboard/datasets).
