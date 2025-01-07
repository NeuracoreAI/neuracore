# Neuracore Examples

This contains examples for using Neuracore with a simulated robot environment. You'll learn how to:
- Collect and record robot demonstrations
- Deploy trained models locally
- Visualize robot behavior

## Table of Contents
1. [Installation](#installation)
2. [Examples](#examples)
   - [Data Collection](#data-collection)
   - [Local Model Deployment](#local-model-deployment)

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
python example_data_collection_vx300s.py --record True
```
2. The script will automatically start and stop recordings for each demonstration. You can see this process happening in the [robots](https://neuracore.app/dashboard/robots) tab in the app
3. Navigate to the [data](https://neuracore.app/dashboard/data) tab in the app to see your dataset


### Local Model Deployment
The local deployment example shows how to:
- Download a trained model from Neuracore
- Deploy and run the model locally
- Visualize the model's performance


For local model deployment, you'll need additional packages:
```bash
pip install "neuracore[local_endpoint]"
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
