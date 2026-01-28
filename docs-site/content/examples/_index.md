---
title: Examples
---

This section contains examples for using Neuracore with simulated robot environments.

We provide:

- [ALOHA](https://tonyzhaozh.github.io/aloha/) as a manipulation-focused scenario
- [Bi Gym](https://chernyadev.github.io/bigym/) as a humanoid-focused scenario

You'll learn how to:

- Collect and record robot demonstrations
- Deploy trained models locally
- Visualize robot behavior

## Installation

{{< callout type="info" >}}
You will need Git LFS to run examples.
{{< /callout >}}

```bash
conda create -n neuracore_examples python=3.10
conda activate neuracore_examples
pip install "neuracore[examples]"
```

Make sure you have an account on [neuracore.com](https://neuracore.com).

### Installing Bi Gym

To run Bi Gym examples, install it using:

```bash
git clone https://github.com/chernyadev/bigym.git
pip install .
```

## Example Workflows

{{< cards >}}
  {{< card link="/examples/#data-collection" title="Data Collection" icon="play" subtitle="Record robot demonstrations and stream to Neuracore" >}}
  {{< card link="/examples/#launching-training" title="Launching Training" icon="academic-cap" subtitle="Start training runs from UI or Python API" >}}
  {{< card link="/examples/#local-model-deployment" title="Local Model Deployment" icon="chip" subtitle="Deploy and run models locally" >}}
  {{< card link="/examples/#server-model-deployment" title="Server Model Deployment" icon="cloud" subtitle="Deploy endpoints on Neuracore servers" >}}
  {{< card link="/examples/#view-dataset" title="View Dataset" icon="eye" subtitle="Stream and inspect datasets programmatically" >}}
{{< /cards >}}

## Data Collection

The data collection example demonstrates how to:

- Connect to the Neuracore platform
- Record robot demonstrations
- Visualize the robot in real-time
- Save demonstrations for future use

{{< callout type="info" >}}
You might need to set `MUJOCO_GL=egl` if RGB camera feed visualizations are glitchy in the Robot Data Visualiser Console.
{{< /callout >}}

1. Run the ALOHA example and record demos:

```bash
python example_data_collection_vx300s.py --record --num_episodes=1
```

2. Run the Bi Gym example and record demos:

```bash
python example_data_collection_bigym.py --record --num_episodes=1
```

3. Navigate to the **Robots** tab in the Neuracore Dashboard to see a live view.
4. The script will automatically start and stop recordings for each demonstration.
5. Navigate to the **Data** tab in the Neuracore Dashboard to see your dataset.

## Launching Training

The launching training example shows how to:

- Launch a training run from the UI
- Launch a training run using the Python API on the server

{{< callout type="warning" >}}
Before running this example, collect a dataset following the **Data Collection** example.
{{< /callout >}}

Once you have data:

- Go to the **Data** tab in the Neuracore Dashboard to launch a training run on your newly collected data, or
- Launch training runs from the Python API:

```bash
python example_launch_training.py \
   --name 'My Training Job' \
   --algorithm_name 'CNNMLP' \
   --dataset_name 'Example Dataset'
```

For more available arguments:

```bash
python example_launch_training.py --help
```

## Local Model Deployment

The local deployment example shows how to:

- Deploy and run a model locally
- Visualize the model's performance

{{< callout type="warning" >}}
Before running this example:

- Collect a dataset following **Data Collection**
- Start a training run (from the **Training** tab or via **Launching Training**)
- Wait for the training run to finish
{{< /callout >}}

For local model deployment, you'll need additional packages:

```bash
pip install "neuracore[ml]"
```

Run the local model with the ALOHA example:

```bash
python example_local_endpoint.py
```

Or for the Bi Gym example:

```bash
python example_local_endpoint_bigym.py
```

## Server Model Deployment

The server deployment example shows how to:

- Start a model endpoint
- Visualize the model's performance using that active endpoint

{{< callout type="warning" >}}
Before running this example:

- Collect a dataset following **Data Collection**
- Start a training run and wait for it to finish
- Go to the **Endpoint** tab on your Neuracore Dashboard and start an endpoint named **"MyExampleEndpoint"**
- Wait for the status to be **Active**
{{< /callout >}}

Once ready:

```bash
python example_server_endpoint.py
```

Unlike the local deployment example, this endpoint runs on Neuracore servers.

## View Dataset

This example shows you how to:

- Stream data from Neuracore to your Python application (for saving or training)

```bash
python example_view_dataset.py
```

If you want to just view your data, the best way is via the [web interface](https://www.neuracore.com/dashboard/datasets).

## Work With Your Own Robot

If you want to get your own robot working with Neuracore, refer to the [Getting Started](/docs/getting-started/) tutorial and use the example files as a reference.

