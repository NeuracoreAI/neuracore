# Neuracore Data Streaming Test

Stream saved NPY/video teleop data into Neuracore, as if recorded live.

## Install

```bash
conda env create -f environment.yaml
conda activate neuracore-test-streaming
```

Run from the `test_streaming/` directory so `common` imports work.

## Usage

### Single episode (`stream_npy_video_to_neuracore_single_episode.py`)

Stream one episode once to Neuracore.

```bash
python stream_npy_video_to_neuracore_single_episode.py \
  --input-dir streaming_test_teleop_data/ \
  --episode-index 0 \
  [--dataset-name MY_DATASET]
```

- `--input-dir` — directory with `episode_XXXX` subdirs
- `--episode-index` — which episode (0-based)
- `--dataset-name` — optional dataset name

### Loop (`stream_npy_video_to_neuracore_loop.py`)

Stream the same episode in a loop until Ctrl+C.

```bash
python stream_npy_video_to_neuracore_loop.py \
  --input-dir streaming_test_teleop_data/ \
  --episode-index 0 \
  [--dataset-name MY_DATASET] \
  [--record]
```

- Same options as single episode, plus:
- `--record` — start Neuracore recording when you run; stop when you Ctrl+C
