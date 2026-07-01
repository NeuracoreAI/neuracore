"""Refresh the public getting-started Colab demo for a new release.

Retrain the demo policy against the installed neuracore version, download the
model.nc.zip, upload it to the public bucket, and bump the notebook's pinned
version. Each step can be skipped.

    python update_google_colab.py --bucket public-bucket-us-central1
    python update_google_colab.py --skip-upload --skip-patch --model-output model.nc.zip
    python update_google_colab.py --skip-train --skip-upload --version 13.3.0
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

DEFAULT_NOTEBOOK = EXAMPLES_DIR / "getting_started_with_neuracore.ipynb"

ROBOT_NAME = "Mujoco UnitreeH1 Example"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}

# Object path of the per-release model within the public bucket.
MODEL_OBJECT = "colab/bigym_pretrained_model.nc.zip"

VERSION_PIN = re.compile(r"(neuracore\[[^\]]*\]==)\d+\.\d+\.\d+")


# --------------------------------------------------------------------------- #
# Train + download
# --------------------------------------------------------------------------- #
def collect_dataset(dataset_name: str, num_episodes: int) -> None:
    """Record bigym ReachTarget expert demonstrations into a new dataset."""
    import bigym
    from bigym_utils.utils import FREQUENCY, make_env
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata
    from example_data_collection_bigym import run_episode

    import neuracore as nc
    from neuracore.data_daemon.lifecycle.daemon_os_control import ensure_daemon_running

    mjcf_path = Path(bigym.__file__).parent / "envs" / "xmls" / "h1" / "h1.xml"
    nc.connect_robot(robot_name=ROBOT_NAME, mjcf_path=str(mjcf_path), overwrite=True)
    nc.create_dataset(
        name=dataset_name,
        description="Data collection on the Bigym simulation environments",
    )
    ensure_daemon_running(timeout_s=30)
    env = make_env()
    metadata = Metadata.from_env(env)
    demos = DemoStore().get_demos(metadata, amount=num_episodes, frequency=FREQUENCY)
    try:
        for episode_idx in range(num_episodes):
            run_episode(
                episode_idx=episode_idx,
                record=True,
                demo=demos[episode_idx],
                env=env,
                render=False,
            )
    finally:
        env.close()


def launch_training(dataset_name: str, job_name: str, epochs: int) -> str:
    """Start the cloud CNNMLP training run and return its job id."""
    from bigym_utils.utils import FREQUENCY, JOINT_ACTUATORS, JOINT_NAMES
    from neuracore_types import CrossEmbodimentDescription, DataType

    import neuracore as nc

    dataset = nc.get_dataset(dataset_name)
    robot_id = dataset.robot_ids[0]
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        robot_id: {
            DataType.JOINT_POSITIONS: {i: name for i, name in enumerate(JOINT_NAMES)},
            DataType.RGB_IMAGES: {0: "head"},
        }
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        robot_id: {
            DataType.JOINT_TARGET_POSITIONS: {
                i: name for i, name in enumerate(JOINT_ACTUATORS)
            },
        }
    }
    job_data = nc.start_training_run(
        name=job_name,
        dataset_name=dataset_name,
        algorithm_name="CNNMLP",
        frequency=FREQUENCY,
        num_gpus=1,
        gpu_type="NVIDIA_TESLA_T4",
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        algorithm_config={
            "batch_size": "auto",
            "epochs": epochs,
            "output_prediction_horizon": 10,
        },
    )
    return job_data["id"]


def wait_for_completion(job_id: str, timeout_minutes: int, poll_seconds: int) -> str:
    """Block until the training job reaches a terminal state and return it."""
    import time

    import neuracore as nc

    deadline = time.time() + timeout_minutes * 60
    while True:
        status = nc.get_training_job_status(job_id=job_id)
        print(f"Training job {job_id}: {status}")
        if status in TERMINAL_STATES:
            return status
        if time.time() >= deadline:
            raise TimeoutError(
                f"Training job {job_id} did not finish within "
                f"{timeout_minutes} minutes"
            )
        time.sleep(poll_seconds)


def train_demo_model(
    dataset_name: str,
    job_name: str,
    num_episodes: int,
    epochs: int,
    output: Path,
    timeout_minutes: int,
    poll_seconds: int,
) -> None:
    """Collect data, train in the cloud, and save the trained model.nc.zip."""
    import neuracore as nc
    from neuracore.core.config.get_current_org import get_current_org
    from neuracore.core.endpoint import _download_model

    nc.login()
    collect_dataset(dataset_name, num_episodes)
    job_id = launch_training(dataset_name, job_name, epochs)
    status = wait_for_completion(job_id, timeout_minutes, poll_seconds)
    if status != "COMPLETED":
        raise RuntimeError(f"Training job {job_id} ended with status {status}")
    model_path = _download_model(job_id, get_current_org())
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, output)
    print(f"Demo model saved to {output}")


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def upload_model(model_path: Path, bucket: str) -> None:
    """Upload the trained model to gs://<bucket>/<MODEL_OBJECT>."""
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob(MODEL_OBJECT)
    blob.upload_from_filename(str(model_path))
    print(f"Uploaded {model_path} to gs://{bucket}/{MODEL_OBJECT}")


# --------------------------------------------------------------------------- #
# Notebook patch
# --------------------------------------------------------------------------- #
def patch_notebook(path: Path, version: str, output: Path) -> None:
    """Bump the pinned neuracore version in the notebook.

    Rewrite the raw file text so cells other than the version pin stay unchanged.
    """
    patched = VERSION_PIN.sub(rf"\g<1>{version}", path.read_text())
    json.loads(patched)
    output.write_text(patched)
    print(f"Patched notebook to {output}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def resolve_version(explicit: str | None) -> str:
    """Return the version to pin, defaulting to the installed neuracore."""
    if explicit:
        return explicit
    import neuracore

    return neuracore.__version__


def main(args: argparse.Namespace) -> None:
    """Run the requested refresh steps in order."""
    if not args.skip_train:
        train_demo_model(
            dataset_name=args.dataset_name,
            job_name=args.job_name,
            num_episodes=args.num_episodes,
            epochs=args.epochs,
            output=args.model_output,
            timeout_minutes=args.timeout_minutes,
            poll_seconds=args.poll_seconds,
        )

    if not args.skip_upload:
        if not args.bucket:
            raise SystemExit("--bucket is required unless --skip-upload is set")
        upload_model(args.model_output, args.bucket)

    if not args.skip_patch:
        version = resolve_version(args.version)
        out = args.notebook_output or args.notebook
        patch_notebook(args.notebook, version, out)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("--bucket", help="Public GCS bucket to upload the model to.")
    parser.add_argument(
        "--version",
        help="neuracore version to pin (defaults to the installed version).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip GCS upload.")
    parser.add_argument(
        "--skip-patch", action="store_true", help="Skip the notebook rewrite."
    )

    train = parser.add_argument_group("training")
    train.add_argument("--dataset-name", default="Getting started Bigym Example")
    train.add_argument("--job-name", default="Bigym example")
    train.add_argument("--num-episodes", type=int, default=10)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument(
        "--model-output", type=Path, default=Path("bigym_pretrained_model.nc.zip")
    )
    train.add_argument("--timeout-minutes", type=int, default=120)
    train.add_argument("--poll-seconds", type=int, default=30)

    notebook = parser.add_argument_group("notebook")
    notebook.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    notebook.add_argument(
        "--notebook-output", type=Path, default=None, help="Defaults to in-place."
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
