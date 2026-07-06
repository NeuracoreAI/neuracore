"""Refresh the public getting-started Colab demo for a new release.

Retrain the demo policy against the installed neuracore version, roll it out
to confirm it is functional, upload the model.nc.zip to the public bucket
under a version-tagged filename, and bump the notebook's pinned version and
model filename. Each step can be skipped.

Reuses examples/bigym_utils, example_data_collection_bigym.py, and
example_local_endpoint_bigym.py wherever possible. The Colab notebook itself
cannot import repo modules, so its cells embed the same logic inline; keep
both in sync when changing shared behavior.

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

VERSION_PIN = re.compile(r"(neuracore\[[^\]]*\]==)\d+\.\d+\.\d+")
MODEL_URL_PIN = re.compile(r"bigym_pretrained_nc_v\d+\.\d+\.\d+_model\.nc\.zip")


def model_filename(version: str) -> str:
    """Filename of the per-release pretrained model, tagged with its version."""
    return f"bigym_pretrained_nc_v{version}_model.nc.zip"


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
# Validate
# --------------------------------------------------------------------------- #
def validate_model(
    model_path: Path,
    num_rollouts: int,
    min_success_rate: float,
) -> None:
    """Roll out the trained model and raise if it fails too often to be usable."""
    from bigym_utils.utils import make_env
    from example_local_endpoint_bigym import run_rollout

    import neuracore as nc

    policy = nc.policy(model_file=str(model_path), robot_name=ROBOT_NAME)
    env = make_env()
    success_count = 0
    try:
        for episode_idx in range(num_rollouts):
            succeeded = run_rollout(
                env=env, policy=policy, num_steps=100, sleep_per_step=0.0
            )
            if succeeded:
                success_count += 1
            print(f"Validation rollout {episode_idx}: succeeded={succeeded}")
    finally:
        env.close()
        policy.disconnect()

    success_rate = success_count / num_rollouts
    print(
        f"Rollout validation: {success_count}/{num_rollouts} succeeded "
        f"({success_rate:.0%})"
    )
    if success_rate < min_success_rate:
        raise RuntimeError(
            f"Trained model at {model_path} only succeeded {success_rate:.0%} of "
            f"{num_rollouts} validation rollouts, below the required "
            f"{min_success_rate:.0%}"
        )


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def upload_model(model_path: Path, bucket: str, version: str) -> None:
    """Upload the trained model to gs://<bucket>/colab/<model_filename>."""
    from google.cloud import storage

    object_path = f"colab/{model_filename(version)}"
    blob = storage.Client().bucket(bucket).blob(object_path)
    blob.upload_from_filename(str(model_path))
    print(f"Uploaded {model_path} to gs://{bucket}/{object_path}")


# --------------------------------------------------------------------------- #
# Notebook patch
# --------------------------------------------------------------------------- #
def patch_notebook(path: Path, version: str, output: Path) -> None:
    """Bump the pinned neuracore version and pretrained model filename.

    Rewrite the raw file text so cells other than those two pins stay unchanged.
    """
    patched = VERSION_PIN.sub(rf"\g<1>{version}", path.read_text())
    patched = MODEL_URL_PIN.sub(model_filename(version), patched)
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
    version = resolve_version(args.version)

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
        if not args.skip_validate:
            validate_model(
                model_path=args.model_output,
                num_rollouts=args.num_rollouts,
                min_success_rate=args.min_success_rate,
            )

    if not args.skip_upload:
        if not args.bucket:
            raise SystemExit("--bucket is required unless --skip-upload is set")
        upload_model(args.model_output, args.bucket, version)

    if not args.skip_patch:
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
    parser.add_argument(
        "--skip-validate", action="store_true", help="Skip rollout validation."
    )
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

    validate = parser.add_argument_group("validation")
    validate.add_argument(
        "--num-rollouts", type=int, default=10, help="Rollouts to validate with."
    )
    validate.add_argument(
        "--min-success-rate",
        type=float,
        default=0.5,
        help="Minimum rollout success rate required to proceed.",
    )

    notebook = parser.add_argument_group("notebook")
    notebook.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    notebook.add_argument(
        "--notebook-output", type=Path, default=None, help="Defaults to in-place."
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
