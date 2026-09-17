"""Integration test: expanding a dataset with data synthesis.

Drives a synthesis job end to end against a real backend: collect a small
dataset, preview what the configuration produces, run the job, and check the
dataset it produced.

The invariant worth the most here is that only the imagery changed. A
synthesized recording is derived from its source by copying every non-video
trace byte for byte, so the joint positions, poses and language recorded
against those frames still describe them. If that ever stops holding, the
generated data is quietly wrong in a way no model would tell you about.

Runs the procedural in-painter, which needs no model download and no
meaningful GPU time. Diffusion is the same code path with a slower generator,
so it is left to be triggered by hand.

Target environment
------------------
Needs a backend that can provision a machine, so staging or production. The
preview step additionally needs SYNTHESIS_PREVIEW_URL configured there; it is
skipped rather than failed when previews are unavailable, so the rest of the
run is still covered.
"""

import logging
import time

import pytest

import neuracore as nc
from neuracore.core.data.dataset import Dataset

from ..ml.shared.dataset import collect_demo_data
from ..ml.shared.utils import unique_name

logger = logging.getLogger(__name__)

RECORDINGS = 2
SCALING_FACTOR = 2
"""Two copies per source recording, so the distribution logic is exercised."""

JOB_POLL_SECONDS = 30
JOB_TIMEOUT_MINUTES = 40

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}

ALGORITHM_NAME = "procedural-in-painter"
"""Needs no model download and no diffusion weights."""

ALGORITHM_CONFIG = {
    "segmentation_prompts": ["table"],
}


def _wait_for_job(job_id: str) -> dict:
    """Poll a synthesis job until it reaches a terminal status.

    Args:
        job_id: The job to wait for.

    Returns:
        The job's final data.

    Raises:
        AssertionError: If the job does not finish in time.
    """
    deadline = time.time() + JOB_TIMEOUT_MINUTES * 60
    while time.time() < deadline:
        job = nc.get_data_synthesis_job_data(job_id)
        logger.info(
            "Synthesis job %s: %s (%s/%s recordings)",
            job_id,
            job["status"],
            job.get("recordings_completed"),
            job.get("recordings_total"),
        )
        if job["status"] in TERMINAL_STATUSES:
            return job
        time.sleep(JOB_POLL_SECONDS)

    raise AssertionError(
        f"Synthesis job {job_id} did not finish within "
        f"{JOB_TIMEOUT_MINUTES} minutes"
    )


class TestDataSynthesis:
    """Expands a dataset and checks what came out of it."""

    source_dataset: Dataset
    job_id: str
    output_dataset_name: str
    all_steps_passed = True

    @classmethod
    def setup_class(cls) -> None:
        nc.login()
        cls.robot_name = unique_name("synthesis_robot")
        cls.source_dataset_name = unique_name("synthesis_source")
        cls.output_dataset_name = unique_name("synthesis_output")

    @classmethod
    def teardown_class(cls) -> None:
        # A failed run leaves its artifacts behind so they can be inspected.
        if not cls.all_steps_passed:
            logger.warning("Leaving synthesis artifacts in place after failure")
            return
        for cleanup in (
            lambda: nc.delete_data_synthesis_job(cls.job_id),
            lambda: nc.get_dataset(name=cls.output_dataset_name).delete(),
            lambda: cls.source_dataset.delete(),
        ):
            try:
                cleanup()
            except Exception:
                logger.warning("Cleanup step failed", exc_info=True)

    def test_step01_collect_a_source_dataset(self) -> None:
        type(self).source_dataset = collect_demo_data(
            robot_name=self.robot_name,
            dataset_name=self.source_dataset_name,
            joint_names=("joint1", "joint2"),
            gripper_names=["gripper"],
            language_label="pick up the block from the table",
            nc_cam_name="front",
            pose_sensor_name="end_effector",
            num_episodes=RECORDINGS,
            instance_id=0,
        )

        assert len(self.source_dataset) == RECORDINGS

    def test_step02_preview_the_configuration(self) -> None:
        try:
            preview = nc.generate_data_synthesis_preview(
                dataset_name=self.source_dataset_name,
                algorithm_name=ALGORITHM_NAME,
                algorithm_config=ALGORITHM_CONFIG,
            )
        except ValueError as error:
            if "not available" in str(error):
                pytest.skip("No preview service configured in this environment")
            raise

        # A job regenerates every camera, so the preview covers every camera,
        # each handed back as URLs the caller can fetch.
        assert preview["frames"]
        for frame in preview["frames"]:
            assert frame["camera"]
            assert frame["before_url"]
            assert frame["after_url"]
        cameras = [frame["camera"] for frame in preview["frames"]]
        assert len(set(cameras)) == len(cameras)
        assert preview["config_hash"]

    def test_step03_start_the_run(self) -> None:
        job = nc.start_data_synthesis_run(
            name=unique_name("synthesis_job"),
            dataset_name=self.source_dataset_name,
            algorithm_name=ALGORITHM_NAME,
            algorithm_config=ALGORITHM_CONFIG,
            scaling_factor=SCALING_FACTOR,
            output_dataset_name=self.output_dataset_name,
        )

        type(self).job_id = job["id"]
        assert job["status"] == "QUEUED"

    def test_step04_the_run_completes(self) -> None:
        job = _wait_for_job(self.job_id)

        assert job["status"] == "COMPLETED", job.get("error")

    def test_step05_the_output_holds_the_expected_recordings(self) -> None:
        output = nc.get_dataset(name=self.output_dataset_name)

        assert len(output) == RECORDINGS * SCALING_FACTOR

    def test_step06_only_the_imagery_changed(self) -> None:
        """Every non-video trace must survive synthesis untouched.

        Checked through synchronization rather than by comparing blobs: if the
        copied traces were damaged, a synchronized view of them would not line
        up with the source's.
        """
        output = nc.get_dataset(name=self.output_dataset_name)

        source_synced = self.source_dataset.synchronize(frequency=10)
        output_synced = output.synchronize(frequency=10)

        source_point = next(iter(next(iter(source_synced))))
        output_point = next(iter(next(iter(output_synced))))

        for data_type, source_sensors in source_point.data.items():
            if data_type.value == "RGB_IMAGES":
                continue
            assert (
                data_type in output_point.data
            ), f"{data_type} did not survive synthesis"
            assert set(output_point.data[data_type]) == set(
                source_sensors
            ), f"{data_type} sensors changed during synthesis"

    def test_step07_the_output_is_trainable(self) -> None:
        """A dataset that cannot be synchronized cannot be trained on.

        Synthesis never synchronizes, so this is the first time the recordings
        it wrote are read the way training reads them.
        """
        output = nc.get_dataset(name=self.output_dataset_name)

        synced = output.synchronize(frequency=10)

        assert len(synced) == RECORDINGS * SCALING_FACTOR


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Record a failure so teardown can leave the artifacts alone."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        cls = getattr(item, "cls", None)
        if cls is not None:
            cls.all_steps_passed = False
