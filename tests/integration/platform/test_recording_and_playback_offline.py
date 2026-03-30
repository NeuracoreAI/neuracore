import os
import uuid

import pytest
from recording_playback_shared import (
    HIGH_TIME_TO_DATASET_READY_S,
    assert_dataset_isolation,
    build_multi_producer_specs,
    daemon_mode,
    fetch_expected_trace_count_reported,
    fetch_trace_registration_stats,
    run_minimal_recording_flow,
    run_multi_producers,
    use_offline_daemon_profile,
    wait_for_dataset_ready,
    wait_for_online_recovery,
    wait_for_recording_to_exist_in_db,
)

import neuracore as nc


class TestOfflineProfileBehavior:
    def test_offline_profile_does_not_register_traces(self):
        """Verify that when using an offline profile."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="offline.registration"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            total_traces, non_pending = fetch_trace_registration_stats(recording_id)
            assert total_traces > 0
            assert non_pending == 0

    def test_offline_profile_does_not_report_expected_trace_count(self):
        """Verify that when using an offline profile, the daemon does not report
        any expected trace counts for recordings started in offline mode.

        This ensures that the daemon does not incorrectly report trace counts
        when in offline mode, which can interfere with the expected trace count
        reporting mechanism.

        :return: None
        :rtype: None
        """
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="offline.expected_trace_count"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            expected_trace_count_reported = fetch_expected_trace_count_reported(
                recording_id
            )
            assert expected_trace_count_reported == 0

    def test_offline_pending_data_recovers_when_online(self):
        """Verify offline pending data recovers when switching to online mode.

        This test checks that the daemon can recover trace
        registration and expected trace counts
        when switching from offline to online mode.

        :return: None
        :rtype: None
        """
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="offline.recovery_seed"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            total_traces, non_pending = fetch_trace_registration_stats(recording_id)
            assert total_traces > 0
            assert non_pending == 0

            expected_trace_count_reported = fetch_expected_trace_count_reported(
                recording_id
            )
            assert expected_trace_count_reported == 0

        previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")
        try:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)

            nc.login()
            nc.connect_robot(f"recovery_robot_{uuid.uuid4().hex[:8]}")
            wait_for_online_recovery(recording_id)
        finally:
            if previous_profile is None:
                os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
            else:
                os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile


def test_multiple_producers_offline_record_then_online_upload():
    """Test that multiple producers can complete recording traces offline.

    The Flow:
        1. Create multiple producers with different durations
        2. Start all producers offline
        3. Verify all producers complete recording traces offline
        4. Verify all recordings are uploaded online

    Why This Matters:
        Multiple producers should not block other recordings. Producers with
        different durations should not block other recordings.

    Key Assertions:
        - All producers complete recording traces offline
        - All recordings are uploaded online
    """
    specs = build_multi_producer_specs(
        num_producers=2,
        wait=False,
        with_video=True,
        fps=10,
        duration_sec=1.5,
        image_width=96,
        image_height=72,
    )

    nc.login()
    with daemon_mode(offline=True):
        offline_results = run_multi_producers(specs)
        for result in offline_results:
            with pytest.raises(TimeoutError):
                wait_for_dataset_ready(
                    result["dataset_name"],
                    expected_recording_count=1,
                    timeout_s=12,
                    poll_interval_s=1.0,
                )

    nc.login()
    for result in offline_results:
        wait_for_dataset_ready(
            result["dataset_name"],
            expected_recording_count=1,
            timeout_s=HIGH_TIME_TO_DATASET_READY_S,
        )
        assert_dataset_isolation(result)
