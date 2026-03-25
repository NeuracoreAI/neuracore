import sys
from pathlib import Path

import pytest
from recording_playback_shared import cleanup_test_profiles, daemon_cleanup

sys.path.append(str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def daemon_setup_teardown():
    daemon_cleanup()
    yield
    daemon_cleanup()


@pytest.fixture(autouse=True)
def cleanup_profiles():
    yield
    cleanup_test_profiles()
