"""Tests for validate.py algorithm validation runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuracore.ml import validate

WRITE_SIXTY_LOG_LINES = """
i=1
while [ $i -lt 60 ]; do
    echo "{\\"message\\": \\"line $i\\"}" >> "$LOG"
    i=$((i + 1))
done
echo '{"message": "Initializing model"}' >> "$LOG"
"""


@pytest.fixture
def fake_cgroup_memory_events(tmp_path, monkeypatch):
    proc_self_cgroup = tmp_path / "proc_cgroup"
    proc_self_cgroup.write_text("0::/validation.slice\n")
    memory_events_path = tmp_path / "cgroup" / "validation.slice" / "memory.events"
    memory_events_path.parent.mkdir(parents=True)
    memory_events_path.write_text(
        "\n".join(["oom 0", "oom_kill 0", "oom_group_kill 0"])
    )
    monkeypatch.setattr(validate, "PROC_SELF_CGROUP", proc_self_cgroup)
    monkeypatch.setattr(validate, "CGROUP_ROOT", tmp_path / "cgroup")
    monkeypatch.setattr(validate, "_read_kernel_log", lambda: None)
    return memory_events_path


def run_validation_with_fake_python_script(
    monkeypatch, tmp_path, script_body, storage_handler=None
):
    """Run run_in_venv with a venv whose python runs the given shell body."""

    def create_fake_venv(venv_path, with_pip=True):
        venv_bin_dir = Path(venv_path) / "bin"
        venv_bin_dir.mkdir(parents=True)
        (venv_bin_dir / "pip").symlink_to("/bin/true")
        fake_python = venv_bin_dir / "python"
        fake_python.write_text(
            '#!/bin/sh\nLOG="$(dirname "$1")/validation-output/validate.log"\n'
            + script_body
        )
        fake_python.chmod(0o755)

    monkeypatch.setattr(validate.venv, "create", create_fake_venv)
    return validate.run_in_venv(tmp_path, storage_handler)


def test_run_in_venv_reports_out_of_memory_kill_with_log_tail(
    fake_cgroup_memory_events, monkeypatch, tmp_path
):
    script_body = WRITE_SIXTY_LOG_LINES + (
        f"printf '%s\\n' 'oom 1' 'oom_kill 1' > \"{fake_cgroup_memory_events}\"\n"
        "kill -9 $$\n"
    )

    _, error_msg = run_validation_with_fake_python_script(
        monkeypatch, tmp_path, script_body
    )

    assert error_msg.startswith(
        "Validation failed: it ran out of system RAM, so the system stopped it "
        "(exit code -9, SIGKILL)."
    )
    assert "Last 50 lines of the validation log:" in error_msg
    assert '"line 11"' in error_msg
    assert '"line 10"' not in error_msg
    assert error_msg.endswith('{"message": "Initializing model"}')


def test_run_in_venv_leaves_log_tail_out_of_a_cloud_validation_message(
    fake_cgroup_memory_events, monkeypatch, tmp_path
):
    monkeypatch.setattr(validate, "AlgorithmLogStreamer", MagicMock())
    storage_handler = MagicMock()

    _, error_msg = run_validation_with_fake_python_script(
        monkeypatch,
        tmp_path,
        WRITE_SIXTY_LOG_LINES + "kill -9 $$\n",
        storage_handler=storage_handler,
    )

    assert error_msg.startswith("The system stopped validation by force")
    assert "validation log" not in error_msg
    storage_handler.save_algorithm_validation_check.assert_called_once()
    assert (
        storage_handler.save_algorithm_validation_check.call_args.kwargs[
            "error_message"
        ]
        == error_msg
    )


def test_run_in_venv_reports_plain_sigkill_without_out_of_memory_kill(
    fake_cgroup_memory_events, monkeypatch, tmp_path
):
    _, error_msg = run_validation_with_fake_python_script(
        monkeypatch, tmp_path, WRITE_SIXTY_LOG_LINES + "kill -9 $$\n"
    )

    assert "It recorded no out of memory event" in error_msg
    assert "Initializing model" in error_msg


def test_run_in_venv_reports_segfault_before_any_log_lines(
    fake_cgroup_memory_events, monkeypatch, tmp_path
):
    _, error_msg = run_validation_with_fake_python_script(
        monkeypatch, tmp_path, "echo 'Fatal Python error' >&2\nkill -SEGV $$\n"
    )

    assert error_msg.startswith("Validation crashed with a segmentation fault")
    assert "Validation stopped before it wrote any log lines." in error_msg
    assert error_msg.endswith("Validation errors:\nFatal Python error")


def test_run_in_venv_keeps_traceback_from_validation_exception(
    fake_cgroup_memory_events, monkeypatch, tmp_path
):
    traceback_text = "Traceback (most recent call last):\nValueError: bad model"
    validation_results = json.dumps({
        "algo_check": None,
        "error_msg": traceback_text,
        "success": False,
    })
    script_body = (
        f"echo '{validate.JSON_START_MARKER}'\n"
        f"cat <<'EOF'\n{validation_results}\nEOF\n"
        f"echo '{validate.JSON_END_MARKER}'\n"
        "exit 1\n"
    )

    _, error_msg = run_validation_with_fake_python_script(
        monkeypatch, tmp_path, script_body
    )

    assert error_msg == traceback_text


@pytest.mark.parametrize(
    ("returncode", "expected_message_start"),
    [
        (-6, "Validation aborted (exit code -6, SIGABRT)."),
        (-15, "Validation received a request to stop (exit code -15, SIGTERM)"),
        (-10, "Validation stopped on SIGUSR1 (exit code -10)."),
        (3, "Validation exited with code 3 before reporting a result."),
    ],
)
def test_describe_exit_code_explains_each_exit_code(returncode, expected_message_start):
    exit_code_description = validate._describe_exit_code(
        returncode, pid=424, oom_kill_count_before=None
    )

    assert exit_code_description.startswith(expected_message_start)


def test_forced_stop_names_kernel_log_line_for_the_killed_pid(monkeypatch):
    kernel_log_text = (
        "[10.0] Out of memory: Killed process 4242 (python) total-vm:9000kB\n"
        "[11.0] Memory cgroup out of memory: Killed process 424 (python) "
        "total-vm:8000kB\n"
    )
    monkeypatch.setattr(validate, "_read_kernel_log", lambda: kernel_log_text)

    exit_code_description = validate._describe_exit_code(
        -9, pid=424, oom_kill_count_before=None
    )

    assert exit_code_description.startswith(
        "Validation failed: it ran out of system RAM"
    )
    assert exit_code_description.endswith(
        "Kernel log: [11.0] Memory cgroup out of memory: Killed process 424 "
        "(python) total-vm:8000kB"
    )


def test_forced_stop_says_reason_is_unreadable_without_sources(monkeypatch):
    monkeypatch.setattr(validate, "_read_kernel_log", lambda: None)
    monkeypatch.setattr(validate, "_read_oom_kill_count", lambda: None)

    exit_code_description = validate._describe_exit_code(
        -9, pid=424, oom_kill_count_before=None
    )

    assert exit_code_description.endswith(
        "This machine does not allow reading the exact reason."
    )
