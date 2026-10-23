"""Algorithm validation script for neuracore ML algorithms.

This module provides a command-line tool for validating ML algorithms in an
isolated virtual environment. It creates a temporary venv, installs dependencies,
and runs validation to ensure algorithms meet neuracore requirements.
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import venv
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import neuracore as nc
from neuracore.ml.logging.algorithm_log_streamer import AlgorithmLogStreamer
from neuracore.ml.utils.algorithm_storage_handler import AlgorithmStorageHandler
from neuracore.ml.utils.validate import AlgorithmCheck

logger = logging.getLogger(__name__)

# Define the markers used in the validation script
JSON_START_MARKER = "<<<NEURACORE_VALIDATION_JSON_START>>>"
JSON_END_MARKER = "<<<NEURACORE_VALIDATION_JSON_END>>>"

LOG_TAIL_LINES = 50
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")


class SignalExitCode(IntEnum):
    """Return codes of a process that a POSIX signal stopped."""

    SIGABRT = -6
    SIGKILL = -9
    SIGSEGV = -11
    SIGTERM = -15


@dataclass
class OutOfMemoryKillEvidence:
    """What the kernel recorded about an out of memory kill of one process.

    Attributes:
        killed_by_out_of_memory: True when a source shows an out of memory
            kill, False when the readable sources show none, and None when no
            source is readable.
        kernel_log_line: The kernel log line that reports the kill, or an
            empty string.
    """

    killed_by_out_of_memory: bool | None
    kernel_log_line: str = ""


def run_in_venv(
    algorithm_folder: Path, storage_handler: AlgorithmStorageHandler | None = None
) -> tuple[AlgorithmCheck, str]:
    """Run algorithm validation in a temporary virtual environment.

    Creates an isolated virtual environment, installs neuracore[ml], and
    executes validation to ensure the algorithm meets all requirements.

    Args:
        algorithm_folder: Path to the algorithm directory to validate
        storage_handler: Optional storage handler for saving validation results

    Returns:
        bool: True if validation succeeded, False otherwise
    """
    with tempfile.TemporaryDirectory(prefix="nc-validate-venv-") as temp_dir:
        venv_path = Path(temp_dir) / "venv"
        validation_output_dir = Path(temp_dir) / "validation-output"
        validation_output_dir.mkdir(parents=True, exist_ok=True)
        log_streamer: AlgorithmLogStreamer | None = None
        algo_check = AlgorithmCheck()
        error_msg = ""

        # Create virtual environment
        venv.create(venv_path, with_pip=True)

        # Determine the python executable path
        if os.name == "nt":  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"

        try:
            # Install neuracore in the virtual environment
            subprocess.run(
                [
                    str(pip_exe),
                    "install",
                    "neuracore[ml]",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(f"Created virtual environment at {venv_path}")

            validation_script = f"""
import sys
import json
from pathlib import Path
from neuracore.ml.utils.validate import run_validation

try:
    algorithm_folder = Path(r"{algorithm_folder.absolute()}")
    algo_check, error_msg = run_validation(
        output_dir=Path(r"{validation_output_dir}"),
        algorithm_dir=algorithm_folder,
        port=8080,
        skip_endpoint_check=True,
    )

    # Output results as JSON with markers to stdout
    results = {{
        "algo_check": algo_check.model_dump(),
        "error_msg": error_msg,
        "success": not error_msg
    }}
    success = not error_msg    

except Exception as e:
    # Output error as JSON with markers
    success = False
    results = {{
        "algo_check": None,
        "error_msg": str(e),
        "success": success
    }}

print("{JSON_START_MARKER}")
print(json.dumps(results))
print("{JSON_END_MARKER}")
sys.exit(0 if success else 1)
"""

            script_path = Path(temp_dir) / "validate.py"
            script_path.write_text(validation_script)

            if storage_handler is not None:
                log_streamer = AlgorithmLogStreamer(
                    storage_handler=storage_handler,
                    output_dir=validation_output_dir,
                )
                log_streamer.start()

            # Run validation in virtual environment
            logger.info("Validating algorithm...")
            oom_kill_count_before = _read_oom_kill_count()
            with subprocess.Popen(
                [
                    str(python_exe),
                    str(script_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as validation_process:
                stdout, stderr = validation_process.communicate()

            failure_description = ""
            if validation_process.returncode != 0:
                failure_description = _describe_validation_process_failure(
                    returncode=validation_process.returncode,
                    pid=validation_process.pid,
                    log_path=validation_output_dir / "validate.log",
                    oom_kill_count_before=oom_kill_count_before,
                    include_log_tail=storage_handler is None,
                )

            # Extract algo_check and error_msg from subprocess output
            algo_check, error_msg = _parse_validation_results(
                stdout, stderr, failure_description
            )

            # Save validation results if storage handler is provided
            if storage_handler:
                storage_handler.save_algorithm_validation_check(
                    checklist=algo_check,
                    error_message=error_msg,
                )

        except subprocess.CalledProcessError as e:
            error_msg = "Failed to validate.\n"
            if e.stderr:
                error_msg += e.stderr
            logger.error(error_msg, exc_info=True)
            # Save error to storage handler if provided
            if storage_handler:
                storage_handler.save_algorithm_validation_check(
                    checklist=AlgorithmCheck(),
                    error_message=error_msg,
                )
        finally:
            if log_streamer is not None:
                log_streamer.close()
    return algo_check, error_msg


def _read_oom_kill_count() -> int | None:
    """Read the oom_kill counter of the cgroup v2 group this process belongs to.

    Returns:
        The number of processes the OOM killer stopped in this cgroup, or None
        when the cgroup memory events are unreadable.
    """
    try:
        for cgroup_line in PROC_SELF_CGROUP.read_text().splitlines():
            if not cgroup_line.startswith("0::"):
                continue
            cgroup_path = cgroup_line.removeprefix("0::").lstrip("/")
            memory_events_path = CGROUP_ROOT / cgroup_path / "memory.events"
            for memory_event_line in memory_events_path.read_text().splitlines():
                counter_name, _, counter_value = memory_event_line.partition(" ")
                if counter_name == "oom_kill":
                    return int(counter_value)
    except (OSError, ValueError):
        return None
    return None


def _read_kernel_log() -> str | None:
    """Read the kernel ring buffer with dmesg.

    Returns:
        The kernel log text, or None when dmesg is missing or not permitted.
    """
    try:
        dmesg_result = subprocess.run(
            ["dmesg"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return dmesg_result.stdout


def _find_out_of_memory_kill_evidence(
    pid: int, oom_kill_count_before: int | None
) -> OutOfMemoryKillEvidence:
    """Look for an out of memory kill of the process in the kernel records.

    Args:
        pid: Process ID of the killed validation process.
        oom_kill_count_before: The cgroup oom_kill counter read before the
            process started, or None when it was unreadable.

    Returns:
        The kernel log line for the process when present, otherwise the
        result of comparing the cgroup oom_kill counter before and after.
    """
    kernel_log_text = _read_kernel_log()
    if kernel_log_text is not None:
        kill_line_pattern = re.compile(rf"Killed process {pid}\b")
        matching_kernel_log_lines = [
            kernel_log_line
            for kernel_log_line in kernel_log_text.splitlines()
            if kill_line_pattern.search(kernel_log_line)
        ]
        if matching_kernel_log_lines:
            return OutOfMemoryKillEvidence(
                killed_by_out_of_memory=True,
                kernel_log_line=matching_kernel_log_lines[-1].strip(),
            )

    oom_kill_count_after = _read_oom_kill_count()
    match (oom_kill_count_before, oom_kill_count_after):
        case (int() as count_before, int() as count_after) if (
            count_after > count_before
        ):
            return OutOfMemoryKillEvidence(killed_by_out_of_memory=True)
        case (int(), int()):
            return OutOfMemoryKillEvidence(killed_by_out_of_memory=False)
        case _ if kernel_log_text is not None:
            return OutOfMemoryKillEvidence(killed_by_out_of_memory=False)
        case _:
            return OutOfMemoryKillEvidence(killed_by_out_of_memory=None)


def _describe_forced_stop(evidence: OutOfMemoryKillEvidence) -> str:
    """Describe a validation process that SIGKILL stopped.

    Args:
        evidence: What the kernel recorded about an out of memory kill.

    Returns:
        A message that names out of memory as the cause when the evidence
        shows it, and tells the user what to change.
    """
    match evidence.killed_by_out_of_memory:
        case True:
            description = (
                "Validation failed: it ran out of system RAM, so the system "
                "stopped it (exit code -9, SIGKILL). Reduce the system RAM your "
                "model needs, for example by using a smaller model."
            )
        case False:
            description = (
                "The system stopped validation by force (exit code -9, SIGKILL). "
                "It recorded no out of memory event, so another process or a "
                "person stopped it."
            )
        case None:
            description = (
                "The system stopped validation by force (exit code -9, SIGKILL). "
                "This usually means validation ran out of system RAM. This machine "
                "does not allow reading the exact reason."
            )
    if evidence.kernel_log_line:
        description += f"\nKernel log: {evidence.kernel_log_line}"
    return description


def _describe_exit_code(
    returncode: int, pid: int, oom_kill_count_before: int | None
) -> str:
    """Describe in plain words how the validation process ended.

    Args:
        returncode: Exit code of the validation process. A negative value is
            the number of the signal that stopped it.
        pid: Process ID of the validation process.
        oom_kill_count_before: The cgroup oom_kill counter read before the
            process started, or None when it was unreadable.

    Returns:
        A user facing message for the exit code.
    """
    match returncode:
        case SignalExitCode.SIGKILL:
            return _describe_forced_stop(
                _find_out_of_memory_kill_evidence(pid, oom_kill_count_before)
            )
        case SignalExitCode.SIGSEGV:
            return (
                "Validation crashed with a segmentation fault (exit code -11, "
                "SIGSEGV). This usually comes from native code your model uses, "
                "such as a CUDA or C++ extension."
            )
        case SignalExitCode.SIGABRT:
            return (
                "Validation aborted (exit code -6, SIGABRT). Native code your "
                "model uses, such as a CUDA or C++ library, stopped the process "
                "after an internal error."
            )
        case SignalExitCode.SIGTERM:
            return (
                "Validation received a request to stop (exit code -15, SIGTERM), "
                "for example because its machine shut down. Try validating again."
            )
        case negative_returncode if negative_returncode < 0:
            try:
                signal_name = signal.Signals(-negative_returncode).name
            except ValueError:
                signal_name = "an unknown signal"
            return (
                f"Validation stopped on {signal_name} "
                f"(exit code {negative_returncode})."
            )
        case _:
            return (
                f"Validation exited with code {returncode} before reporting "
                "a result."
            )


def _read_log_tail(log_path: Path, max_line_count: int = LOG_TAIL_LINES) -> str:
    """Read the last lines of the validation log.

    Args:
        log_path: Path to the validation log file.
        max_line_count: Maximum number of lines to return.

    Returns:
        A heading followed by the last lines of the log, or a note when the log
        holds no lines or is unreadable.
    """
    try:
        with log_path.open(errors="replace") as log_file:
            last_log_lines = deque(log_file, maxlen=max_line_count)
    except FileNotFoundError:
        last_log_lines = deque()
    except OSError:
        return "The validation log is unreadable."
    if not last_log_lines:
        return "Validation stopped before it wrote any log lines."
    return (
        f"Last {len(last_log_lines)} lines of the validation log:\n"
        + "".join(last_log_lines).rstrip()
    )


def _describe_validation_process_failure(
    returncode: int,
    pid: int,
    log_path: Path,
    oom_kill_count_before: int | None,
    include_log_tail: bool,
) -> str:
    """Describe a validation process that exited with a nonzero code.

    Args:
        returncode: Exit code of the validation process.
        pid: Process ID of the validation process.
        log_path: Path to the validation log file.
        oom_kill_count_before: The cgroup oom_kill counter read before the
            process started, or None when it was unreadable.
        include_log_tail: Whether to add the last lines of the validation log,
            for runs whose log reaches no cloud storage.

    Returns:
        The plain language exit description, followed by the log tail when
        include_log_tail is set.
    """
    exit_code_description = _describe_exit_code(returncode, pid, oom_kill_count_before)
    if not include_log_tail:
        return exit_code_description
    return f"{exit_code_description}\n\n{_read_log_tail(log_path)}"


def _parse_validation_results(
    stdout: str, stderr: str, failure_description: str = ""
) -> tuple[AlgorithmCheck, str]:
    """Parse validation results from subprocess output.

    Args:
        stdout: Standard output from the validation subprocess
        stderr: Standard error from the validation subprocess
        failure_description: Plain language exit description and log tail of
            a validation subprocess that exited with a nonzero code, used as
            the error when the output has no results

    Returns:
        Tuple of (algo_check, error_msg)
    """
    algo_check = AlgorithmCheck()
    error_msg = ""

    try:
        # Extract JSON from between markers
        stdout = stdout.strip()
        if stdout and JSON_START_MARKER in stdout and JSON_END_MARKER in stdout:
            # Find the JSON content between markers
            start_idx = stdout.find(JSON_START_MARKER) + len(JSON_START_MARKER)
            end_idx = stdout.find(JSON_END_MARKER)

            if start_idx < end_idx:
                json_content = stdout[start_idx:end_idx].strip()
                results = json.loads(json_content)

                if results.get("algo_check"):
                    algo_check = AlgorithmCheck.model_validate(
                        results.get("algo_check")
                    )
                error_msg = results.get("error_msg", "")
            else:
                error_msg = (
                    "Could not extract JSON content from output. "
                    f"STDOUT: {stdout}, STDERR: {stderr}"
                )
        else:
            # If no markers found, treat as error
            if failure_description:
                error_msg = failure_description
                if stdout:
                    error_msg += f"\n\nValidation output:\n{stdout}"
                if stderr.strip():
                    error_msg += f"\n\nValidation errors:\n{stderr.strip()}"
            else:
                error_msg = (
                    "No validation JSON output found. "
                    f"STDOUT: {stdout}, STDERR: {stderr}"
                )

    except json.JSONDecodeError as e:
        error_msg = (
            "Failed to parse validation results: "
            f"{e}. STDOUT: {stdout}, STDERR: {stderr}"
        )
    except Exception as e:
        error_msg = (
            "Unexpected error parsing results: "
            f"{e}. STDOUT: {stdout}, STDERR: {stderr}"
        )

    return algo_check, error_msg


def main() -> None:
    """Main entry point for the neuracore-validate command-line tool.

    Parses command-line arguments, validates the provided algorithm folder,
    and exits with appropriate status code.

    Usage:
        neuracore-validate <path_to_algorithm_folder>
        neuracore-validate --algorithm_folder <path_to_algorithm_folder>
        neuracore-validate --algorithm_id <algorithm_id>

    Exit codes:
        0: Validation succeeded
        1: Validation failed or invalid arguments
    """
    parser = argparse.ArgumentParser(
        description="Validate neuracore ML algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to algorithm folder",
    )
    parser.add_argument(
        "--algorithm_folder",
        type=Path,
        help="Path to the algorithm directory to validate",
    )
    parser.add_argument(
        "--algorithm_id",
        type=str,
        help="Algorithm ID to pass to validation",
    )
    parser.add_argument(
        "--org_id",
        type=str,
        help="Organization ID to use for validation",
    )

    args = parser.parse_args()

    # Determine algorithm folder
    algorithm_folder = None
    storage_handler = None

    if args.algorithm_folder:
        algorithm_folder = args.algorithm_folder
    elif args.path:
        algorithm_folder = Path(args.path)
    elif args.algorithm_id:
        nc.login()
        nc.set_organization(args.org_id)
        tempfile_dir = Path(tempfile.gettempdir())
        storage_handler = AlgorithmStorageHandler(algorithm_id=args.algorithm_id)
        algorithm_folder = Path(tempfile_dir) / "algorithm"
        storage_handler.download_algorithm(extract_dir=algorithm_folder)
        logger.info(f"Algorithm extracted to {algorithm_folder}")

    # Validate that the folder exists
    if not algorithm_folder or not algorithm_folder.is_dir():
        print(f"Error: {algorithm_folder} is not a valid directory.")
        sys.exit(1)

    algo_check, error_msg = run_in_venv(algorithm_folder, storage_handler)
    success = not error_msg

    if success:
        logger.info(f"✅ Validation succeeded for {algorithm_folder}")
    else:
        logger.error(f"❌ Validation failed for {algorithm_folder}")
        logger.error(f"Error message: {error_msg}")
        if algo_check:
            logger.error(
                f"Validation checklist: {algo_check.model_dump_json(indent=2)}"
            )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
