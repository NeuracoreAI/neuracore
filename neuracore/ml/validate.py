"""Algorithm validation script for neuracore ML algorithms.

This module provides a command-line tool for validating ML algorithms in an
isolated virtual environment. It creates a temporary venv, installs dependencies,
and runs validation to ensure algorithms meet neuracore requirements.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Optional, Tuple

import neuracore as nc
from neuracore.ml.utils.algorithm_storage_handler import AlgorithmStorageHandler
from neuracore.ml.utils.validate import AlgorithmCheck

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the markers used in the validation script
JSON_START_MARKER = "<<<NEURACORE_VALIDATION_JSON_START>>>"
JSON_END_MARKER = "<<<NEURACORE_VALIDATION_JSON_END>>>"


def run_in_venv(
    algorithm_folder: Path, storage_handler: Optional[AlgorithmStorageHandler] = None
) -> Tuple[AlgorithmCheck, str]:
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
import tempfile
from neuracore.ml.utils.validate import run_validation

try:
    algorithm_folder = Path(r"{algorithm_folder.absolute()}")
    algo_check, error_msg = run_validation(
        output_dir=Path(tempfile.TemporaryDirectory(prefix="nc-validate-").name),
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

            # Run validation in virtual environment
            logger.info("Validating algorithm...")
            result = subprocess.run(
                [
                    str(python_exe),
                    str(script_path),
                ],
                capture_output=True,
                text=True,
            )

            # Extract algo_check and error_msg from subprocess output
            algo_check, error_msg = _parse_validation_results(
                result.stdout, result.stderr
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
    return algo_check, error_msg


def _parse_validation_results(stdout: str, stderr: str) -> Tuple[AlgorithmCheck, str]:
    """Parse validation results from subprocess output.

    Args:
        stdout: Standard output from the validation subprocess
        stderr: Standard error from the validation subprocess

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
            error_msg = (
                f"No validation JSON output found. STDOUT: {stdout}, STDERR: {stderr}"
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
