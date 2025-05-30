import logging
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_in_venv(algorithm_folder):
    """Run validation in a temporary virtual environment"""
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

            # Create validation script
            validation_script = f"""
import sys
from pathlib import Path
import tempfile
from neuracore.ml.utils.validate import run_validation

algorithm_folder = Path(r"{algorithm_folder.absolute()}")
_, error_msg = run_validation(
    output_dir=Path(tempfile.TemporaryDirectory(prefix="nc-validate-").name),
    algorithm_dir=algorithm_folder,
    port=8080,
    skip_endpoint_check=True,
)
success = not error_msg
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
                check=True,
            )

            return result.returncode == 0

        except subprocess.CalledProcessError as e:
            error_msg = "Failed to validate.\n"
            if e.stderr:
                error_msg += e.stderr
            logger.error(error_msg)
            return False
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: neuracore-validate <path_to_algorithm_folder>")
        sys.exit(1)

    algorithm_folder = Path(sys.argv[1])
    if not algorithm_folder.is_dir():
        print(f"Error: {algorithm_folder} is not a valid directory.")
        sys.exit(1)

    success = run_in_venv(algorithm_folder)
    if success:
        logger.info(f"✅ Validation succeeded for {algorithm_folder}")
    else:
        logger.error(f"❌ Validation failed for {algorithm_folder}")
    sys.exit(0 if success else 1)
