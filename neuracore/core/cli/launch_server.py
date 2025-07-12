"""Algorithm validation script for neuracore ML algorithms.

This module provides a command-line tool for validating ML algorithms in an
isolated virtual environment. It creates a temporary venv, installs dependencies,
and runs validation to ensure algorithms meet neuracore requirements.
"""

import argparse
import logging

import neuracore as nc
from neuracore.core.endpoint import policy_local_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        "--job_id",
        type=str,
        help="Job ID to run",
    )
    parser.add_argument(
        "--org_id",
        type=str,
        help="Organization ID",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
    )

    args = parser.parse_args()

    nc.login()
    nc.set_organization(args.org_id)
    policy = policy_local_server(
        train_run_name="",  # Use job id instead
        port=args.port,
        host=args.host,
        job_id=args.job_id,
    )
    assert policy.server_process is not None
    policy.server_process.wait()


if __name__ == "__main__":
    main()
