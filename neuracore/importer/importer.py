"""Dataset import script for processing and importing datasets to Neuracore."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from neuracore_types.importer.config import DatasetTypeConfig
from neuracore_types.nc_data import DatasetImportConfig
from rich.logging import RichHandler

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from neuracore.importer.core.base import get_shared_console
from neuracore.importer.core.dataset_detector import (
    DatasetDetector,
    iter_first_two_levels,
)
from neuracore.importer.core.exceptions import (
    CLIError,
    ConfigLoadError,
    DatasetDetectionError,
    DatasetOperationError,
    UploaderError,
)
from neuracore.importer.core.utils import populate_robot_info
from neuracore.importer.core.validation import (
    validate_dataset_config_against_robot_model,
)
from neuracore.importer.lerobot_importer import LeRobotDatasetImporter
from neuracore.importer.rlds_importer import RLDSDatasetImporter

LOG_FORMAT = "%(message)s"
logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a concise, consistent format."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        console=get_shared_console(),
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])


def load_dataset_config(path: Path) -> DatasetImportConfig:
    """Read the user-provided YAML/JSON into a strongly typed config."""
    try:
        return DatasetImportConfig.from_file(path)
    except Exception as exc:  # noqa: BLE001 - show root cause to user
        raise ConfigLoadError(f"Failed to load dataset config '{path}': {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset import.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run dataset processing with a dataset config and robot model."
    )

    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Path to dataset configuration file (e.g. YAML or JSON)",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory",
    )

    parser.add_argument(
        "--robot-dir",
        type=Path,
        required=True,
        help="Path to directory containing robot description files (.urdf/.xml/.mjcf)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the dataset before importing if it already exists.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually logging data to Neuracore.",
    )

    parser.add_argument(
        "--skip-on-error",
        choices=["episode", "step", "all"],
        default="episode",
        help=(
            "Error handling strategy: "
            "'episode' skips the failed episode; "
            "'step' skips only the failing step; "
            "'all' aborts on the first error."
        ),
    )

    parser.add_argument(
        "--no-validation-warnings",
        action="store_true",
        help="Suppress warning messages from data validation.",
    )

    args = parser.parse_args()

    return args


def load_or_detect_dataset_type(
    dataconfig: DatasetImportConfig, dataset_dir: Path
) -> DatasetTypeConfig:
    """Prefer the explicit dataset type in config, otherwise auto-detect."""
    if dataconfig.dataset_type:
        return dataconfig.dataset_type

    try:
        detected = detect_dataset_type(dataset_dir)
        logger.info("Detected dataset type: %s", detected.value.upper())
        return detected
    except Exception as exc:  # noqa: BLE001 - surface detection failure
        raise DatasetDetectionError(str(exc)) from exc


def cli_args_validation(args: argparse.Namespace) -> None:
    """Validate the provided arguments."""
    for path in [args.dataset_config, args.dataset_dir]:
        if not path.exists():
            raise CLIError(f"Path does not exist: {path}")
    if args.robot_dir and not args.robot_dir.exists():
        raise CLIError(f"Robot description directory does not exist: {args.robot_dir}")


def detect_dataset_type(dataset_dir: Path) -> DatasetTypeConfig:
    """Detect whether the dataset is TFDS, RLDS, or LeRobot."""
    detector = DatasetDetector()
    try:
        return detector.detect(dataset_dir)
    except DatasetDetectionError as exc:
        # Preserve previous ValueError interface for callers/tests
        raise ValueError(str(exc)) from exc


def _resolve_robot_descriptions(
    config_urdf_path: str | None,
    config_mjcf_path: str | None,
    robot_dir: Path | None,
) -> tuple[str | None, str | None]:
    """Find URDF/MJCF files either from config paths or by scanning a folder."""
    urdf_path: str | None = None
    mjcf_path: str | None = None
    suffix_to_target = {".urdf": "urdf", ".xml": "mjcf", ".mjcf": "mjcf"}
    candidates = [
        Path(p)
        for p in (
            config_urdf_path,
            config_mjcf_path,
            robot_dir,
        )
        if p
    ]

    for candidate in candidates:
        if urdf_path and mjcf_path:
            break
        if candidate.is_file():
            target = suffix_to_target.get(candidate.suffix.lower())
            if target == "urdf" and urdf_path is None:
                urdf_path = str(candidate)
            elif target == "mjcf" and mjcf_path is None:
                mjcf_path = str(candidate)
            continue
        if candidate.is_dir():
            for path in iter_first_two_levels(candidate):
                if not path.is_file():
                    continue
                target = suffix_to_target.get(path.suffix.lower())
                if target == "urdf" and urdf_path is None:
                    urdf_path = str(path)
                elif target == "mjcf" and mjcf_path is None:
                    mjcf_path = str(path)
                if urdf_path and mjcf_path:
                    break

    return urdf_path, mjcf_path


def main() -> None:
    """Main entry point for dataset import script."""
    configure_logging()
    args = parse_args()

    try:
        cli_args_validation(args)
    except CLIError as exc:
        logger.error("Argument validation failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Starting dataset import\n  config: %s\n  data:   %s\n  robot:  %s",
        args.dataset_config,
        args.dataset_dir,
        args.robot_dir,
    )

    dataconfig = load_dataset_config(args.dataset_config)

    dataset_type = load_or_detect_dataset_type(dataconfig, args.dataset_dir)

    output_dataset = dataconfig.output_dataset
    if not output_dataset or not output_dataset.name:
        raise CLIError("'output_dataset.name' is required in the dataset config.")

    nc.login()

    dataset_name = output_dataset.name
    dataset = Dataset.get_by_name(dataset_name, non_exist_ok=True)
    if dataset is not None:
        if args.overwrite:
            logger.warning(
                "Dataset '%s' already exists. Overwrite requested; "
                "deleting existing dataset.",
                dataset_name,
            )
            try:
                dataset.delete()
            except Exception as exc:  # noqa: BLE001 - preserve traceback for user
                raise DatasetOperationError(
                    f"Failed to delete dataset '{dataset_name}': {exc}"
                ) from exc
            logger.info("Deleted existing dataset '%s'.", dataset_name)
            dataset = None
        else:
            logger.warning(
                "Dataset '%s' already exists; new data will be appended.",
                dataset_name,
            )
    if dataset is None:
        dataset = nc.create_dataset(
            name=dataset_name,
            description=dataconfig.output_dataset.description,
            tags=dataconfig.output_dataset.tags,
        )

    robot_config = dataconfig.robot
    urdf_path, mjcf_path = _resolve_robot_descriptions(
        robot_config.urdf_path,
        robot_config.mjcf_path,
        args.robot_dir,
    )
    if urdf_path is None and mjcf_path is None:
        search_hints = [
            str(p)
            for p in (
                robot_config.urdf_path,
                robot_config.mjcf_path,
                args.robot_dir,
            )
            if p
        ]
        searched_locations = (
            ", ".join(search_hints) if search_hints else "none provided"
        )
        raise CLIError(
            "Could not find a robot description file (.urdf or .xml/.mjcf). "
            f"Searched: {searched_locations}."
        )

    robot = nc.connect_robot(
        robot_name=robot_config.name,
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        overwrite=robot_config.overwrite_existing,
    )
    logger.info("Using robot model: %s (id=%s)", robot.name, robot.id)

    if robot.joint_info:
        validate_dataset_config_against_robot_model(dataconfig, robot.joint_info)
        dataconfig = populate_robot_info(dataconfig, robot.joint_info)

    logger.info("Setup complete; beginning import.")

    importer: RLDSDatasetImporter | LeRobotDatasetImporter | None = None
    skip_on_error = args.skip_on_error
    if dataset_type == DatasetTypeConfig.TFDS:
        raise NotImplementedError("TFDS import not yet implemented.")
    elif dataset_type == DatasetTypeConfig.RLDS:
        logger.info("Starting RLDS dataset import from %s", args.dataset_dir)
        importer = RLDSDatasetImporter(
            input_dataset_name=dataconfig.input_dataset_name,
            output_dataset_name=dataconfig.output_dataset.name,
            dataset_dir=args.dataset_dir,
            dataset_config=dataconfig,
            joint_info=robot.joint_info,
            dry_run=args.dry_run,
            suppress_warnings=args.no_validation_warnings,
            skip_on_error=skip_on_error,
        )
        importer.upload_all()
    elif dataset_type == DatasetTypeConfig.LEROBOT:
        logger.info("Starting LeRobot dataset import from %s", args.dataset_dir)
        importer = LeRobotDatasetImporter(
            input_dataset_name=dataconfig.input_dataset_name,
            output_dataset_name=dataconfig.output_dataset.name,
            dataset_dir=args.dataset_dir,
            dataset_config=dataconfig,
            joint_info=robot.joint_info,
            dry_run=args.dry_run,
            suppress_warnings=args.no_validation_warnings,
            skip_on_error=skip_on_error,
        )
        importer.upload_all()

    logger.info("Finished importing dataset.")


if __name__ == "__main__":
    try:
        main()
    except UploaderError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:  # noqa: BLE001 - unexpected crash; show stack
        logger.exception("Unexpected error during dataset import.")
        sys.exit(1)
