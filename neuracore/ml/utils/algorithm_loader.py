import importlib.machinery
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Type

from ..neuracore_model import NeuracoreModel

logger = logging.getLogger(__name__)


class AlgorithmLoader:
    """
    Utility for loading the first NeuracoreModel subclass found in a directory,
    properly importing the directory as a package to support relative imports.
    """

    def __init__(self, algorithm_dir: Path):
        """
        Initialize the algorithm loader.

        Args:
            algorithm_dir: Directory containing the algorithm code
        """
        self.algorithm_dir = algorithm_dir

    def install_requirements(self) -> bool:
        """
        Check for requirements.txt in the algorithm directory and install packages.

        Returns:
            bool: True if requirements were installed successfully, False otherwise
        """
        req_file = self.algorithm_dir / "requirements.txt"
        if not req_file.exists():
            logger.info("No requirements.txt found in algorithm directory")
            return True

        logger.info(f"Found requirements.txt at {req_file}")
        try:
            # Install requirements using pip
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            )
            logger.info("Successfully installed requirements")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False

    def parse_requirements(self) -> Optional[List[str]]:
        """
        Parse requirements.txt file to get a list of required packages.

        Returns:
            Optional[List[str]]: List of required packages or None if file doesn't exist
        """
        req_file = self.algorithm_dir / "requirements.txt"
        if not req_file.exists():
            return None

        requirements = []
        try:
            with open(req_file, "r") as f:
                for line in f:
                    # Skip comments and empty lines
                    line = line.strip()
                    if line and not line.startswith("#"):
                        requirements.append(line)
            return requirements
        except Exception as e:
            logger.error(f"Failed to parse requirements.txt: {e}")
            return None

    def get_all_files(self) -> List[Path]:
        """
        Get all files in the algorithm directory.

        Returns:
            List[Path]: List of all files in the directory
        """
        files = []
        for root, _, filenames in os.walk(self.algorithm_dir):
            for filename in filenames:
                if filename.endswith(".py") and filename != "__init__.py":
                    files.append(Path(root) / filename)
        return files

    def load_model(self) -> Type[NeuracoreModel]:
        """
        Find and load the first class that inherits from NeuracoreModel.

        This method checks for requirements.txt and installs dependencies,
        then properly sets up the directory as a package to allow
        relative imports between files in the same directory.

        Returns:
            The first NeuracoreModel subclass found

        Raises:
            ImportError: If no NeuracoreModel subclass is found
        """
        # Install requirements if they exist
        self.install_requirements()
        # Create __init__.py if it doesn't exist to make the directory a proper package
        init_path = self.algorithm_dir / "__init__.py"
        if not init_path.exists():
            init_path.touch()
            logger.info(f"Created __init__.py at {init_path}")

        # Get the package name from the directory name
        package_name = self.algorithm_dir.name

        # Add the parent directory to sys.path so the package can be imported
        parent_dir = str(self.algorithm_dir.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.info(f"Added {parent_dir} to sys.path")

        found_model = None

        # First try importing the entire package
        try:
            package = importlib.import_module(package_name)
            logger.info(f"Successfully imported package: {package_name}")

            # Check for NeuracoreModel subclasses in the main package
            for attr_name in dir(package):
                package_attr = getattr(package, attr_name)
                if (
                    isinstance(package_attr, type)
                    and issubclass(package_attr, NeuracoreModel)
                    and package_attr != NeuracoreModel
                ):
                    found_model = package_attr
                    logger.info(f"Found model in main package: {found_model.__name__}")
                    return found_model
        except ImportError:
            logger.warning(f"Failed to import package {package_name} directly")

        # Then check each Python file in the directory
        for file_path in self.algorithm_dir.glob("**/*.py"):
            if file_path.name == "__init__.py":
                continue

            # Skip system files
            if file_path.stem.startswith("."):
                continue

            # Determine the module name relative to the package
            relative_path = file_path.relative_to(self.algorithm_dir.parent)
            module_path = str(relative_path).replace(os.sep, ".")[
                :-3
            ]  # Remove .py extension

            logger.info(f"Attempting to import: {module_path}")

            try:
                # Import the module as part of the package
                module = importlib.import_module(module_path)

                # Search for NeuracoreModel subclasses
                for attr_name in dir(module):
                    module_attr = getattr(module, attr_name)
                    if (
                        isinstance(module_attr, type)
                        and issubclass(module_attr, NeuracoreModel)
                        and module_attr != NeuracoreModel
                    ):
                        found_model = module_attr
                        logger.info(
                            f"Found model in {module_path}: {found_model.__name__}"
                        )
                        return found_model

            except ImportError as e:
                logger.warning(f"Failed to import {module_path}: {e}")

                # Fallback to spec-based import if package-based import fails
                try:
                    module_name = f"{package_name}_{file_path.stem}"
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_path
                    )
                    if spec is None:
                        continue

                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Search for NeuracoreModel subclasses
                    for attr_name in dir(module):
                        module_attr = getattr(module, attr_name)
                        if (
                            isinstance(module_attr, type)
                            and issubclass(module_attr, NeuracoreModel)
                            and module_attr != NeuracoreModel
                        ):
                            found_model = module_attr
                            logger.info(
                                f"Found model in {module_name} "
                                f"(fallback): {found_model.__name__}"
                            )
                            return found_model

                except Exception as e:
                    logger.warning(f"Failed fallback import for {file_path}: {e}")

        if found_model is None:
            raise ImportError(
                "Could not find any class inheriting "
                f"from NeuracoreModel in {self.algorithm_dir}"
            )

        return found_model
