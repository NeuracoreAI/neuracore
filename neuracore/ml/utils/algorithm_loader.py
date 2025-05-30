import importlib.util
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Type

from ..neuracore_model import NeuracoreModel

logger = logging.getLogger(__name__)


class AlgorithmLoaderError(Exception):
    """Base exception for algorithm loading errors."""

    pass


class RequirementsInstallError(AlgorithmLoaderError):
    """Raised when requirements installation fails."""

    pass


class PackageSetupError(AlgorithmLoaderError):
    """Raised when package setup fails."""

    pass


class ModuleImportError(AlgorithmLoaderError):
    """Raised when module import fails."""

    pass


class ModelNotFoundError(AlgorithmLoaderError):
    """Raised when no NeuracoreModel subclass is found."""

    pass


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

        Raises:
            RequirementsInstallError: If requirements installation fails
        """
        req_file = self.algorithm_dir / "requirements.txt"
        if not req_file.exists():
            logger.info("No requirements.txt found in algorithm directory")
            return True

        logger.info(f"Found requirements.txt at {req_file}")

        # Strip out neuracore dependencies to avoid conflicts
        try:
            with open(req_file, "r") as f:
                lines = f.readlines()
            purged_lines = [line for line in lines if "neuracore" not in line.lower()]
            if len(purged_lines) != len(lines):
                logger.info("Purging Neuracore dependencies from requirements.txt")
                with open(req_file, "w") as f:
                    f.writelines(purged_lines)
        except Exception as e:
            error_msg = f"Failed to process requirements.txt: {e}"
            logger.error(error_msg)
            raise RequirementsInstallError(error_msg)

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-Ir", str(req_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Successfully installed requirements")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = "Failed to install requirements from your requirements.txt\n"
            if e.stderr:
                error_msg += e.stderr
            logger.error(error_msg)
            raise RequirementsInstallError(error_msg)
        except FileNotFoundError as e:
            error_msg = f"pip executable not found: {e}"
            logger.error(error_msg)
            raise RequirementsInstallError(error_msg)

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

    def _setup_package_environment(self) -> str:
        """
        Set up the package environment for importing.

        Returns:
            str: The package name

        Raises:
            PackageSetupError: If package setup fails
        """
        try:
            # Create __init__.py if it doesn't exist
            init_path = self.algorithm_dir / "__init__.py"
            if not init_path.exists():
                init_path.touch()
                logger.info(f"Created __init__.py at {init_path}")
        except (OSError, PermissionError) as e:
            error_msg = f"Failed to create __init__.py: {e}"
            logger.error(error_msg)
            raise PackageSetupError(error_msg) from e

        # Get the package name from the directory name
        package_name = self.algorithm_dir.name

        try:
            # Add the parent directory to sys.path
            parent_dir = str(self.algorithm_dir.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                logger.info(f"Added {parent_dir} to sys.path")
        except Exception as e:
            error_msg = f"Failed to modify sys.path: {e}"
            logger.error(error_msg)
            raise PackageSetupError(error_msg) from e

        return package_name

    def _find_model_in_module(
        self, module, module_name: str
    ) -> Optional[Type[NeuracoreModel]]:
        """
        Search for NeuracoreModel subclasses in a module.

        Args:
            module: The imported module
            module_name: Name of the module for logging

        Returns:
            Optional[Type[NeuracoreModel]]: Found model class or None
        """
        try:
            for attr_name in dir(module):
                try:
                    module_attr = getattr(module, attr_name)
                    if (
                        isinstance(module_attr, type)
                        and issubclass(module_attr, NeuracoreModel)
                        and module_attr != NeuracoreModel
                    ):
                        logger.info(
                            f"Found model in {module_name}: {module_attr.__name__}"
                        )
                        return module_attr
                except (TypeError, AttributeError) as e:
                    # Some attributes might not be inspectable
                    logger.debug(
                        f"Could not inspect attribute {attr_name} in {module_name}: {e}"
                    )
                    continue
            return None
        except Exception as e:
            logger.warning(f"Error searching for models in {module_name}: {e}")
            return None

    def _try_import_package(self, package_name: str) -> Optional[Type[NeuracoreModel]]:
        """
        Try to import the entire package and find models.

        Args:
            package_name: Name of the package to import

        Returns:
            Optional[Type[NeuracoreModel]]: Found model class or None
        """
        try:
            package = importlib.import_module(package_name)
            logger.info(f"Successfully imported package: {package_name}")
            return self._find_model_in_module(package, package_name)
        except ImportError as e:
            error_msg = (
                f"Failed to import package {package_name}: "
                f"{e}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise ModuleImportError(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error importing package {package_name}: "
                f"{e}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise ModuleImportError(error_msg)

    def _try_import_module_by_path(
        self, file_path: Path, package_name: str
    ) -> Optional[Type[NeuracoreModel]]:
        """
        Try to import a specific module file.

        Args:
            file_path: Path to the Python file
            package_name: Base package name

        Returns:
            Optional[Type[NeuracoreModel]]: Found model class or None
        """
        # First try package-relative import
        try:
            relative_path = file_path.relative_to(self.algorithm_dir.parent)
            module_path = str(relative_path).replace(os.sep, ".")[
                :-3
            ]  # Remove .py extension

            logger.info(f"Attempting package import: {module_path}")
            module = importlib.import_module(module_path)
            found_model = self._find_model_in_module(module, module_path)
            if found_model:
                return found_model

        except ImportError as e:
            logger.error(
                f"Package import failed for {file_path.name}: "
                f"{e}\n{traceback.format_exc()}"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error in package import for {file_path.name}: {e}"
            )

        # Fallback to spec-based import
        try:
            module_name = f"{package_name}_{file_path.stem}"
            logger.info(f"Attempting spec-based import: {module_name}")

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.info(f"Could not create spec for {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            found_model = self._find_model_in_module(module, module_name)
            if found_model:
                logger.info(f"Found model via spec import: {found_model.__name__}")
                return found_model

        except ImportError as e:
            logger.info(f"Spec-based import failed for {file_path.name}: {e}")
        except Exception as e:
            logger.warning(
                f"Unexpected error in spec-based import for {file_path.name}: {e}"
            )

        return None

    def _get_python_files(self) -> List[Path]:
        """
        Get all Python files in the algorithm directory, excluding system files.

        Returns:
            List[Path]: List of Python files to process
        """
        python_files = []
        try:
            for file_path in self.algorithm_dir.glob("**/*.py"):
                if file_path.name == "__init__.py":
                    continue

                # Skip system files
                if file_path.stem.startswith("."):
                    continue

                python_files.append(file_path)
        except Exception as e:
            logger.warning(f"Error scanning Python files: {e}")

        return python_files

    def load_model(self) -> Type[NeuracoreModel]:
        """
        Find and load the first class that inherits from NeuracoreModel.

        This method checks for requirements.txt and installs dependencies,
        then properly sets up the directory as a package to allow
        relative imports between files in the same directory.

        Returns:
            Type[NeuracoreModel]: The first NeuracoreModel subclass found

        Raises:
            RequirementsInstallError: If requirements installation fails
            PackageSetupError: If package setup fails
            ModelNotFoundError: If no NeuracoreModel subclass is found
        """
        # Install requirements if they exist
        self.install_requirements()

        # Set up package environment
        package_name = self._setup_package_environment()

        # Try importing the entire package first
        found_model = self._try_import_package(package_name)
        if found_model:
            return found_model

        # Get all Python files to process
        python_files = self._get_python_files()
        if not python_files:
            raise ModelNotFoundError(f"No Python files found in {self.algorithm_dir}")

        # Try importing individual modules to find NeuracoreModel subclasses
        for file_path in python_files:
            try:
                found_model = self._try_import_module_by_path(file_path, package_name)
                if found_model:
                    return found_model
            except Exception:
                continue

        raise ModelNotFoundError(
            "Could not find a class that inherits from NeuracoreModel. "
            "Ensure your algorithm inherits from NeuracoreModel and "
            "is properly defined."
        )
