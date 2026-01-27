"""Exception classes for uploader workflow."""


class UploaderError(Exception):
    """Base error for uploader workflow."""


class ImportError(UploaderError):
    """Base error for importer workflow."""


class CLIError(UploaderError):
    """Raised when CLI arguments are invalid."""


class ConfigLoadError(UploaderError):
    """Raised when dataset config cannot be loaded."""


class ConfigValidationError(UploaderError):
    """Raised when dataset config fails validation."""

    def __init__(self, errors: list[str]):
        """Initialize ConfigValidationError with list of error messages.

        Args:
            errors: List of error messages from validation.
        """
        message = "\n".join(errors)
        super().__init__(message)
        self.errors = errors


class DatasetDetectionError(UploaderError):
    """Raised when dataset type cannot be determined."""


class DatasetOperationError(UploaderError):
    """Raised when dataset CRUD operations fail."""
