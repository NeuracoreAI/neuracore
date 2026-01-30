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


class DataValidationError(UploaderError):
    """Raised when data fails validation and should skip the episode."""


class DataValidationWarning(UploaderError):
    """Raised when data fails validation and should log a warning but continue."""


class DatasetDetectionError(UploaderError):
    """Raised when dataset type cannot be determined."""


class DatasetOperationError(UploaderError):
    """Raised when dataset CRUD operations fail."""
