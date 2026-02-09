"""Exception classes for importer workflow."""


class ImporterError(Exception):
    """Base error for importer workflow."""


class ImportError(ImporterError):
    """Base error for importer workflow."""


class CLIError(ImporterError):
    """Raised when CLI arguments are invalid."""


class ConfigLoadError(ImporterError):
    """Raised when dataset config cannot be loaded."""


class ConfigValidationError(ImporterError):
    """Raised when dataset config fails validation."""


class DataValidationError(ImporterError):
    """Raised when data fails validation and should skip the episode."""


class DataValidationWarning(ImporterError):
    """Raised when data fails validation and should log a warning but continue."""


class DatasetDetectionError(ImporterError):
    """Raised when dataset type cannot be determined."""


class DatasetOperationError(ImporterError):
    """Raised when dataset CRUD operations fail."""
