"""Exceptions for Neuracore."""


class EncodingError(Exception):
    """Raised for issues with encoding video."""


class EndpointError(Exception):
    """Raised for endpoint-related errors."""


class AuthenticationError(Exception):
    """Raised for authentication-related errors."""


class ValidationError(Exception):
    """Raised when input validation fails."""


class RobotError(Exception):
    """Raised for robot-related errors."""


class DatasetError(Exception):
    """Exception raised for errors in the dataset module."""


class SynchronizationError(Exception):
    """Exception raised for errors during data synchronization."""


class OrganizationError(Exception):
    """Exception raised for errors gathering organization information."""


class InputError(Exception):
    """Exception raised when the user does not provide valid input."""


class ConfigError(Exception):
    """Exception raised when there is an error attempting to read or write config."""


class InsufficientSyncPointError(Exception):
    """Exception when SyncPoint doesn't contain sufficient data for inference."""

    pass
