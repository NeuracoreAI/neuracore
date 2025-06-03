"""Exceptions for Neuracore."""


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
