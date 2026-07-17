"""Exceptions for Neuracore."""

from typing import Any


class EncodingError(Exception):
    """Raised for issues with encoding video."""


class EndpointError(Exception):
    """Raised for endpoint-related errors."""


class AuthenticationError(Exception):
    """Raised for authentication-related errors."""

    def __init__(
        self, message: str = "", *, response_payload: dict[str, Any] | None = None
    ):
        """Create an authentication error.

        Args:
            message: User-facing error message.
            response_payload: Optional structured backend response body.
        """
        super().__init__(message)
        self.response_payload = response_payload or {}

    @property
    def required_version(self) -> str | None:
        """Backend-required client version, if the response included one."""
        required_version = self.response_payload.get("required_version")

        detail = self.response_payload.get("detail")
        if required_version is None and isinstance(detail, dict):
            required_version = detail.get("required_version")

        if isinstance(required_version, str):
            return required_version
        return None


class VersionMismatchError(AuthenticationError):
    """Raised when the client version is incompatible with the server."""


class ValidationError(Exception):
    """Raised when input validation fails."""


class RobotError(Exception):
    """Raised for robot-related errors."""


class RecordingError(Exception):
    """Raised for recording lifecycle errors."""


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


class InsufficientSynchronizedPointError(Exception):
    """Error when SynchronizedPoint contain insufficient data for inference."""

    pass


class TrainingRunError(Exception):
    """Exception raised for errors related to training runs."""

    pass


class RobotMismatchError(ValueError):
    """Raised when inference robot_id is not in the model's training spec."""

    pass
