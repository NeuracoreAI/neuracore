class NeuraCoreError(Exception):
    """Base exception class for all NeuraCore errors."""

    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)


class EndpointError(NeuraCoreError):
    """Raised for endpoint-related errors.

    Examples:
        - Endpoint not found
        - Endpoint not active
        - Prediction failed
        - Invalid response format
    """

    pass


class AuthenticationError(NeuraCoreError):
    """Raised for authentication-related errors.

    Examples:
        - No API key provided
        - Invalid API key
        - Authentication server unreachable
        - Session expired
    """

    pass


class ValidationError(NeuraCoreError):
    """Raised when input validation fails.

    Examples:
        - Invalid URDF file
        - Missing required mesh files
        - Invalid image format
        - Invalid joint names
    """

    pass


class RobotError(NeuraCoreError):
    """Raised for robot-related errors.

    Examples:
        - Robot not initialized
        - Robot disconnected
        - Invalid robot ID
        - Robot already exists
    """

    pass


class DatasetError(Exception):
    """Exception raised for errors in the dataset module."""

    pass
