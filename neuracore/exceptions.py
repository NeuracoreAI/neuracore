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


class StreamingError(NeuraCoreError):
    """Raised for data streaming errors.

    Examples:
        - Connection lost
        - Failed to send data
        - Queue overflow
        - Invalid data format
    """

    pass


class ConfigurationError(NeuraCoreError):
    """Raised for configuration-related errors.

    Examples:
        - Invalid configuration file
        - Missing required settings
        - Invalid file paths
    """

    pass


class RateLimitError(NeuraCoreError):
    """Raised when API rate limits are exceeded.

    Examples:
        - Too many requests
        - Bandwidth limit exceeded
        - Too many concurrent connections
    """

    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


class ResourceNotFoundError(NeuraCoreError):
    """Raised when a requested resource is not found.

    Examples:
        - URDF file not found
        - Mesh file not found
        - Robot not found
        - Dataset not found
    """

    pass


class ResourceExistsError(NeuraCoreError):
    """Raised when attempting to create a resource that already exists.

    Examples:
        - Robot ID already in use
        - Dataset already exists
        - Duplicate recording ID
    """

    pass


class ServerError(NeuraCoreError):
    """Raised for server-side errors.

    Examples:
        - Internal server error
        - Service unavailable
        - Database error
        - Storage error
    """

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class DataError(NeuraCoreError):
    """Raised for data-related errors.

    Examples:
        - Invalid data format
        - Data corruption
        - Incompatible data types
        - Missing required data fields
    """

    pass


class NetworkError(NeuraCoreError):
    """Raised for network-related errors.

    Examples:
        - Connection timeout
        - Network unreachable
        - DNS resolution failed
        - SSL/TLS errors
    """

    pass


class QuotaExceededError(NeuraCoreError):
    """Raised when account quotas are exceeded.

    Examples:
        - Storage quota exceeded
        - Bandwidth quota exceeded
        - API calls quota exceeded
        - Maximum robots limit reached
    """

    def __init__(
        self, message: str, limit: str, current_usage: float, max_allowed: float
    ):
        super().__init__(message)
        self.limit = limit
        self.current_usage = current_usage
        self.max_allowed = max_allowed


class PermissionError(NeuraCoreError):
    """Raised for permission-related errors.

    Examples:
        - Insufficient permissions
        - Invalid credentials
        - Account suspended
        - Resource access denied
    """

    pass


class VersionError(NeuraCoreError):
    """Raised for version compatibility errors.

    Examples:
        - API version mismatch
        - Protocol version incompatible
        - URDF version unsupported
        - Client version outdated
    """

    def __init__(
        self, message: str, required_version: str = None, current_version: str = None
    ):
        super().__init__(message)
        self.required_version = required_version
        self.current_version = current_version


def convert_exception(error: Exception) -> NeuraCoreError:
    """Convert standard Python exceptions to NeuraCore exceptions."""
    if isinstance(error, ConnectionError):
        return NetworkError(f"Network error: {str(error)}")
    elif isinstance(error, TimeoutError):
        return NetworkError(f"Connection timeout: {str(error)}")
    elif isinstance(error, FileNotFoundError):
        return ResourceNotFoundError(f"File not found: {str(error)}")
    elif isinstance(error, PermissionError):
        return PermissionError(f"Permission denied: {str(error)}")
    elif isinstance(error, ValueError):
        return ValidationError(f"Invalid value: {str(error)}")
    else:
        return NeuraCoreError(f"Unexpected error: {str(error)}")
