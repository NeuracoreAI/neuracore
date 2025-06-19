"""This module provide a method for reading the current api key."""

from neuracore.core.cli.generate_api_key import generate_api_key
from neuracore.core.cli.input_lock import user_input_lock
from neuracore.core.config.config_manager import get_config


def get_api_key() -> str:
    """Get the api key from storage or generate and save a new one.

    Returns:
        The new api key

    Raises:
        AuthenticationError: If API key verification fails due to invalid
            credentials, network issues, or server errors.
        InputError: If there is an issue with the user's input when gathering
            credentials
    """
    with user_input_lock:
        api_key = get_config().api_key
        if api_key:
            return api_key

        print("No API key provided. Attempting to log you in...")
        return generate_api_key()
