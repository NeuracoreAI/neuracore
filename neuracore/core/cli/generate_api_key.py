"""Interactive API key generation for Neuracore authentication.

This module provides functionality for generating API keys through interactive
user authentication. It prompts for email and password credentials, obtains
an access token, and creates a new API key for CLI usage.
"""

import argparse
from getpass import getpass
from typing import Optional  # For hidden password input

import requests
from pydantic import BaseModel, ValidationError

from neuracore.core.cli.get_user_confirmation import get_user_confirmation
from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.exceptions import AuthenticationError, InputError

from ..const import API_URL, MAX_INPUT_ATTEMPTS


class APIKey(BaseModel):
    """API key model."""

    key: str


class Token(BaseModel):
    """OAuth2 token response model."""

    access_token: str
    token_type: str


def generate_api_key(
    email: Optional[str] = None, password: Optional[str] = None
) -> str:
    """Generate a new API key through interactive user authentication.

    Prompts the user for their registered email and password, authenticates
    with the Neuracore server to obtain an access token, then uses that token
    to create a new API key for programmatic access. The process is interactive
    and handles authentication securely by hiding password input.

    Args:
        email: Optionally provide the email to try initially
        password: Optionally provide the password to try initially

    Returns:
        The generated API key string if successful, None if authentication
        or API key generation fails.

    Raises:
        AuthenticationError: If API key verification fails due to invalid
            credentials, network issues, or server errors.
        InputError: If there is an issue with the user's input
    """
    # Prompt the user for credentials
    access_token = None
    for i in range(MAX_INPUT_ATTEMPTS):
        try:
            if not email:
                email = input("Enter your registered email: ")
            if not password:
                password = getpass("Enter your password: ")
            auth_response = requests.post(
                f"{API_URL}/auth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"username": email, "password": password},
            )
            if auth_response.status_code == 401 and i + 1 < MAX_INPUT_ATTEMPTS:
                print("Incorrect email or password.")

                again = get_user_confirmation("Do you wish to try again?")
                if again:
                    email = None
                    password = None
                    continue
                else:
                    raise InputError("Invalid Email or Password")

            auth_response.raise_for_status()
            token_data = Token.model_validate(auth_response.json())
            access_token = token_data.access_token
            break
        except KeyboardInterrupt:
            raise InputError("User cancelled the operation.")
        except ValidationError:
            raise AuthenticationError("Invalid token from server")
        except requests.exceptions.ConnectionError:
            raise AuthenticationError((
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Failed to get Auth Token: {e}")

    if not access_token:
        print("Too many failed attempts, perhaps check your Caps Lock?")
        raise InputError("Out of attempts")

    # Use the access token to request an API key
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        api_key_response = requests.get(
            f"{API_URL}/auth/api-key",
            headers=headers,
        )
        api_key_response.raise_for_status()
        api_key = APIKey.model_validate(api_key_response.json()).key
        print(f"Your API key is: {api_key}")
        config_manager = get_config_manager()
        config_manager.config.api_key = api_key
        config_manager.save_config()
        return api_key
    except ValidationError:
        raise AuthenticationError("Invalid api-key from server")
    except requests.exceptions.ConnectionError:
        raise AuthenticationError((
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        ))
    except requests.exceptions.RequestException as e:
        raise AuthenticationError(f"Failed to create API key: {e}")


def main() -> None:
    """Main function to run the API key generation process."""
    try:
        parser = argparse.ArgumentParser(
            description="Generate a new API key for access to the neuracore platform.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--email",
            "--username",
            "-e",
            dest="email",
            required=False,
            type=str,
            help="The email to login with.",
        )
        parser.add_argument(
            "--password",
            "-p",
            dest="password",
            required=False,
            type=str,
            help="The password to login with.",
        )
        args = parser.parse_args()
        generate_api_key(email=args.email, password=args.password)
    except AuthenticationError:
        print("Failed to generate API key, please try again later")
    except InputError:
        print("Failed to generate API key due to incorrect input")


if __name__ == "__main__":
    main()
