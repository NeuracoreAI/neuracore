from getpass import getpass  # For hidden password input

import requests

from .const import API_URL


def generate_api_key():
    # Prompt the user for credentials
    email = input("Enter your registered email: ")
    password = getpass("Enter your password: ")
    try:
        auth_response = requests.post(
            f"{API_URL}/auth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"username": email, "password": password},
        )
        auth_response.raise_for_status()
        token_data = auth_response.json()
        access_token = token_data.get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Authentication failed: {e}")
        return

    # Use the access token to request an API key
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        api_key_response = requests.post(
            f"{API_URL}/auth/api-key",
            json={"name": "CLI API Key"},  # Name can be customized
            headers=headers,
        )
        api_key_response.raise_for_status()
        api_key = api_key_response.json().get("key")
        print(f"Your new API key is: {api_key}")
        return api_key
    except requests.exceptions.RequestException as e:
        print(f"Failed to create API key: {e}")


if __name__ == "__main__":
    generate_api_key()
