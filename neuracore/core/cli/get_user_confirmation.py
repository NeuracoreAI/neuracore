"""This module provides a method for confirming with the user."""

from neuracore.core.const import CONFIRMATION_INPUT, MAX_INPUT_ATTEMPTS, REJECTION_INPUT
from neuracore.core.exceptions import InputError


def get_user_confirmation(prompt: str) -> bool:
    """Gets a users confirmation to a prompt.

    The prompt should be a yes or no question i.e. "do spiders have legs?"

    Args:
        prompt (str): a string with a yes or no question

    Raises:
        InputError: if the user does not provide a valid input

    Returns:
        bool: true if the user responds positively
    """
    for i in range(MAX_INPUT_ATTEMPTS):
        confirm = input(f"{prompt} (y/n): ").lower().strip()
        if confirm in CONFIRMATION_INPUT:
            return True
        if confirm in REJECTION_INPUT:
            return False
        if i + 1 < MAX_INPUT_ATTEMPTS:
            print("We did not recognize that input, please try again.")
        else:
            print("We did not recognize that input, you are out of attempts.")

    raise InputError("Out of attempts")
