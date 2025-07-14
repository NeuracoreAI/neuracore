"""Interactive Organization selection.

This module provides functionality for setting the organization used by other
neuracore operations interactively. It provides a list of their organizations
and a input to select which to use and saves to global config.
"""

import os

from neuracore.core.auth import get_auth
from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.exceptions import AuthenticationError, InputError, OrganizationError
from neuracore.core.organizations import Organization, list_my_orgs

from ..const import MAX_INPUT_ATTEMPTS


def select_current_org() -> Organization:
    """Prompt the user to select on of their organizations.

    Returns:
        The  selected organization's name and id

    Raises:
        AuthenticationError: If the user is not logged in
        OrganizationError: If there is an issue contacting the backend
        InputError: If the user fails to select a organization
    """
    orgs = list_my_orgs()

    if len(orgs) == 0:
        print(
            "You are not a member of any organizations, please create an "
            "organization first"
        )
        raise OrganizationError("No organizations")

    elif len(orgs) == 1:
        print(f"One organization found, using organization:{os.linesep}{orgs[0].name}")
        return orgs[0]

    org_list = os.linesep.join(
        f"  {index+1}) {org.name}" for index, org in enumerate(orgs)
    )
    orgs_prompt = f"Please select which organization to use:{os.linesep*2}{org_list}"
    max_input = len(orgs)
    input_prompt = f"(1-{max_input}): "
    for i in range(MAX_INPUT_ATTEMPTS):
        try:
            print(orgs_prompt)
            input_selection = int(input(input_prompt))
            if input_selection < 1:
                raise ValueError()
            if input_selection > max_input:
                raise ValueError()

            organization = orgs[input_selection - 1]
            print(f"You have selected the organization: {organization.name}")
            return organization
        except ValueError:
            print(
                "Invalid Input: please provide a valid integer between "
                f"1 and {max_input}"
            )
            if i + 1 < MAX_INPUT_ATTEMPTS:
                print("Please try again.")
        except KeyboardInterrupt:
            raise InputError("User cancelled the operation.")
        except Exception as e:
            print(e)
    print("Out of attempts.")
    raise InputError("Out of attempts.")


def main() -> None:
    """Main function to run the organization selection process."""
    auth = get_auth()
    try:
        if not auth.is_authenticated:
            auth.login()
        organization = select_current_org()
        config_manager = get_config_manager()
        config_manager.config.current_org_id = organization.id
        config_manager.save_config()

    except AuthenticationError:
        print("Failed to Authenticate, please try again")
    except (OrganizationError, InputError):
        print("Failed to select organization")


if __name__ == "__main__":
    main()
