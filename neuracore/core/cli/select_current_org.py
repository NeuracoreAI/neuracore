"""Interactive Organization selection.

This module provides functionality for setting the organization used by other
neuracore operations interactively. It provides a list of their organizations
and a input to select which to use and saves to global config.
"""

import argparse
import os
from typing import Optional

from neuracore.core.auth import get_auth
from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.exceptions import AuthenticationError, InputError, OrganizationError
from neuracore.core.organizations import Organization, list_my_orgs

from ..const import MAX_INPUT_ATTEMPTS, REJECTION_INPUT


def select_current_org(org_name_or_id: Optional[str] = None) -> Organization:
    """Prompt the user to select on of their organizations.

    Args:
        org_name_or_id: The name or id of the organization to select. If not
            provided or not found, the user will be prompted to select one.

    Returns:
        The selected organization's name and id

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

    if org_name_or_id is not None:
        pre_selected_org = next(
            (
                org
                for org in orgs
                if org.name.strip() == org_name_or_id.strip()
                or org.id == org_name_or_id.strip()
            ),
            None,
        )
        if pre_selected_org is not None:
            return pre_selected_org
        else:
            print(f"Organization not found with name or id: {org_name_or_id}")

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
            input_raw = input(input_prompt)
            if input_raw.lower().strip() in REJECTION_INPUT:
                raise InputError("Exited")
            input_selection = int(input_raw)
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
        except InputError:
            raise
        except Exception as e:
            print(e)
    print("Out of attempts.")
    raise InputError("Out of attempts.")


def main() -> None:
    """Main function to run the organization selection process."""
    parser = argparse.ArgumentParser(
        description="Select an organization to use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--org-name",
        "--org-id",
        "-n",
        "-o",
        dest="org_name_or_id",
        required=False,
        type=str,
        help="The name or id of the organization to select.",
    )
    org_name_or_id = parser.parse_args().org_name_or_id

    auth = get_auth()
    try:
        if not auth.is_authenticated:
            auth.login()

        organization = select_current_org(org_name_or_id=org_name_or_id)
        config_manager = get_config_manager()
        config_manager.config.current_org_id = organization.id
        config_manager.save_config()

    except AuthenticationError:
        print("Failed to Authenticate, please try again")
    except OrganizationError:
        print("Failed to select organization, please try again")
    except InputError:
        print("No organization selected")


if __name__ == "__main__":
    main()
