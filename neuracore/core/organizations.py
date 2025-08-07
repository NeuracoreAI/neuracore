"""This module provides a method for listing the current user's organizations."""

import requests
from pydantic import BaseModel, ValidationError

from neuracore.core.auth import get_auth
from neuracore.core.exceptions import AuthenticationError, OrganizationError

from .const import API_URL


class Organization(BaseModel):
    """Model of the organization information provided from the my-orgs list."""

    id: str
    name: str


class OrgWithMembership(BaseModel):
    """Model of the org and membership information."""

    org: Organization


def list_my_orgs() -> list[Organization]:
    """Get the list of organizations the user is currently a member of.

    Returns:
        list of organizations the user is a member of

    Raises:
        AuthenticationError: If the user is not logged in
        OrganizationError: If there is an issue contacting the backend
    """
    auth = get_auth()
    try:
        org_response = requests.get(
            f"{API_URL}/org-management/my-orgs", headers=auth.get_headers()
        )
        org_response.raise_for_status()
        orgs_raw = org_response.json()
        assert isinstance(
            orgs_raw, list
        ), "/org-management/my-orgs must return list of organizations"
        return [
            OrgWithMembership.model_validate(orgWithMember).org
            for orgWithMember in orgs_raw
        ]
    except requests.exceptions.ConnectionError:
        raise OrganizationError((
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        ))
    except requests.exceptions.RequestException as e:
        raise OrganizationError(f"Failed to get organizations: {e}")
    except AuthenticationError as e:
        if not auth.is_authenticated:
            print("Your not logged in, please log in to see current organizations")
        raise e
    except ValidationError:
        raise OrganizationError(
            "Failed to get organizations: Invalid response from server"
        )
