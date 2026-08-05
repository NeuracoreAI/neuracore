"""Integration coverage for organization-scoped SDK resources."""

import logging
import os
from uuid import uuid4

import pytest

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError, RobotError
from neuracore.core.organizations import Organization, list_my_orgs
from neuracore.core.robot import get_robot_id_from_name
from neuracore.core.utils.http_session import thread_local_session

logger = logging.getLogger(__name__)

ROBOT_NAME = "multi-org-isolation-robot"


def _two_organizations() -> tuple[Organization, Organization]:
    """Return two writable organizations or skip this environment."""
    if "NEURACORE_ORG_ID" in os.environ:
        pytest.skip("NEURACORE_ORG_ID pins the SDK and prevents organization switching")

    nc.login()
    organizations = list_my_orgs()
    if len(organizations) < 2:
        pytest.skip("Multi-organization isolation requires access to at least two orgs")
    return organizations[0], organizations[1]


def _delete_dataset(org: Organization, dataset_id: str | None) -> None:
    if dataset_id is None:
        return
    try:
        nc.set_organization(org.id)
        dataset = Dataset.get_by_id(dataset_id, non_exist_ok=True)
        if dataset is not None and not dataset.deleted:
            dataset.delete()
    except Exception:  # noqa: BLE001 - cleanup must preserve the test failure
        logger.warning("Failed to delete dataset %s in org %s", dataset_id, org.id)


def _list_endpoint_ids() -> list[str]:
    """List endpoint IDs in the selected org for cross-org lookup coverage."""
    response = thread_local_session().get(
        f"{API_URL}/org/{get_current_org()}/models/endpoints",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()
    return [endpoint["id"] for endpoint in response.json()]


def test_resources_and_active_caches_are_isolated_between_organizations() -> None:
    """Switch orgs in-process without leaking resources or active SDK state."""
    org_a, org_b = _two_organizations()
    original_org_id = get_current_org()
    dataset_name = f"multi-org-isolation-{uuid4().hex}"
    dataset_ids: dict[str, str | None] = {org_a.id: None, org_b.id: None}
    robots = []

    try:
        nc.set_organization(org_a.id)
        dataset_a = nc.create_dataset(dataset_name)
        dataset_ids[org_a.id] = dataset_a.id
        robot_a = nc.connect_robot(ROBOT_NAME)
        robots.append(robot_a)
        assert GlobalSingleton()._active_dataset_id == dataset_a.id
        assert GlobalSingleton()._active_robot is robot_a

        breakpoint()

        nc.set_organization(org_b.id)
        assert GlobalSingleton()._active_dataset_id is None
        assert GlobalSingleton()._active_dataset is None
        assert GlobalSingleton()._active_robot is None
        with pytest.raises(DatasetError):
            nc.get_dataset(id=dataset_a.id)
        with pytest.raises(RuntimeError):
            dataset_a.set_description("must not cross the organization boundary")
        with pytest.raises(RobotError):
            nc.update_robot_name(robot_a.id, f"{ROBOT_NAME}-forbidden")

        breakpoint()

        dataset_b = nc.create_dataset(dataset_name)
        dataset_ids[org_b.id] = dataset_b.id
        robot_b = nc.connect_robot(ROBOT_NAME)
        robots.append(robot_b)
        assert dataset_b.id != dataset_a.id
        assert robot_b.id != robot_a.id
        assert nc.get_dataset(name=dataset_name).id == dataset_b.id
        assert get_robot_id_from_name(ROBOT_NAME) == robot_b.id

        breakpoint()

        nc.set_organization(org_a.id)
        assert GlobalSingleton()._active_dataset_id is None
        assert GlobalSingleton()._active_dataset is None
        assert GlobalSingleton()._active_robot is None
        assert nc.get_dataset(name=dataset_name).id == dataset_a.id
        assert get_robot_id_from_name(ROBOT_NAME) == robot_a.id
    finally:
        for robot in robots:
            robot.close()
        _delete_dataset(org_a, dataset_ids[org_a.id])
        _delete_dataset(org_b, dataset_ids[org_b.id])
        nc.set_organization(original_org_id)


def test_training_jobs_and_endpoints_cannot_be_read_from_another_org() -> None:
    """Existing ML resource IDs must not resolve outside their owning org."""
    org_a, org_b = _two_organizations()
    original_org_id = get_current_org()
    assertions = 0

    try:
        nc.set_organization(org_a.id)
        jobs_a = nc.get_training_jobs()
        endpoints_a = _list_endpoint_ids()

        nc.set_organization(org_b.id)
        if jobs_a:
            with pytest.raises(ValueError, match="Job not found"):
                nc.get_training_job_data(jobs_a[0]["id"])
            assertions += 1
        if endpoints_a:
            with pytest.raises(ValueError):
                nc.get_endpoint_status(endpoints_a[0])
            assertions += 1

        if assertions == 0:
            pytest.skip("Org A has no existing training jobs or endpoints to probe")
    finally:
        nc.set_organization(original_org_id)
