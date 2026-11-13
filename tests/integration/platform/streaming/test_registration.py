"""A provider registers in a signalling region and a consumer lists it there."""

import neuracore as nc
from neuracore.core.streaming.p2p.consumer.regional_availability import LEGACY_SOURCE
from tests.integration.platform.streaming.consumer import (
    wait_for_joint_positions,
    wait_for_remote_nodes,
    wait_for_streams,
)
from tests.integration.platform.streaming.nodes import (
    REMOTE_JOINT_NAME,
    REMOTE_JOINT_VALUE,
    ProviderFactory,
)

INSTANCE = 0


def test_pinned_provider_is_listed_in_its_region(
    regions: list[str], robot_name: str, start_provider: ProviderFactory
) -> None:
    """A provider pinned to one region is listed in that region at a set epoch."""
    robot = nc.connect_robot(robot_name, instance=INSTANCE)
    region = regions[0]
    start_provider(robot_name, INSTANCE, region)

    [stream] = wait_for_streams(robot, 1)

    assert stream.region == region
    assert stream.epoch is not None and stream.epoch > 0


def test_provider_through_the_shared_endpoint_is_listed_in_a_region(
    regions: list[str], robot_name: str, start_provider: ProviderFactory
) -> None:
    """A provider on the shared endpoint is listed in a region, not legacy."""
    robot = nc.connect_robot(robot_name, instance=INSTANCE)
    start_provider(robot_name, INSTANCE, None)

    [stream] = wait_for_streams(robot, 1)

    assert stream.region != LEGACY_SOURCE
    assert stream.region in regions


def test_consumer_receives_data_through_the_region(
    regions: list[str], robot_name: str, start_provider: ProviderFactory
) -> None:
    """A consumer receives a provider's data through its region."""
    robot = nc.connect_robot(robot_name, instance=INSTANCE)
    region = regions[0]
    start_provider(robot_name, INSTANCE, region)

    [stream] = wait_for_streams(robot, 1)
    wait_for_remote_nodes(robot, node_count=1)
    positions = wait_for_joint_positions(robot, [REMOTE_JOINT_NAME])

    assert stream.region == region
    assert positions[REMOTE_JOINT_NAME] == REMOTE_JOINT_VALUE
