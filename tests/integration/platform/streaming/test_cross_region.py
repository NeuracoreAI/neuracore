"""Providers in different signalling regions are listed and consumed together."""

import neuracore as nc
from tests.integration.platform.streaming.consumer import (
    wait_for_joint_positions,
    wait_for_remote_nodes,
    wait_for_streams,
)
from tests.integration.platform.streaming.nodes import (
    REMOTE_JOINT_VALUE,
    ProviderFactory,
)

INSTANCE = 0


def test_providers_in_every_region_are_listed_and_consumed(
    regions: list[str], robot_name: str, start_provider: ProviderFactory
) -> None:
    """Each region's provider is listed and its data is received."""
    assert len(regions) >= 2, "Cross-region tests need at least two regions."
    robot = nc.connect_robot(robot_name, instance=INSTANCE)
    joint_by_region = {
        region: f"joint_in_region_{index}" for index, region in enumerate(regions)
    }
    for region, joint_name in joint_by_region.items():
        start_provider(robot_name, INSTANCE, region, joint_name)

    streams = wait_for_streams(robot, len(regions))
    wait_for_remote_nodes(robot, node_count=len(regions))
    positions = wait_for_joint_positions(robot, list(joint_by_region.values()))

    assert {stream.region for stream in streams} == set(regions)
    assert set(positions.values()) == {REMOTE_JOINT_VALUE}
