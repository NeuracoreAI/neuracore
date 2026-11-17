"""Each robot instance has its own stream, in its own region."""

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


def test_each_instance_is_listed_in_its_own_region_with_its_own_data(
    regions: list[str], robot_name: str, start_provider: ProviderFactory
) -> None:
    """Each robot instance is listed in its own region with its own data."""
    assert len(regions) >= 2, "Multiple instances need at least two regions."
    instance_regions = dict(enumerate(regions))
    robots = {
        instance: nc.connect_robot(robot_name, instance=instance)
        for instance in instance_regions
    }
    for instance, region in instance_regions.items():
        start_provider(robot_name, instance, region, f"joint_of_instance_{instance}")

    for instance, region in instance_regions.items():
        robot = robots[instance]
        [stream] = wait_for_streams(robot, 1)
        wait_for_remote_nodes(robot, node_count=1)
        positions = wait_for_joint_positions(robot, [f"joint_of_instance_{instance}"])

        assert stream.region == region
        assert positions[f"joint_of_instance_{instance}"] == REMOTE_JOINT_VALUE
