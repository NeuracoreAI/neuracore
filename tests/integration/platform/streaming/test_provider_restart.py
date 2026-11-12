"""A provider that stops leaves the list, and a restarted provider is listed again."""

import neuracore as nc
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
RESTARTED_JOINT_NAME = "joint_after_restart"


def test_stopped_provider_leaves_the_list_and_a_restart_is_listed_again(
    regions: list[str], robot_name: str, start_provider: ProviderFactory
) -> None:
    """A stopped provider leaves the list and its restart is listed again."""
    robot = nc.connect_robot(robot_name, instance=INSTANCE)
    region = regions[0]
    first_provider = start_provider(robot_name, INSTANCE, region, REMOTE_JOINT_NAME)
    [first_stream] = wait_for_streams(robot, 1)

    first_provider.stop()
    wait_for_streams(robot, 0)

    start_provider(robot_name, INSTANCE, region, RESTARTED_JOINT_NAME)
    [restarted_stream] = wait_for_streams(robot, 1)
    wait_for_remote_nodes(robot, node_count=1)
    positions = wait_for_joint_positions(robot, [RESTARTED_JOINT_NAME])

    assert restarted_stream.region == region
    assert restarted_stream.epoch is not None and first_stream.epoch is not None
    assert restarted_stream.epoch > first_stream.epoch
    assert positions[RESTARTED_JOINT_NAME] == REMOTE_JOINT_VALUE
