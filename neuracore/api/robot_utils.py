"""Robot helper utilities for the public API layer."""

from neuracore.core.exceptions import RobotError
from neuracore.core.robot import (
    Robot,
    get_robot,
    get_robot_id_from_name,
    register_existing_robot,
)


def _try_get_local_robot(robot_name: str, instance: int) -> Robot | None:
    try:
        return get_robot(robot_name, instance)
    except RobotError:
        return None


def _try_get_robot_id_from_name(robot_name: str) -> str | None:
    try:
        return get_robot_id_from_name(robot_name)
    except RobotError:
        return None


def get_existing_robot_for_create(
    robot_name: str, instance: int, shared: bool, exist_ok: bool
) -> tuple[Robot | None, str | None]:
    """Return an existing robot for create_robot, plus the duplicate error."""
    robot = _try_get_local_robot(robot_name, instance)
    if robot is not None:
        return robot, (
            f"Robot '{robot_name}' with instance '{instance}' already exists. "
            "Call connect_robot() to connect to the existing robot."
        )

    robot_id = _try_get_robot_id_from_name(robot_name)
    if robot_id is None:
        return None, None

    robot = (
        register_existing_robot(robot_name, robot_id, instance, shared)
        if exist_ok
        else None
    )
    return robot, (
        f"Robot '{robot_name}' already exists in Neuracore. "
        "Call connect_robot() to connect to the existing robot."
    )
