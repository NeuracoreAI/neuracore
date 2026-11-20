"""Waiting for the server to confirm a robot or dataset the daemon holds.

``nc.connect_robot``, ``nc.create_dataset``, and ``nc.get_dataset`` by name
after an offline login, go through the data daemon: it keeps a row for the
robot instance or dataset and has the server confirm it in the background.
The SDK uses the row's index; the server's id is only needed by calls that
talk to the server, which wait for it here.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from neuracore.core.auth import get_auth
from neuracore.core.exceptions import (
    DatasetError,
    RobotArchivedError,
    RobotError,
    ServerUnreachableError,
)
from neuracore.data_daemon import bridge
from neuracore.data_daemon.bridge import RowState

# The longest a call waits for the server to confirm a robot or dataset.
ROW_WAIT_TIMEOUT_S = 60.0
# How often the daemon is asked whether the row is confirmed.
ROW_POLL_INTERVAL_S = 0.2

# What each reason the daemon stores means, and what the user can do.
_REASON_MESSAGES = {
    "not_in_org": (
        "the account cannot record in the organization it was made in. Log in "
        "with an account that can record there"
    ),
    "credentials": (
        "the API key was rejected or is missing. Run `neuracore login` and try again"
    ),
    "name_taken": (
        "a dataset of the other kind, shared or private, already has this name. "
        "Rename or delete it on the Neuracore dashboard"
    ),
    "robot_archived": (
        "the robot is archived. Unarchive it on the Archived Robots page of the "
        "Neuracore dashboard"
    ),
    "shared_not_allowed": (
        "the account may not create shared datasets. Email contact@neuracore.com "
        "to request access"
    ),
    "rejected": "the server rejected it",
}


def explain_reason(reason: str) -> str:
    """What a stored reason code means, in words."""
    return _REASON_MESSAGES.get(reason, reason)


def wait_for_dataset_id(dataset_index: int, name: str) -> str:
    """Wait for the server to confirm a dataset row and return its id.

    Args:
        dataset_index: The dataset row's index.
        name: The dataset's name, for messages.

    Returns:
        The server's id for the dataset.

    Raises:
        ServerUnreachableError: If the last login could not reach the server.
        DatasetError: If the server refused the dataset, the user must fix
            something first, or it was not confirmed within
            :data:`ROW_WAIT_TIMEOUT_S`.
    """

    def error(message: str, reason: str | None = None) -> Exception:
        return DatasetError(f"Dataset '{name}': {message}.")

    return _wait_for_row(
        lambda: bridge.dataset_row_state(dataset_index), f"dataset '{name}'", error
    )


def wait_for_robot_id(robot_index: int, name: str, instance: int) -> str:
    """Wait for the server to confirm a robot row and return its id.

    Args:
        robot_index: The robot row's index.
        name: The robot's name, for messages.
        instance: The robot instance, for messages.

    Returns:
        The server's id for the robot.

    Raises:
        ServerUnreachableError: If the last login could not reach the server.
        RobotArchivedError: If the robot is archived.
        RobotError: If the server refused the robot, the user must fix
            something first, or it was not confirmed within
            :data:`ROW_WAIT_TIMEOUT_S`.
    """
    describe = f"robot '{name}' instance {instance}"

    def error(message: str, reason: str | None = None) -> Exception:
        if reason == "robot_archived":
            return RobotArchivedError(
                f"Robot '{name}' is archived. Unarchive it on the Archived Robots "
                "page of the Neuracore dashboard, then connect again."
            )
        return RobotError(f"Robot '{name}': {message}.")

    return _wait_for_row(lambda: bridge.robot_row_state(robot_index), describe, error)


def _wait_for_row(
    read_state: Callable[[], RowState | None],
    describe: str,
    error: Callable[[str, str | None], Exception],
) -> str:
    """Poll a row until the server confirms it, then return its server id."""
    if get_auth().is_offline:
        raise ServerUnreachableError(
            f"Neuracore cannot be reached, so the server has not confirmed "
            f"{describe}. It is kept on this machine and registered once the "
            "server can be reached; pass wait=False to carry on offline."
        )
    deadline = time.monotonic() + ROW_WAIT_TIMEOUT_S
    while True:
        state = read_state()
        if state is None:
            raise error("the data daemon has no record of it", None)
        if state.status == "done" and state.cloud_id:
            return state.cloud_id
        if state.reason is not None:
            raise error(explain_reason(state.reason), state.reason)
        if time.monotonic() >= deadline:
            raise error(
                f"the server did not confirm it within {ROW_WAIT_TIMEOUT_S:.0f} s", None
            )
        time.sleep(ROW_POLL_INTERVAL_S)
