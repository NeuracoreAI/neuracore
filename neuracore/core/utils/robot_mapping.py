"""Robot name/id mapping cache for a single organization."""

from __future__ import annotations

import logging

import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL

logger = logging.getLogger(__name__)


class RobotMapping:
    """Singleton class for robot name/id mappings per organization.

    This cache avoids repeated org-wide robot listing calls by keeping
    forward (id -> name) and reverse (name -> id(s)) mappings in memory.
    """

    _instances: dict[tuple[str, bool], RobotMapping] = {}

    def __init__(self, org_id: str, is_shared: bool = False) -> None:
        """Initialize a RobotMapping instance.

        Args:
            org_id: Organization ID for this mapping.
            is_shared: Whether this mapping is for shared robots.
        """
        self._org_id = org_id
        self._is_shared = is_shared
        self._id_to_name: dict[str, str] = {}
        self._name_to_id: dict[str, str] = {}
        self._initialized = False

    @classmethod
    def for_org(cls, org_id: str, is_shared: bool = False) -> RobotMapping:
        """Return the mapping cache for an organization.

        Args:
            org_id: Organization ID to scope the mapping.
            is_shared: Whether to include shared robots.

        Returns:
            The singleton RobotMapping for the org and sharing mode.
        """
        key = (org_id, is_shared)
        if key not in cls._instances:
            cls._instances[key] = cls(org_id, is_shared=is_shared)
        return cls._instances[key]

    def ensure_loaded(self) -> None:
        """Ensure the mapping is loaded from the API."""
        if not self._initialized:
            self.refresh()

    def refresh(self) -> None:
        """Refresh the mapping data from the server.

        On failure, keeps existing data (or initializes empty data if this
        is the first load attempt).
        """
        try:
            response = requests.get(
                f"{API_URL}/org/{self._org_id}/robots",
                headers=get_auth().get_headers(),
                params={"is_shared": self._is_shared},
            )
            response.raise_for_status()
            robots = response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to fetch robot metadata: %s", exc)
            if not self._initialized:
                self._id_to_name = {}
                self._name_to_ids = {}
            return

        id_to_name: dict[str, str] = {}
        name_to_ids: dict[str, list[str]] = {}
        for robot in robots:
            robot_id = robot.get("id") or robot.get("robot_id")
            if not robot_id:
                continue
            robot_name = robot.get("name")
            if robot_name:
                id_to_name[robot_id] = robot_name
                name_to_ids.setdefault(robot_name, []).append(robot_id)
        self._id_to_name = id_to_name
        self._name_to_ids = name_to_ids
        self._initialized = True

    def get_ids_for_name(self, robot_name: str) -> list[str]:
        """Get robot IDs that match a given name.

        Args:
            robot_name: Robot name to look up.

        Returns:
            List of robot IDs that match the name; empty if unknown.
        """
        self.ensure_loaded()
        return list(self._name_to_ids.get(robot_name, []))

    def robot_key_to_id(
        self,
        robot_key: str,
    ) -> str | None:
        """Resolve a robot key (name or ID) to an ID if known.

        Args:
            robot_key: Robot name or robot ID.

        Returns:
            The resolved robot ID if known; otherwise None.

        Raises:
            ValueError: If the robot name is ambiguous or conflicts with a different ID.
        """
        self.ensure_loaded()
        name_matches = self._name_to_ids.get(robot_key, [])
        if len(name_matches) > 1:
            raise ValueError(
                f"Robot name {robot_key} is ambiguous. Use robot_id instead."
            )
        if name_matches:
            name_match_id = name_matches[0]
            if robot_key in self._id_to_name and name_match_id != robot_key:
                raise ValueError(
                    f"Robot key {robot_key} matches both a robot_id and a different "
                    "robot_name. Use robot_id instead."
                )
            return name_match_id
        if robot_key in self._id_to_name:
            return robot_key
        return None

    def robot_key_to_name(
        self,
        robot_key: str,
    ) -> str | None:
        """Resolve a robot key (name or ID) to a name if known.

        Args:
            robot_key: Robot name or robot ID.

        Returns:
            The resolved robot name if known; otherwise None.
        """
        self.ensure_loaded()
        if robot_key in self._id_to_name:
            return self._id_to_name[robot_key]
        if robot_key in self._name_to_ids:
            return robot_key
        return None
