from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, overload

from neuracore.core.const import MAX_DATA_STREAMS

from ..core.robot import Robot
from ..core.streaming.data_stream import DataStream


class GlobalSingleton(object):
    _instance = None
    _has_validated_version = False
    _active_robot: Optional[Robot] = None
    _active_dataset_id: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


@dataclass(slots=True)
class DatastreamStore:
    instance: int
    robot_id: str
    __data_streams: dict[str, DataStream] = field(default_factory=dict)

    def add_data_stream(self, stream_id: str, stream: DataStream):
        if len(self.__data_streams) >= MAX_DATA_STREAMS:
            raise RuntimeError("Excessive number of data streams")
        if stream_id in self.__data_streams:
            raise ValueError("Stream already exists")
        self.__data_streams[stream_id] = stream

    def get_data_stream(self, stream_id: str) -> DataStream:
        if stream_id not in self.__data_streams:
            raise ValueError(
                f"Stream {stream_id} not found for robot {self.robot_id} instance {self.instance}"
            )
        return self.__data_streams[stream_id]

    def list_all_streams(self) -> dict[str, DataStream]:
        """List all data streams for a given robot."""
        return self.__data_streams


_robot_data_stream_store: dict[tuple[str, int], DatastreamStore] = dict()


@overload
def get_data_stream_store(robot: Robot) -> DatastreamStore:
    pass


@overload
def get_data_stream_store(robot_name: str, instance: int) -> DatastreamStore:
    pass


def get_data_stream_store(
    robot: Robot | str, instance: Optional[int] = None
) -> DatastreamStore:
    """Get the data stream store for a given robot."""
    if isinstance(robot, Robot):
        robot_id = robot.id
        instance = robot.instance
    else:
        robot_id = robot
        if instance is None:
            raise ValueError("Instance must be provided when passing a robot name")
    key = (robot_id, instance)
    if key not in _robot_data_stream_store:
        _robot_data_stream_store[key] = DatastreamStore(instance, robot_id)
    return _robot_data_stream_store[key]
