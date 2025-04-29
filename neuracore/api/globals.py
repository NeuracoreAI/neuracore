from collections import defaultdict
from typing import Optional

from neuracore.core.const import MAX_DATA_STREAMS

from ..core.robot import Robot
from ..core.streaming.data_stream import DataStream


class GlobalSingleton(object):
    _instance = None
    _has_validated_version = False
    _active_robot: Optional[Robot] = None
    _active_dataset_id: Optional[str] = None
    _data_streams: defaultdict[tuple[str, int], dict[str, DataStream]] = defaultdict(
        dict
    )

    def add_data_stream(self, robot: Robot, stream_id: str, stream: DataStream):
        robot_key = (robot.id, robot.instance)
        if len(self._data_streams) >= MAX_DATA_STREAMS:
            raise RuntimeError("Excessive number of data streams")
        if robot_key in self._data_streams:
            raise ValueError("Stream already exists")
        self._data_streams[robot_key][stream_id] = stream
        return stream

    def list_all_streams(self, instance_key: tuple[str, int]) -> dict[str, DataStream]:
        return self._data_streams[instance_key]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
