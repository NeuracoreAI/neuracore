from typing import Optional

from ..core.robot import Robot
from ..core.streaming.data_stream import DataStream


class GlobalSingleton(object):
    _instance = None
    _has_validated_version = False
    _active_robot: Optional[Robot] = None
    _active_dataset_id: Optional[str] = None
    _active_recording_ids: dict[str, str] = {}
    _data_streams: dict[str, DataStream] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
