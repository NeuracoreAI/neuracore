from typing import Optional

from neuracore.core.const import MAX_DATA_STREAMS

from ..core.robot import Robot
from ..core.streaming.data_stream import DataStream


class GlobalSingleton(object):
    _instance = None
    _has_validated_version = False
    _active_robot: Optional[Robot] = None
    _active_dataset_id: Optional[str] = None
    _active_recording_ids: dict[str, str] = {}
    _data_streams: dict[str, DataStream] = {}

    def add_data_stream(self, stream_id: str, stream: DataStream):
        if len(self._data_streams) > MAX_DATA_STREAMS:
            raise RuntimeError("Excessive number of data streams")
        if stream_id in self._data_streams:
            raise ValueError("Stream already exists")
        self._data_streams[stream_id] = stream
        return stream

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
