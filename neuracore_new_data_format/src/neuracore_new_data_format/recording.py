from abc import ABC, abstractmethod
from enum import Enum
import sqlite3
from typing import Generator

from neuracore_new_data_format.data_marshaller.data_marshaller import (
    DataMarshaller,
)
from neuracore_new_data_format.data_marshaller.sqlite_marshallers import (
    CameraDataSqliteMarshaller,
    CustomDataSqliteMarshaller,
    JointDataSqliteMarshaller,
    LanguageDataSqliteMarshaller,
)
from neuracore_new_data_format.ncdata import NCData

BATCH_SIZE = 5000


class RecordingType(Enum):
    SQLITE = "sqlite"
    MCAP = "mcap"


class Recording(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _get_marshaller(self, data_type: str) -> DataMarshaller:
        raise NotImplementedError("_get_marshaller not implemented")

    def log_data(self, data_type: str, data: NCData):
        self._get_marshaller(data_type).write(data)

        self.batch_size += 1
        if self.batch_size > BATCH_SIZE:
            self.con.commit()
            self.batch_size = 0

    def stop(self):
        for marshaller in self.data_marshaller.values():
            marshaller.close()

        self.con.commit()
        self.con.close()

    def read_data(self, data_type: str) -> Generator[NCData, None, None]:
        yield from self._get_marshaller(data_type).read()


class RecordingFactory:
    def __init__(self, recording_type: RecordingType):
        self.recording_type = recording_type

    def create_recording(self, name: str) -> Recording:
        if self.recording_type == RecordingType.SQLITE:
            return SqliteRecording(name)
        elif self.recording_type == RecordingType.MCAP:
            return McapRecording(name)
        else:
            raise ValueError(f"Unknown recording type: {self.recording_type}")


class SqliteRecording(Recording):
    def __init__(self, name: str):
        self.name = name
        self.con = sqlite3.connect(f"{self.name}.db")
        self.batch_size = 0
        self.cur = self.con.cursor()

        self.cur.execute("PRAGMA journal_mode = OFF;")  # no rollback journal
        self.cur.execute("PRAGMA synchronous = OFF;")  # don't wait for disk flush
        self.cur.execute("PRAGMA locking_mode = EXCLUSIVE;")  # fewer lock overheads
        self.cur.execute("PRAGMA temp_store = MEMORY;")  # temp tables in RAM
        self.cur.execute("PRAGMA cache_size = -100000;")  # 100MB cache in RAM
        self.cur.execute("PRAGMA foreign_keys = OFF;")

        self.data_marshaller: dict[str, DataMarshaller] = {}

    def _get_marshaller(self, data_type: str):

        if data_type in self.data_marshaller:
            return self.data_marshaller[data_type]

        match data_type:
            case "custom":
                self.data_marshaller[data_type] = CustomDataSqliteMarshaller(
                    cur=self.cur
                )
            case (
                "joint_positions"
                | "joint_velocities"
                | "joint_torques"
                | "joint_target_positions"
            ):
                self.data_marshaller[data_type] = JointDataSqliteMarshaller(
                    cur=self.cur, data_type=data_type
                )
            case "language":
                self.data_marshaller[data_type] = LanguageDataSqliteMarshaller(
                    cur=self.cur
                )
            case "rgb":
                self.data_marshaller[data_type] = CameraDataSqliteMarshaller(
                    cur=self.cur
                )
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

        return self.data_marshaller[data_type]


class McapRecording(Recording):
    def __init__(self, name: str):
        self.name = name

    def _get_marshaller(self, data_type: str):
        raise NotImplementedError("_get_marshaller not implemented yet")
