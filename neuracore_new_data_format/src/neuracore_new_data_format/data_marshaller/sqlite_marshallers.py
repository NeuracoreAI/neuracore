import json
from sqlite3 import Cursor
from typing import Generator

import numpy as np

from neuracore_new_data_format.data_marshaller.data_marshaller import (
    CameraDataEncoder,
    DataMarshaller,
)
from neuracore_new_data_format.ncdata import (
    CameraData,
    CustomData,
    JointData,
    LanguageData,
    NCData,
)


class SqliteMarshaller(DataMarshaller):
    def __init__(self, cur: Cursor):
        self.cur = cur


class CustomDataTrackSqliteMarshaller(SqliteMarshaller):
    TABLE_NAME = "custom_data_track"

    def __init__(self, cur: Cursor, name: str):
        super().__init__(cur=cur)
        self.name = name
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME}_{self.name} (
                timestamp REAL,
                data TEXT
            )
            """
        )

    def write(self, data: CustomData) -> None:
        self.cur.execute(
            f"INSERT INTO {self.TABLE_NAME}_{self.name} (timestamp, data) VALUES (?, ?)",
            (data.timestamp, json.dumps(data.data)),
        )

    def read(self) -> Generator["NCData", None, None]:
        self.cur.execute(f"SELECT timestamp, data FROM {self.TABLE_NAME}_{self.name}")
        for timestamp, data in self.cur.fetchall():
            yield CustomData(name=self.name, timestamp=timestamp, data=json.loads(data))


class CustomDataSqliteMarshaller(SqliteMarshaller):
    TABLE_NAME = "custom_data"

    def __init__(self, cur: Cursor):
        super().__init__(cur=cur)
        self.tracks: dict[str, CustomDataTrackSqliteMarshaller] = {}

    def get_track(self, name: str) -> CustomDataTrackSqliteMarshaller:

        if name not in self.tracks:

            self.tracks[name] = CustomDataTrackSqliteMarshaller(cur=self.cur, name=name)
        return self.tracks[name]

    def write(self, data: CustomData) -> None:
        self.get_track(data.name).write(data)

    def read(self) -> Generator["NCData", None, None]:
        self.cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' and NAME LIKE '{CustomDataTrackSqliteMarshaller.TABLE_NAME}_%';"
        )
        for (name,) in self.cur.fetchall():
            yield from self.get_track(
                name[len(CustomDataTrackSqliteMarshaller.TABLE_NAME) + 1 :]
            ).read()


class LanguageDataSqliteMarshaller(SqliteMarshaller):
    TABLE_NAME = "language_data"

    def __init__(self, cur: Cursor):
        super().__init__(cur=cur)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                timestamp REAL,
                text TEXT
            )
            """
        )

    def write(self, data: LanguageData) -> None:
        self.cur.execute(
            f"INSERT INTO {self.TABLE_NAME} (timestamp, text) VALUES (?, ?)",
            (data.timestamp, data.text),
        )

    def read(
        self,
    ) -> Generator["NCData", None, None]:
        self.cur.execute(f"SELECT timestamp, text FROM {self.TABLE_NAME}")
        for timestamp, text in self.cur.fetchall():
            yield LanguageData(timestamp=timestamp, text=text)


class CameraDataMetadataSqliteMarshaller(SqliteMarshaller):
    TABLE_NAME = "camera_metadata"

    def __init__(self, cur: Cursor, camera_id: str):
        self.cur = cur
        self.camera_id = camera_id

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME}_{self.camera_id} (
                timestamp REAL,
                frame_idx INTEGER,
                extrinsics BLOB,
                intrinsics BLOB
            )
            """
        )

    def read(
        self,
        frames: list[np.ndarray],
    ) -> Generator["NCData", None, None]:
        self.cur.execute(
            f"SELECT timestamp, frame_idx, extrinsics, intrinsics FROM {self.TABLE_NAME}_{self.camera_id}"
        )

        for timestamp, frame_idx, extrinsics, intrinsics in self.cur.fetchall():
            yield CameraData(
                timestamp=timestamp,
                camera_id=self.camera_id,
                frame_idx=frame_idx,
                frame=frames[frame_idx],
                extrinsics=(
                    np.frombuffer(extrinsics, dtype=np.float32)
                    if extrinsics is not None
                    else None
                ),
                intrinsics=(
                    np.frombuffer(intrinsics, dtype=np.float32)
                    if intrinsics is not None
                    else None
                ),
            )

    def write(
        self,
        data: CameraData,
    ) -> None:
        self.cur.execute(
            f"INSERT INTO {self.TABLE_NAME}_{self.camera_id} (timestamp, frame_idx, extrinsics, intrinsics) VALUES (?, ?, ?, ?)",
            (
                data.timestamp,
                data.frame_idx,
                (
                    np.array(data.extrinsics).tobytes()
                    if data.extrinsics is not None
                    else None
                ),
                (
                    np.array(data.intrinsics).tobytes()
                    if data.intrinsics is not None
                    else None
                ),
            ),
        )


class CameraDataVideoSqliteMarshaller:
    TABLE_NAME = "camera_data"

    def __init__(self, cur: Cursor, camera_id: str, encoder: CameraDataEncoder):
        self.cur = cur
        self.camera_id = camera_id
        self.encoder = encoder

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                camera_id TEXT,
                frames BLOB
            )
            """
        )

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Add a numpy RGB frame to the encoder with a timestamp in seconds.
        PTS is calculated relative to the first frame.
        """
        self.encoder.add_frame(frame, timestamp)

    def read_frames(self) -> list[np.ndarray]:
        """
        Decode MP4 bytes back into frames.
        Returns list of frame ndarray
        """
        self.cur.execute(
            f"SELECT frames FROM {self.TABLE_NAME} WHERE camera_id = ?",
            (self.camera_id,),
        )
        blob = self.cur.fetchone()[0]
        return self.encoder.read_frames(blob)

    def close(self) -> None:
        """Finalize encoding and write MP4 bytes to SQLite."""
        blob = self.encoder.get_blob()
        self.cur.execute(
            f"INSERT INTO {self.TABLE_NAME} (camera_id, frames) VALUES (?, ?)",
            (self.camera_id, blob),
        )


class CameraDataSqliteMarshaller(SqliteMarshaller):
    def __init__(
        self,
        cur: Cursor,
    ):
        super().__init__(cur=cur)

        self.metadata_marshaller: dict[str, CameraDataMetadataSqliteMarshaller] = {}
        self.video_marshallers: dict[str, CameraDataVideoSqliteMarshaller] = {}

    def get_metadata_marshaller(
        self, camera_id: str
    ) -> CameraDataMetadataSqliteMarshaller:
        if camera_id not in self.metadata_marshaller:
            self.metadata_marshaller[camera_id] = CameraDataMetadataSqliteMarshaller(
                cur=self.cur, camera_id=camera_id
            )
        return self.metadata_marshaller[camera_id]

    def get_video_encoder(self, camera_id: str) -> CameraDataVideoSqliteMarshaller:
        if camera_id not in self.video_marshallers:
            self.video_marshallers[camera_id] = CameraDataVideoSqliteMarshaller(
                cur=self.cur, camera_id=camera_id, encoder=CameraDataEncoder()
            )

        return self.video_marshallers[camera_id]

    def write(
        self,
        data: CameraData,
    ) -> None:
        self.get_metadata_marshaller(data.camera_id).write(data)
        self.get_video_encoder(data.camera_id).add_frame(data.frame, data.timestamp)

    def read(
        self,
    ) -> Generator["NCData", None, None]:
        for camera_id in self.list_camera_ids():
            metadata_marshaller = self.get_metadata_marshaller(camera_id)

            frames = self.get_video_encoder(camera_id).read_frames()
            yield from metadata_marshaller.read(frames)

    def close(self):
        for video_encoder in self.video_marshallers.values():
            video_encoder.close()

    def list_camera_ids(self) -> list[str]:
        self.cur.execute(
            f"SELECT DISTINCT camera_id FROM {CameraDataVideoSqliteMarshaller.TABLE_NAME}"
        )
        return [row[0] for row in self.cur.fetchall()]


class JointDataSqliteMarshaller(SqliteMarshaller):
    def __init__(self, cur: Cursor, data_type: str):
        super().__init__(cur=cur)
        self.data_type = data_type
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {data_type} (
                timestamp REAL
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {data_type}_values (
                timestamp REAL,
                joint_name TEXT,
                joint_value REAL,
                FOREIGN KEY (timestamp) REFERENCES {data_type}(timestamp)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {data_type}_additional_values (
                timestamp REAL,
                additional_name TEXT,
                additional_value REAL,
                FOREIGN KEY (timestamp) REFERENCES {data_type}(timestamp)
            )
            """
        )

    def write(self, data: JointData) -> None:
        self.cur.execute(
            f"INSERT INTO {self.data_type} (timestamp) VALUES (?)", (data.timestamp,)
        )

        self.cur.executemany(
            f"INSERT INTO {self.data_type}_values (timestamp, joint_name, joint_value) VALUES (?, ?, ?)",
            [
                (data.timestamp, joint_name, joint_value)
                for joint_name, joint_value in data.values.items()
            ],
        )
        if data.additional_values:
            self.cur.executemany(
                f"INSERT INTO {self.data_type}_additional_values (timestamp, additional_name, additional_value) VALUES (?, ?, ?)",
                [
                    (data.timestamp, additional_name, additional_value)
                    for additional_name, additional_value in data.additional_values.items()
                ],
            )

    def read(
        self,
    ) -> Generator["NCData", None, None]:
        self.cur.execute(f"SELECT timestamp FROM {self.data_type}")
        for (timestamp,) in self.cur.fetchall():
            self.cur.execute(
                f"SELECT joint_name, joint_value FROM {self.data_type}_values WHERE timestamp = ?",
                (timestamp,),
            )
            values = {name: value for name, value in self.cur.fetchall()}

            self.cur.execute(
                f"SELECT additional_name, additional_value FROM {self.data_type}_additional_values WHERE timestamp = ?",
                (timestamp,),
            )
            additional_values = {
                name: value for name, value in self.cur.fetchall()
            } or None
            yield JointData(
                timestamp=timestamp, values=values, additional_values=additional_values
            )
