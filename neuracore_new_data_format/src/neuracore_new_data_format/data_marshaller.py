import io
import json
from abc import ABC, abstractmethod
from fractions import Fraction
from sqlite3 import Cursor
from typing import Generator

import av
import numpy as np

from neuracore_new_data_format.ncdata import (
    CameraData,
    CustomData,
    JointData,
    LanguageData,
    NCData,
)


class MarshallingOutput(ABC):
    pass


class DataMarshaller(ABC):
    @abstractmethod
    def write_to_table(self, data: NCData) -> None:
        raise NotImplementedError("write_to_table not implemented")

    @abstractmethod
    def read_from_table(
        self,
    ) -> Generator["NCData", None, None]:
        raise NotImplementedError("read_from_table not implemented")

    def close(self) -> None:
        pass


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

    def write_to_table(self, data: CustomData) -> None:
        self.cur.execute(
            f"INSERT INTO {self.TABLE_NAME}_{self.name} (timestamp, data) VALUES (?, ?)",
            (data.timestamp, json.dumps(data.data)),
        )

    def read_from_table(self) -> Generator["NCData", None, None]:
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

    def write_to_table(self, data: CustomData) -> None:
        self.get_track(data.name).write_to_table(data)

    def read_from_table(self) -> Generator["NCData", None, None]:
        self.cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' and NAME LIKE '{CustomDataTrackSqliteMarshaller.TABLE_NAME}_%';"
        )
        for (name,) in self.cur.fetchall():
            yield from self.get_track(
                name[len(CustomDataTrackSqliteMarshaller.TABLE_NAME) + 1 :]
            ).read_from_table()


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

    def write_to_table(self, data: LanguageData) -> None:
        self.cur.execute(
            f"INSERT INTO {self.TABLE_NAME} (timestamp, text) VALUES (?, ?)",
            (data.timestamp, data.text),
        )

    def read_from_table(
        self,
    ) -> Generator["NCData", None, None]:
        self.cur.execute(f"SELECT timestamp, text FROM {self.TABLE_NAME}")
        for timestamp, text in self.cur.fetchall():
            yield LanguageData(timestamp=timestamp, text=text)


class CameraDataMetadataMarshaller:
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

    def read_from_table(
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

    def write_to_table(
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

PTS_FRACT = 1000000  # Timebase for pts in microseconds


class CameraDataEncoder:
    TABLE_NAME = "camera_data"

    def __init__(
        self,
        cur: Cursor,
        camera_id: str,
        codec: str = "libx264",
        pixel_format: str = "yuv444p10le",
    ):
        self.cur = cur
        self.camera_id = camera_id

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                camera_id TEXT,
                frames BLOB
            )
            """
        )

        self.codec = codec
        self.pixel_format = pixel_format

        self.buffer = io.BytesIO()
        self.container = av.open(
            self.buffer,
            mode="w",
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov"},
        )
        self.stream = None

        self.start_ts = None  # first timestamp
        self.time_base = None  # time base of the stream
        self.last_pts = None  # last pts

    

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Add a numpy RGB frame to the encoder with a timestamp in seconds.
        PTS is calculated relative to the first frame.
        """
        if self.stream is None:
            h, w, _ = frame.shape
            self.stream = self.container.add_stream(self.codec)
            self.stream.width = w
            self.stream.height = h
            self.stream.pix_fmt = self.pixel_format
            self.stream.options = {
                "preset": "ultrafast",
            }
            self.stream.codec_context.options = {
                "qp": "0", # lossless quantization
                "preset": "ultrafast", # low compression fast speed
            }
            # let PyAV pick time_base automatically (usually 1/1000 or 1/90000)
            self.time_base = Fraction(1, PTS_FRACT)
            self.stream.time_base = self.time_base

        if self.start_ts is None:
            self.start_ts = timestamp

        rel_ts = timestamp - self.start_ts
        pts = int(rel_ts * PTS_FRACT)  # Convert to microseconds

        # Ensure pts is monotonically increasing (required by most codecs)
        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1

        self.last_pts = pts


        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame = av_frame.reformat(format=self.pixel_format)
        av_frame.pts = pts
        av_frame.time_base = self.time_base

        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

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
        buffer = io.BytesIO(blob)
        container = av.open(buffer, mode="r", format="mp4")

        frames = []
        for packet in container.demux(video=0):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))
        return frames

    def close(self) -> None:
        """Finalize encoding and write MP4 bytes to SQLite."""
        if self.stream is not None:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        self.container.close()

        blob = self.buffer.getvalue()
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

        self.metadata_marshaller: dict[str, CameraDataMetadataMarshaller] = {}
        self.video_encoders: dict[str, CameraDataEncoder] = {}

    def get_metadata_marshaller(self, camera_id: str) -> CameraDataMetadataMarshaller:
        if camera_id not in self.metadata_marshaller:
            self.metadata_marshaller[camera_id] = CameraDataMetadataMarshaller(
                cur=self.cur, camera_id=camera_id
            )
        return self.metadata_marshaller[camera_id]

    def get_video_encoder(self, camera_id: str) -> CameraDataEncoder:
        if camera_id not in self.video_encoders:
            self.video_encoders[camera_id] = CameraDataEncoder(
                cur=self.cur, camera_id=camera_id
            )

        return self.video_encoders[camera_id]

    def write_to_table(
        self,
        data: CameraData,
    ) -> None:
        self.get_metadata_marshaller(data.camera_id).write_to_table(data)
        self.get_video_encoder(data.camera_id).add_frame(data.frame, data.timestamp)

    def read_from_table(
        self,
    ) -> Generator["NCData", None, None]:
        for camera_id in self.list_camera_ids():
            metadata_marshaller = self.get_metadata_marshaller(camera_id)

            frames = self.get_video_encoder(camera_id).read_frames()
            yield from metadata_marshaller.read_from_table(frames)

    def close(self):
        for video_encoder in self.video_encoders.values():
            video_encoder.close()

    def list_camera_ids(self) -> list[str]:
        self.cur.execute(
            f"SELECT DISTINCT camera_id FROM {CameraDataEncoder.TABLE_NAME}"
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

    def write_to_table(self, data: JointData) -> None:
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

    def read_from_table(
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
