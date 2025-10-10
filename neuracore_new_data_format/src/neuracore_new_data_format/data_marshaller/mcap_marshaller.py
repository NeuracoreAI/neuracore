from __future__ import annotations

import json
from abc import abstractmethod
from typing import Generator, Iterable, Optional

import numpy as np
from google.protobuf.message import Message
from mcap.reader import McapReader, make_reader
from mcap_protobuf.decoder import DecoderFactory
from mcap_protobuf.writer import Writer

from neuracore_new_data_format.data_marshaller.data_marshaller import (
    CameraDataEncoder,
    DataMarshaller,
)

# Import your dataclasses
from neuracore_new_data_format.ncdata import (
    CameraData,
    CameraMetadata,
    CustomData,
    JointData,
    LanguageData,
    NCData,
)
from neuracore_new_data_format.protos import neuracore_new_data_format_pb2 as pb

# Import generated protobuf classes
from neuracore_new_data_format.protos.neuracore_new_data_format_pb2 import (
    CameraMetadata as CameraMetadataProto,
)
from neuracore_new_data_format.protos.neuracore_new_data_format_pb2 import (
    CustomData as CustomDataProto,
)
from neuracore_new_data_format.protos.neuracore_new_data_format_pb2 import (
    JointData as JointDataProto,
)
from neuracore_new_data_format.protos.neuracore_new_data_format_pb2 import (
    LanguageData as LanguageDataProto,
)
from neuracore_new_data_format.protos.neuracore_new_data_format_pb2 import VideoBlob


class ReaderGenerator:
    def __init__(self, filename):
        self.filename = filename
        self._reader = None

    @property
    def reader(self) -> McapReader:
        if not self._reader:
            file = open(self.filename, "rb")
            print(f"start reading {self.filename}")
            self._reader = make_reader(file, decoder_factories=[DecoderFactory()])
        return self._reader


class BaseMcapMarshaller(DataMarshaller):
    """Helper utilities for shared conversions."""

    def __init__(self, writer: Writer, reader: ReaderGenerator):
        super().__init__()
        self.writer = writer
        self.reader = reader

    @abstractmethod
    def topic_name(self, data: NCData) -> str:
        raise NotImplementedError()

    @abstractmethod
    def list_topics(self) -> Iterable[str]:
        raise NotImplementedError()

    def write(self, data: NCData) -> None:
        self.writer.write_message(
            topic=self.topic_name(data),
            message=self.serialize_data(data),
            log_time=int(data.timestamp * 1e9),
            publish_time=int(data.timestamp * 1e9),
        )

    def close(self) -> None:
        self.writer.finish()

    def read(
        self,
    ) -> Generator["NCData", None, None]:
        for (
            schema,
            channel,
            message,
            proto_msg,
        ) in self.reader.reader.iter_decoded_messages(topics=self.list_topics()):
            yield self.deserialize_data(proto_msg)

    @staticmethod
    @abstractmethod
    def serialize_data(data: NCData) -> Message:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def deserialize_data(msg: Message) -> NCData:
        raise NotImplementedError()

    # ---- ndarray <-> FloatArray ----
    @staticmethod
    def _ndarray_to_proto(array: np.ndarray) -> pb.FloatArray:
        return pb.FloatArray(
            values=array.astype(np.float32).ravel().tolist(),
            shape=list(array.shape),
        )

    @staticmethod
    def _proto_to_ndarray(fa: pb.FloatArray) -> np.ndarray:
        arr = np.array(fa.values, dtype=np.float32)
        if fa.shape:
            arr = arr.reshape(tuple(fa.shape))
        return arr

    # ---- list[ndarray] <-> repeated FloatArray ----
    @staticmethod
    def _ndarray_list_to_proto(arrays: Optional[list[np.ndarray]]):
        if arrays is None:
            return []
        return [BaseMcapMarshaller._ndarray_to_proto(a) for a in arrays]

    @staticmethod
    def _proto_to_ndarray_list(fas):
        if not fas:
            return None
        return [BaseMcapMarshaller._proto_to_ndarray(fa) for fa in fas]


# ---------------------------
# JointData
# ---------------------------
class JointDataMcapMarshaller(BaseMcapMarshaller):
    def __init__(self, writer, reader=None, data_type: str = None):
        super().__init__(writer, reader)
        self.data_type = data_type

    def topic_name(self, data: JointData) -> str:
        return f"joint_data/{self.data_type}"

    def list_topics(self) -> list[str]:
        return [
            key
            for key in self.reader.get_summary().channels.keys()
            if key.startswith("joint_data")
        ]

    @staticmethod
    def serialize_data(data: JointData) -> JointDataProto:
        msg = JointDataProto(
            timestamp=data.timestamp,
            values=data.values,
        )
        if data.additional_values:
            msg.additional_values.update(data.additional_values)
        return msg

    @staticmethod
    def deserialize_data(msg: JointDataProto) -> JointData:
        return JointData(
            timestamp=msg.timestamp,
            values=dict(msg.values),
            additional_values=(
                dict(msg.additional_values) if msg.additional_values else None
            ),
        )


# ---------------------------
# CameraData
# ---------------------------


class CameraDataVideoMcapMarshaller:
    def __init__(self, writer: Writer, camera_id: str, encoder: CameraDataEncoder):
        self.writer = writer
        self.camera_id = camera_id
        self.encoder = encoder

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Add a numpy RGB frame to the encoder with a timestamp in seconds.
        PTS is calculated relative to the first frame.
        """
        self.encoder.add_frame(frame, timestamp)

    def read_frames(self, blob: bytes) -> list[np.ndarray]:
        """
        Decode MP4 bytes back into frames.
        Returns list of frame ndarray
        """
        return self.encoder.read_frames(blob)

    def close(self) -> None:
        """Finalize encoding and write MP4 bytes to MCAP file."""
        blob = VideoBlob(blob=self.encoder.get_blob())
        # Here, you would define how to write the blob to the MCAP file.
        # This might involve a custom message type for video frames.
        # For simplicity, let's assume a placeholder mechanism.
        self.writer.write_message(
            topic=f"camera_video/{self.camera_id}",
            message=blob,  # This needs a proper protobuf message
            log_time=0,  # Adjust timestamp logic as needed
            publish_time=0,
        )


class CameraDataMcapMarshaller(DataMarshaller):
    def __init__(self, writer: Writer, reader: McapReader | None = None):
        super().__init__()
        self.writer = writer
        self.reader = reader

        self.metadata_marshaller: dict[str, CameraDataMcapMetadataMarshaller] = {}
        self.video_marshallers: dict[str, CameraDataVideoMcapMarshaller] = {}

    def get_metadata_marshaller(
        self, camera_id: str
    ) -> CameraDataMcapMetadataMarshaller:
        if camera_id not in self.metadata_marshaller:
            self.metadata_marshaller[camera_id] = CameraDataMcapMetadataMarshaller(
                writer=self.writer, reader=self.reader
            )
        return self.metadata_marshaller[camera_id]

    def get_video_encoder(self, camera_id: str) -> CameraDataVideoMcapMarshaller:
        if camera_id not in self.video_marshallers:
            self.video_marshallers[camera_id] = CameraDataVideoMcapMarshaller(
                writer=self.writer, camera_id=camera_id, encoder=CameraDataEncoder()
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

            # This part is tricky, as it requires reading the video blob first
            # and then associating it with metadata. MCAP does not guarantee order.
            # You might need a more sophisticated reading mechanism.
            # For now, let's assume a simplified (and likely incorrect) approach.

            video_frames_blob = None  # Placeholder for where you'd read the video blob
            frames = self.get_video_encoder(camera_id).read_frames(video_frames_blob)

            # Now, iterate through metadata and associate with frames
            # This logic needs to be robust.
            yield from metadata_marshaller.read(frames)

    def close(self):
        for video_encoder in self.video_marshallers.values():
            video_encoder.close()
        super().close()

    def list_camera_ids(self) -> list[str]:
        return [
            key.topic.split("/")[-1]
            for key in self.reader.get_summary().channels.values()
            if key.startswith("camera_metadata")
        ]


class CameraDataMcapMetadataMarshaller(BaseMcapMarshaller):
    def topic_name(self, data: CameraMetadata) -> str:
        return f"camera_metadata/{data.camera_id}"

    def list_topics(self) -> list[str]:
        return [
            key
            for key in self.reader.get_summary().channels.values()
            if key.startswith("camera_metadata")
        ]

    @staticmethod
    def serialize_data(data: CameraMetadata) -> CameraMetadataProto:
        return CameraMetadataProto(
            timestamp=data.timestamp,
            camera_id=data.camera_id,
            frame_idx=data.frame_idx,
            extrinsics=BaseMcapMarshaller._ndarray_list_to_proto(data.extrinsics),
            intrinsics=BaseMcapMarshaller._ndarray_list_to_proto(data.intrinsics),
        )

    @staticmethod
    def deserialize_data(msg: CameraMetadataProto) -> CameraMetadata:
        return CameraMetadata(
            timestamp=msg.timestamp,
            camera_id=msg.camera_id,
            frame_idx=msg.frame_idx,
            extrinsics=BaseMcapMarshaller._proto_to_ndarray_list(msg.extrinsics),
            intrinsics=BaseMcapMarshaller._proto_to_ndarray_list(msg.intrinsics),
        )


# ---------------------------
# LanguageData
# ---------------------------
class LanguageDataMcapMarshaller(BaseMcapMarshaller):
    def topic_name(self, data: LanguageData) -> str:
        return "language_data"

    def list_topics(self) -> list[str]:
        return ["language_data"]

    @staticmethod
    def serialize_data(data: LanguageData) -> LanguageDataProto:
        msg = LanguageDataProto(
            timestamp=data.timestamp,
            text=data.text,
        )
        return msg

    @staticmethod
    def deserialize_data(msg: LanguageDataProto) -> LanguageData:
        return LanguageData(timestamp=msg.timestamp, text=msg.text)


# ---------------------------
# CustomData (JSON serialization)
# ---------------------------
class CustomDataMcapMarshaller(BaseMcapMarshaller):
    def topic_name(self, data: CustomData) -> str:
        return f"custom_data/{data.name}"

    def list_topics(self) -> list[str]:
        return [
            key
            for key in self.reader.get_summary().channels.keys()
            if key.startswith("custom_data")
        ]

    @staticmethod
    def serialize_data(data: CustomData) -> CustomDataProto:
        msg = CustomDataProto(
            timestamp=data.timestamp,
            name=data.name,
            json_data=json.dumps(data.data),
        )
        return msg

    @staticmethod
    def deserialize_data(msg: CustomDataProto) -> CustomData:
        try:
            parsed = json.loads(msg.json_data)
        except json.JSONDecodeError:
            parsed = msg.json_data
        return CustomData(timestamp=msg.timestamp, name=msg.name, data=parsed)
