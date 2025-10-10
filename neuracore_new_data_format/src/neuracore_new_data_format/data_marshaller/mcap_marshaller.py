from __future__ import annotations

import json
from abc import abstractmethod
from typing import Generator, Iterable, Optional

import numpy as np
from google.protobuf.message import Message
from mcap.reader import McapReader
from mcap_protobuf.writer import Writer

from neuracore_new_data_format.data_marshaller.data_marshaller import DataMarshaller

# Import your dataclasses
from neuracore_new_data_format.ncdata import (
    CameraMetadata,
    CustomData,
    JointData,
    LanguageData,
    NCData,
)
from neuracore_new_data_format.protos import neuracore_new_data_format_pb2 as pb

# Import generated protobuf classes
from neuracore_new_data_format.protos.neuracore_new_data_format_pb2 import (
    CameraData as CameraDataProto,
)
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


class BaseMcapMarshaller(DataMarshaller):
    """Helper utilities for shared conversions."""

    def __init__(self, writer: Writer, reader: McapReader):
        super().__init__()
        self.writer = writer
        self.reader = reader

    @abstractmethod
    def topic_name(self, data: NCData):
        raise NotImplementedError()

    @abstractmethod
    def list_topics(self) -> Iterable[str]:
        raise NotImplementedError()

    def write(self, data: NCData) -> None:
        self.writer.write_message(self.TOPIC_NAME, self.serialize_data(data))

    def read(
        self,
    ) -> Generator["NCData", None, None]:
        for schema, channel, message, proto_msg in self.reader.iter_decoded_messages(
            topics=self.list_topics()
        ):
            yield self.deserialize_data(proto_msg)

    @abstractmethod
    @staticmethod
    def serialize_data(data: NCData) -> Message:
        raise NotImplementedError()

    @abstractmethod
    @staticmethod
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


class CameraDataMcapMarshaller(BaseMcapMarshaller):
    def __init__(
        self, writer, reader, metadata_marshaller: CameraDataMcapMetadataMarshaller
    ):
        super().__init__(writer, reader)
        self.metadata_marshaller = metadata_marshaller
        sel

    @staticmethod
    def serialize_data(data: CameraMetadata) -> CameraMetadataProto:
        metadata = CameraMetadataProto(
            timestamp=data.timestamp,
            camera_id=data.camera_id,
            frame_idx=data.frame_idx,
            extrinsics=BaseMcapMarshaller._float_lists_to_proto(data.extrinsics),
            intrinsics=BaseMcapMarshaller._float_lists_to_proto(data.intrinsics),
        )
        msg = CameraDataProto(
            metadata=metadata,
            frame=data.frame.tobytes(),
        )
        return msg

    @staticmethod
    def deserialize_data(msg: CameraMetadataProto) -> CameraMetadata:
        meta = msg.metadata
        return CameraMetadata(
            timestamp=meta.timestamp,
            camera_id=meta.camera_id,
            frame_idx=meta.frame_idx,
            extrinsics=BaseMcapMarshaller._float_lists_from_proto(meta.extrinsics),
            intrinsics=BaseMcapMarshaller._float_lists_from_proto(meta.intrinsics),
        )


class CameraDataMcapMetadataMarshaller(BaseMcapMarshaller):
    @staticmethod
    def serialize_data(data: CameraMetadata) -> CameraMetadataProto:
        metadata = CameraMetadataProto(
            timestamp=data.timestamp,
            camera_id=data.camera_id,
            frame_idx=data.frame_idx,
            extrinsics=BaseMcapMarshaller._float_lists_to_proto(data.extrinsics),
            intrinsics=BaseMcapMarshaller._float_lists_to_proto(data.intrinsics),
        )
        msg = CameraDataProto(
            metadata=metadata,
            frame=data.frame.tobytes(),
        )
        return msg

    @staticmethod
    def deserialize_data(msg: CameraMetadataProto) -> CameraMetadata:
        meta = msg.metadata
        return CameraMetadata(
            timestamp=meta.timestamp,
            camera_id=meta.camera_id,
            frame_idx=meta.frame_idx,
            extrinsics=BaseMcapMarshaller._float_lists_from_proto(meta.extrinsics),
            intrinsics=BaseMcapMarshaller._float_lists_from_proto(meta.intrinsics),
        )


# ---------------------------
# LanguageData
# ---------------------------
class LanguageDataMcapMarshaller(BaseMcapMarshaller):
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
