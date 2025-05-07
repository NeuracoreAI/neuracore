from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, NonNegativeInt

from neuracore.core.nc_types import DataType


class MessageType(str, Enum):
    SDP_OFFER = "offer"
    SDP_ANSWER = "answer"
    ICE_CANDIDATE = "ice"
    STREAM_END = "end"
    CONNECTION_TOKEN = "token"


class HandshakeMessage(BaseModel):
    from_id: str
    to_id: str
    data: str
    connection_id: str
    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)


class BaseRecodingUpdatePayload(BaseModel):
    recording_id: str
    robot_id: str
    instance: NonNegativeInt


class RecodingRequestedPayload(BaseRecodingUpdatePayload):
    created_by: str
    dataset_ids: list[str] = Field(default_factory=list)
    data_types: set[DataType] = Field(default_factory=set)


class RecordingStartPayload(RecodingRequestedPayload):
    start_time: float


class RecordingNotificationType(str, Enum):
    INIT = "init"
    REQUESTED = "requested"
    START = "start"
    STOP = "stop"
    SAVED = "saved"
    DISCARDED = "discarded"
    EXPIRED = "expired"


class RecordingNotification(BaseModel):
    type: RecordingNotificationType
    payload: (
        RecordingStartPayload
        | RecodingRequestedPayload
        | list[RecordingStartPayload | RecodingRequestedPayload]
        | BaseRecodingUpdatePayload
    )


class RobotStreamTrack(BaseModel):
    robot_id: str
    robot_instance: NonNegativeInt
    stream_id: str
    kind: str
    label: str
    mid: Optional[str] = Field(default=None)
    id: str = Field(default_factory=lambda: uuid4().hex)
