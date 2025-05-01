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


class RecordingStartPayload(BaseRecodingUpdatePayload):
    created_by: str
    start_time: float
    dataset_ids: list[str] = Field(default_factory=list)
    data_types: set[DataType] = Field(default_factory=set)

class RecordingNotificationType(str, Enum):
    START = "start"
    STOP = "stop"
    SAVED = "saved"


class RecordingNotification(BaseModel):
    type: RecordingNotificationType
    payload: RecordingStartPayload | BaseRecodingUpdatePayload


class RobotStreamTrack(BaseModel):
    robot_id: str
    robot_instance: int
    stream_id: str
    kind: str
    label: str
    mid: Optional[str] = Field(default=None)
    id: str = Field(default_factory=lambda: uuid4().hex)
