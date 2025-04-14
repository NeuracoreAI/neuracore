from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


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


# TODO: Note all of these can be defined in api, and used by the backend
class RecordingNotification(BaseModel):
    recording: bool
    robot_id: str
    recording_id: str


class RobotStreamTrack(BaseModel):
    robot_id: str
    robot_instance: int
    stream_id: str
    kind: str
    label: str
    mid: Optional[str] = Field(default=None)
    id: str = Field(default_factory=lambda: uuid4().hex)
