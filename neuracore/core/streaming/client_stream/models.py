

from enum import Enum
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field

class MessageType(str, Enum):
    CONNECTION_REQUEST = "request"
    SDP_OFFER = "offer"
    SDP_ANSWER = "answer"
    ICE_CANDIDATE = "ice"
    HEARTBEAT = "heartbeat"
    STREAM_END = "end"

class HandshakeMessage(BaseModel):
    from_id: str
    to_id: str
    data: str
    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)

class RobotStreamTrack(BaseModel):
    robot_id: str
    stream_id: str
    kind: str
    label: str
    mid: Optional[str] = Field(default=None)
    id: str = Field(default_factory=lambda: uuid4().hex)