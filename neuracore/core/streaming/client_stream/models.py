
from enum import Enum
from pydantic import BaseModel


class MessageType(str, Enum):
    SDP_OFFER = "offer"
    SDP_ANSWER = "answer"
    ICE_CANDIDATE = "ice"
    HEARTBEAT = "heartbeat"
    STREAM_INFO = "stream_info"


class HandshakeMessage(BaseModel):
    from_id: str
    to_id: str
    data: str
    type: MessageType
    id: str
