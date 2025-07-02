"""Pydantic models for WebRTC streaming and recording notifications.

This module defines data models used for WebRTC signaling, recording
lifecycle management, and robot stream tracking. All models use Pydantic
for validation and serialization.
"""

from enum import Enum
from typing import Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, NonNegativeInt

from neuracore.core.nc_types import DataType


class MessageType(str, Enum):
    """Types of WebRTC signaling messages."""

    SDP_OFFER = "offer"
    SDP_ANSWER = "answer"
    ICE_CANDIDATE = "ice"
    STREAM_END = "end"
    CONNECTION_TOKEN = "token"


class HandshakeMessage(BaseModel):
    """WebRTC signaling message for peer-to-peer connection establishment.

    Used for exchanging SDP offers/answers, ICE candidates, and connection
    tokens between peers during the WebRTC handshake process.
    """

    from_id: str
    to_id: str
    data: str
    connection_id: str
    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)


class BaseRecodingUpdatePayload(BaseModel):
    """Base payload for recording update notifications.

    Contains the minimum information needed to identify a recording
    and the robot instance it belongs to.
    """

    recording_id: str
    robot_id: str
    instance: NonNegativeInt


class RecodingRequestedPayload(BaseRecodingUpdatePayload):
    """Payload for recording request notifications.

    Contains information about who requested the recording and what
    data types should be captured.
    """

    created_by: str
    dataset_ids: list[str] = Field(default_factory=list)
    data_types: set[DataType] = Field(default_factory=set)


class RecordingStartPayload(RecodingRequestedPayload):
    """Payload for recording start notifications.

    Extends the request payload with the actual start timestamp
    when recording begins.
    """

    start_time: float


class RecordingNotificationType(str, Enum):
    """Types of recording lifecycle notifications."""

    INIT = "init"
    REQUESTED = "requested"
    START = "start"
    STOP = "stop"
    SAVED = "saved"
    DISCARDED = "discarded"
    EXPIRED = "expired"


class RecordingNotification(BaseModel):
    """Notification message for recording lifecycle events.

    Used to communicate recording state changes across the system,
    including initialization, start/stop events, and final disposition.
    """

    type: RecordingNotificationType
    payload: Union[
        RecordingStartPayload,
        RecodingRequestedPayload,
        list[Union[RecordingStartPayload, RecodingRequestedPayload]],
        BaseRecodingUpdatePayload,
    ]


class RobotStreamTrack(BaseModel):
    """Metadata for a robot's media stream track.

    Contains all information needed to identify and route a specific
    media track from a robot instance, including video feeds and
    data channels.
    """

    robot_id: str
    robot_instance: NonNegativeInt
    stream_id: str
    kind: str
    label: str
    mid: Optional[str] = Field(default=None)
    id: str = Field(default_factory=lambda: uuid4().hex)
