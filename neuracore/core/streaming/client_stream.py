from dataclasses import Field, dataclass
from enum import Enum
import json

from pydantic import BaseModel

from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.resumable_upload import SensorType, Uploader
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiohttp_sse_client import client as sse_client
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from aiohttp import ClientSession
from ..const import API_URL


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


@dataclass(frozen=True, slots=True)
class PierToPierConnection:
    local_stream_id: str
    remote_stream_id: str
    client_session: ClientSession = Field(
        default_factory=lambda: ClientSession(API_URL)
    )
    auth: Auth = Field(default_factory=get_auth)
    connection: RTCPeerConnection

    async def send_message(self, message_type: MessageType, content: str):
        await self.client_session.post(
            f"/signalling/submit/{str(message_type)}/from/{self.stream_id}/to/{self.remote_stream_id}",
            headers=self.auth.get_headers(),
            data=content,
        )

    async def on_ice(self, ice: str):
        candidate = candidate_from_sdp(ice)
        await self.connection.addIceCandidate(candidate)

    async def on_offer(self, offer: str):
        offer = RTCSessionDescription(offer, type="offer")
        await self.connection.setRemoteDescription(offer)

        answer = await self.connection.createAnswer()
        await self.connection.setLocalDescription(answer)
        await self.send_message(MessageType.SDP_ANSWER, answer.sdp)


MAXIMUM_CLIENT_FREQUENCY_HZ = 60


@dataclass(frozen=True, slots=True)
class ClientStream(Uploader):
    recording_id: str
    sensor_type: SensorType
    sensor_name: str

    def upload_chunk(self, data: bytes, is_final: bool = False) -> bool:
        pass


@dataclass(frozen=True, slots=True)
class ClientStreamingManager:
    client_session: ClientSession = Field(
        default_factory=lambda: ClientSession(API_URL)
    )
    auth: Auth = Field(default_factory=get_auth)
    connections: dict[str, PierToPierConnection] = Field(default_factory=dict)
    robot_id: str

    def start_recording_stream(recording_id: str) -> ClientStream:
        return ClientStream(recording_id)

    async def connect_signalling_stream(self):
        async with sse_client.EventSource(
            f"/signalling/notifications/robot/{self.robot_id}",
            headers=self.auth.get_headers(),
        ) as event_source:
            try:
                async for event in event_source:
                    message = HandshakeMessage.model_validate_json(event.data)
                    connection = self.connections.get(message.from_id)

                    if message.from_id == "system":
                        continue

                    if connection is None:
                        connection = PierToPierConnection(
                            local_stream_id=message.to_id,
                            remote_stream_id=message.from_id,
                            client_session=self.client_session,
                            auth=self.auth,
                        )
                        self.connections[message.from_id] = connection

                    match message.type:
                        case MessageType.SDP_OFFER:
                            connection.on_offer(message.data)
                        case MessageType.ICE_CANDIDATE:
                            connection.on_ice(message.data)

            except ConnectionError:
                pass
