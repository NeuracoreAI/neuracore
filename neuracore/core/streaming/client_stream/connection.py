import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

from aiohttp import ClientSession
from aiortc import (
    RTCConfiguration,
    RTCIceGatherer,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.client_stream.models import HandshakeMessage, MessageType
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

from neuracore.core.streaming.client_stream.video_source import VideoSource, VideoTrack
from ...const import API_URL

ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]


@dataclass
class PierToPierConnection:
    local_stream_id: str
    remote_stream_id: str
    on_close: Callable
    client_session: ClientSession
    loop: asyncio.AbstractEventLoop
    connection_token: asyncio.Future[str]
    auth: Auth = field(default_factory=get_auth)
    connection: RTCPeerConnection = field(
        default_factory=lambda: RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ICE_SERVERS)
        )
    )
    
    _closed: bool = False

    async def get_ice_gatherer(self) -> RTCIceGatherer:
        sctp = self.connection.sctp
        if sctp is not None:
            return sctp.transport.transport.iceGatherer
        iceGather = RTCIceGatherer(iceServers=ICE_SERVERS)
        await iceGather.gather()
        return iceGather

    async def force_ice_negotiation(self):
        iceGatherer = await self.get_ice_gatherer()
        candidates = iceGatherer.getLocalCandidates()
        for candidate in candidates:
            if candidate.sdpMid is None or candidate.sdpMLineIndex is None:
                print(
                    f"Warning: Candidate missing sdpMid or sdpMLineIndex, {candidate}"
                )
                continue

            await self.send_handshake_message(
                MessageType.ICE_CANDIDATE,
                json.dumps({
                    "candidate": candidate_to_sdp(candidate),
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                    "sdpMid": candidate.sdpMid,
                    "usernameFragment": (
                        iceGatherer.getLocalParameters().usernameFragment
                    ),
                }),
            )

    def setup_connection(self):
        """Set up event handlers for the connection"""

        @self.connection.on("signalingstatechange")
        async def on_signalingstatechange():
            print("Signaling state change:", self.connection.signalingState)

        @self.connection.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            print(f"ICE gathering state changed to {self.connection.iceGatheringState}")
            if self.connection.iceGatheringState == "complete":
                await self.force_ice_negotiation()

        @self.connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state changed to: {self.connection.connectionState}")
            match self.connection.connectionState:
                case "closed" | "failed":
                    await self.close()

    def add_video_source(self, source: VideoSource):
        """Add a track to the connection"""

        track = source.get_video_track()
        self.connection.addTrack(track)

    async def send_handshake_message(self, message_type: MessageType, content: str):
        """Send a message to the remote peer through the signaling server"""
        print(
            f"Send Message: {message_type}, "
            f"{self.local_stream_id=} {self.remote_stream_id=}"
        )
        await self.client_session.post(
            f"{API_URL}/signalling/message/submit",
            headers=self.auth.get_headers(),
            json=HandshakeMessage(
                from_id=self.local_stream_id,
                to_id=self.remote_stream_id,
                type=message_type,
                data=content,
            ).model_dump(mode="json"),
        )

    def set_transceiver_direction(self):
        tracks: dict[str, VideoTrack] = {}
        for transceiver in self.connection.getTransceivers():
            transceiver.direction = "sendonly"
            transceiver._offerDirection = "sendonly"
            track = transceiver.sender.track
            if track is not None:
                tracks[track.mid] = track

        for transceiver in self.connection.getTransceivers():
            track = tracks.get(transceiver.mid, None)
            if track is None:
                continue 
            if transceiver.sender.track.id != track.id:
                transceiver.sender.replaceTrack(track)

    async def on_ice(self, ice_message: str):
        """Handle received ICE candidate"""
        if self._closed:
            return
        ice_content = json.loads(ice_message)
        candidate = candidate_from_sdp(ice_content["candidate"])
        candidate.sdpMid = ice_content["sdpMid"]
        candidate.sdpMLineIndex = ice_content["sdpMLineIndex"]
        await self.connection.addIceCandidate(candidate)

    async def on_offer(self, offer: str):
        """Handle received SDP offer"""
        if self._closed:
            print("offer to closed connection")
            return

        offer = RTCSessionDescription(offer, type="offer")
        self.set_transceiver_direction()
        print(
            f"{[{
            "direction":con.direction,
            "currentDirection":con.currentDirection,
            "kind":con.kind,
            "mid": con.mid,
            "trackMid": con.sender.track.mid
        } for con in self.connection.getTransceivers()]}"
        )
        await self.connection.setRemoteDescription(offer)
        self.set_transceiver_direction()
        answer = await self.connection.createAnswer()
        self.set_transceiver_direction()
        await self.connection.setLocalDescription(answer)

        print(f"set local description {self.connection.sctp=}")
        await self.send_handshake_message(MessageType.SDP_ANSWER, answer.sdp)

    async def on_token(self, token: str):
        if self.connection_token.done():
            return
        self.connection_token.set_result(token)

    async def on_answer(self, answer_sdp: str):
        answer = RTCSessionDescription(answer_sdp, type="answer")
        self.set_transceiver_direction()
        await self.connection.setRemoteDescription(answer)
        await self.force_ice_negotiation()

    async def send_offer(self):
        self.set_transceiver_direction()
        offer = await self.connection.createOffer()
        await self.connection.setLocalDescription(offer)
        await self.send_handshake_message(MessageType.SDP_OFFER, offer.sdp)

    async def close(self):
        """Close the connection"""
        if not self._closed:
            self._closed = True
            await self.connection.close()
            self.on_close()
