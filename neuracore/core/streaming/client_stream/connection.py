import asyncio
from dataclasses import dataclass, field
import json
from typing import Callable
from aiohttp import ClientSession
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceGatherer,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.client_stream.models import MessageType
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
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
    client_session: ClientSession = field(default_factory=ClientSession)
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

    def setup_connection(self):
        """Set up event handlers for the connection"""

        @self.connection.on("signalingstatechange")
        async def on_signalingstatechange():
            print("Signaling state change:", self.connection.signalingState)

        @self.connection.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            print(f"ICE gathering state changed to {self.connection.iceGatheringState}")
            if self.connection.iceGatheringState != "complete":
                return
                # candidates are not ready

            print("getting local candidates")
            iceGatherer = await self.get_ice_gatherer()
            candidates = iceGatherer.getLocalCandidates()
            for candidate in candidates:
                if candidate.sdpMid is None or candidate.sdpMLineIndex is None:
                    print(
                        f"Warning: Candidate missing sdpMid or sdpMLineIndex, {candidate}"
                    )
                    continue

                await self.send_message(
                    MessageType.ICE_CANDIDATE,
                    json.dumps(
                        {
                            "candidate": candidate_to_sdp(candidate),
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                            "sdpMid": candidate.sdpMid,
                            "usernameFragment": iceGatherer.getLocalParameters().usernameFragment,
                        }
                    ),
                )

        @self.connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state changed to: {self.connection.connectionState}")
            match self.connection.connectionState:
                case "closed" | "failed":
                    await self.close()


    async def add_track(self, track: MediaStreamTrack):
        """Add a track to the connection"""
        self.connection.addTrack(track)

    async def send_message(self, message_type: MessageType, content: str):
        """Send a message to the remote peer through the signaling server"""
        print(
            f"Send Message: {message_type}, {self.local_stream_id=} {self.remote_stream_id=}"
        )
        await self.client_session.post(
            f"{API_URL}/signalling/submit/{message_type.value}/from/{self.local_stream_id}/to/{self.remote_stream_id}",
            headers=self.auth.get_headers(),
            data=content,
        )

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
        await self.connection.setRemoteDescription(offer)

        answer = await self.connection.createAnswer()

        await self.connection.setLocalDescription(answer)
        print(f"set local description {self.connection.sctp=}")
        await self.send_message(MessageType.SDP_ANSWER, answer.sdp)

    async def close(self):
        """Close the connection"""
        if not self._closed:
            self._closed = True
            await self.connection.close()
