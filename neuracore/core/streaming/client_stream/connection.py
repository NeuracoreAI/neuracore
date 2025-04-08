import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

from aiohttp import ClientSession
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceGatherer,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.client_stream.event_source import EventSource
from neuracore.core.streaming.client_stream.models import HandshakeMessage, MessageType
from neuracore.core.streaming.client_stream.video_source import VideoSource, VideoTrack

from ...const import API_URL

ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]


@dataclass
class PierToPierConnection:
    id: str
    local_stream_id: str
    remote_stream_id: str
    connection_token: str  # not used yet
    on_close: Callable
    client_session: ClientSession
    loop: asyncio.AbstractEventLoop
    auth: Auth = field(default_factory=get_auth)
    event_sources: set[EventSource] = field(default_factory=set)
    data_channel_callback: dict[str, Callable] = field(default_factory=dict)
    connection: RTCPeerConnection = field(
        default_factory=lambda: RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ICE_SERVERS)
        )
    )
    _closed: bool = False

    async def force_ice_negotiation(self):
        if self.connection.iceGatheringState != "complete":
            print("ICE gathering state is not complete")
            return

        for transceiver in self.connection.getTransceivers():
            iceGatherer: RTCIceGatherer = (
                transceiver.sender.transport.transport.iceGatherer
            )
            for candidate in iceGatherer.getLocalCandidates():
                candidate.sdpMid = transceiver.mid
                mLineIndex = transceiver._get_mline_index()
                candidate.sdpMLineIndex = (
                    int(transceiver.mid) if mLineIndex is None else mLineIndex
                )

                if candidate.sdpMid is None or candidate.sdpMLineIndex is None:
                    print(
                        f"Warning: Candidate missing sdpMid or sdpMLineIndex, {candidate=}, {transceiver=}"
                    )
                    continue
                await self.send_handshake_message(
                    MessageType.ICE_CANDIDATE,
                    json.dumps(
                        {
                            "candidate": f"candidate:{candidate_to_sdp(candidate)}",
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                            "sdpMid": candidate.sdpMid,
                            "usernameFragment": (
                                iceGatherer.getLocalParameters().usernameFragment
                            ),
                        }
                    ),
                )

        if self.connection.sctp is not None:
            iceGatherer = self.connection.sctp.transport.transport.iceGatherer
            for candidate in iceGatherer.getLocalCandidates():
                if candidate.sdpMid is None or candidate.sdpMLineIndex is None:
                    print(
                        f"Warning: Candidate missing sdpMid or sdpMLineIndex, {candidate=}"
                    )
                    continue
                await self.send_handshake_message(
                    MessageType.ICE_CANDIDATE,
                    json.dumps(
                        {
                            "candidate": f"candidate:{candidate_to_sdp(candidate)}",
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                            "sdpMid": candidate.sdpMid,
                            "usernameFragment": (
                                iceGatherer.getLocalParameters().usernameFragment
                            ),
                        }
                    ),
                )

    def setup_connection(self):
        """Set up event handlers for the connection"""

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

    def add_event_source(self, source: EventSource):
        data_channel = self.connection.createDataChannel(source.mid)

        @source.on("event")
        async def on_event(event: str):
            if self._closed:
                return
            if data_channel.readyState != "open":
                return
            data_channel.send(event)

        self.event_sources.add(source)
        self.data_channel_callback[source.mid] = on_event

    async def send_handshake_message(self, message_type: MessageType, content: str):
        """Send a message to the remote peer through the signaling server"""
        print(
            f"Send Message: {message_type}, \n {self.local_stream_id=} {self.remote_stream_id=}"
        )
        await self.client_session.post(
            f"{API_URL}/signalling/message/submit",
            headers=self.auth.get_headers(),
            json=HandshakeMessage(
                connection_id=self.id,
                from_id=self.local_stream_id,
                to_id=self.remote_stream_id,
                type=message_type,
                data=content,
            ).model_dump(mode="json"),
        )

    def fix_mid_ordering(self, when: str = "offer"):
        tracks: dict[str, VideoTrack] = {}
        for transceiver in self.connection.getTransceivers():
            # transceiver.direction = "sendonly"
            # transceiver._offerDirection = "sendonly"
            track = transceiver.sender.track
            if track is not None:
                tracks[track.mid] = track

        for transceiver in self.connection.getTransceivers():
            track = tracks.get(transceiver.mid, None)
            if track is None:
                continue
            if transceiver.sender.track.id != track.id:
                print(f"updating track ordering {when}")
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
        self.fix_mid_ordering("before offer")
        await self.connection.setRemoteDescription(offer)
        self.fix_mid_ordering("after offer")
        answer = await self.connection.createAnswer()
        self.fix_mid_ordering("after answer")
        await self.connection.setLocalDescription(answer)
        await self.send_handshake_message(MessageType.SDP_ANSWER, answer.sdp)

    async def on_answer(self, answer_sdp: str):
        answer = RTCSessionDescription(answer_sdp, type="answer")
        self.fix_mid_ordering("before answer")
        await self.connection.setRemoteDescription(answer)
        await self.force_ice_negotiation()

    async def send_offer(self):
        self.fix_mid_ordering("before offer")
        await self.connection.setLocalDescription(await self.connection.createOffer())
        await self.send_handshake_message(
            MessageType.SDP_OFFER, self.connection.localDescription.sdp
        )

    async def close(self):
        """Close the connection"""
        if not self._closed:
            self._closed = True
            await self.connection.close()
            for source in self.event_sources:
                source.remove_listener("event", self.data_channel_callback[source.mid])
            self.on_close()
