"""WebRTC peer-to-peer connection management.

This module provides WebRTC-based peer-to-peer connections for real-time
streaming of video and data channels between robot clients and viewers.
Handles SDP negotiation, ICE candidate exchange, and connection lifecycle.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

from aiohttp import ClientSession
from aiortc import (
    RTCConfiguration,
    RTCIceGatherer,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from neuracore_types import HandshakeMessage, MessageType, OpenConnectionDetails

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.json_source import JSONSource
from neuracore.core.streaming.p2p.provider.video_source import VideoSource, VideoTrack

ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]

logger = logging.getLogger(__name__)


class PierToPierProviderConnection:
    """WebRTC peer-to-peer connection for streaming robot sensor data.

    Manages the complete lifecycle of a WebRTC connection including SDP
    negotiation, ICE candidate exchange, video tracks, and data channels.
    """

    def __init__(
        self,
        connection_id: str,
        local_stream_id: str,
        remote_stream_id: str,
        connection_details: OpenConnectionDetails,
        client_session: ClientSession = None,
        org_id: Optional[str] = None,
        enabled_manager: Optional[EnabledManager] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        auth: Optional[Auth] = None,
    ) -> None:
        """Initialize the connection.

        Args:
            connection_id: the unique identifier for this connection
            local_stream_id: the unique identifier for this node.
            remote_stream_id: the unique identifier for the remote node.
            connection_details: the configuration of the connection established.
            client_session: The http session to use.
            auth: The auth instance used to connect to the signalling server or
                defaults to the global auth provider if not provided.
            org_id: the organization to receive signalling from. If not provided
                defaults to the current org.
            loop: the event loop to run on. Defaults to the running loop if not
                provided.
            enabled_manager: The enabled manager for whether this should be
                consuming. Defaults to a new enabled manger if not provided.

        """
        self.id = connection_id
        self.local_stream_id = local_stream_id
        self.remote_stream_id = remote_stream_id
        self.connection_details = connection_details
        self.client_session = client_session

        self.connection = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ICE_SERVERS)
        )
        self.received_answer_event = asyncio.Event()
        self.handle_answer_lock = asyncio.Lock()
        self.handle_ice_lock = asyncio.Lock()
        self.send_offer_lock = asyncio.Lock()
        self.has_sent_offer: bool = False

        self.org_id = org_id or get_current_org()
        self.auth = auth or get_auth()
        self.loop = loop or get_running_loop()
        self.enabled_manager = enabled_manager or EnabledManager(True, loop=self.loop)
        self.enabled_manager.add_listener(EnabledManager.DISABLED, self._on_close)

        self.event_sources: set[JSONSource] = set()
        self.data_channel_callback: dict[str, Callable] = dict()

        @self.connection.on("connectionstatechange")
        def on_connectionstatechange() -> None:
            if self.connection.connectionState in ("closed", "failed"):
                self.close()

    async def force_ice_negotiation(self) -> None:
        """Force ICE candidate negotiation for all transceivers.

        Manually sends ICE candidates for all active transceivers and SCTP
        transport when ICE gathering is complete. This ensures proper
        connectivity establishment.
        """
        if self.connection.iceGatheringState != "complete":
            logger.warning("ICE gathering state is not complete")
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
                    logger.warning(
                        "Warning: Candidate missing sdpMid or sdpMLineIndex, "
                        f"{candidate=}, {transceiver=}"
                    )
                    continue
                await self.send_handshake_message(
                    MessageType.ICE_CANDIDATE,
                    json.dumps({
                        "candidate": f"candidate:{candidate_to_sdp(candidate)}",
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                        "sdpMid": candidate.sdpMid,
                        "usernameFragment": (
                            iceGatherer.getLocalParameters().usernameFragment
                        ),
                    }),
                )

        if self.connection.sctp is not None:
            iceGatherer = self.connection.sctp.transport.transport.iceGatherer
            for candidate in iceGatherer.getLocalCandidates():
                if candidate.sdpMid is None or candidate.sdpMLineIndex is None:
                    # TODO: fix sctp ice candidates
                    continue
                await self.send_handshake_message(
                    MessageType.ICE_CANDIDATE,
                    json.dumps({
                        "candidate": f"candidate:{candidate_to_sdp(candidate)}",
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                        "sdpMid": candidate.sdpMid,
                        "usernameFragment": (
                            iceGatherer.getLocalParameters().usernameFragment
                        ),
                    }),
                )

    def add_video_source(self, source: VideoSource) -> None:
        """Add a video source to the connection.

        Args:
            source: Video source containing the track to be streamed
        """
        track = source.get_video_track()
        self.connection.addTrack(track)

    def add_event_source(self, source: JSONSource) -> None:
        """Add a JSON event source to the connection via data channel.

        Creates a data channel for the source and sets up listeners to
        send state updates when the data channel is open.

        Args:
            source: JSON source for streaming structured data
        """
        data_channel = self.connection.createDataChannel(source.mid)

        async def on_update(state: str) -> None:
            if self.enabled_manager.is_disabled():
                return
            if data_channel.readyState != "open":
                return
            data_channel.send(state)

        def on_open() -> None:
            last_state = source.get_last_state()
            if last_state:
                data_channel.send(last_state)
            data_channel.remove_listener("open", on_open)

        data_channel.add_listener("open", on_open)

        self.event_sources.add(source)
        source.add_listener(source.STATE_UPDATED_EVENT, on_update)
        self.data_channel_callback[source.mid] = on_update

    async def send_handshake_message(
        self, message_type: MessageType, content: str
    ) -> None:
        """Send a signaling message to the remote peer.

        Args:
            message_type: Type of signaling message
                (SDP offer/answer, ICE candidate, etc.)
            content: Message payload content

        Raises:
            ConfigError: If there is an error trying to get the current org
        """
        await self.client_session.post(
            f"{API_URL}/org/{self.org_id}/signalling/message/submit",
            headers=self.auth.get_headers(),
            json=HandshakeMessage(
                connection_id=self.id,
                from_id=self.local_stream_id,
                to_id=self.remote_stream_id,
                type=message_type,
                data=content,
            ).model_dump(mode="json"),
        )

    def fix_mid_ordering(self, when: str = "offer") -> None:
        """Fix media ID ordering for transceivers.

        Ensures that transceivers have the correct track assignments
        based on their media IDs. This is necessary for proper SDP
        negotiation and track correlation.

        Args:
            when: Description of when this fix is being applied (for logging)
        """
        tracks: dict[str, VideoTrack] = {}
        for transceiver in self.connection.getTransceivers():
            track = transceiver.sender.track
            if track is not None:
                tracks[track.mid] = track

        for transceiver in self.connection.getTransceivers():
            track = tracks.get(transceiver.mid, None)
            if track is None:
                continue
            if transceiver.sender.track.id != track.id:
                logger.info(f"updating track ordering {when}")
                transceiver.sender.replaceTrack(track)

    async def on_ice(self, ice_message: str) -> None:
        """Handle received ICE candidate from remote peer.

        Args:
            ice_message: JSON string containing ICE candidate information
        """
        async with self.handle_ice_lock:
            await self.received_answer_event.wait()

            if self.enabled_manager.is_disabled():
                return
            try:
                ice_content = json.loads(ice_message)
                candidate = candidate_from_sdp(ice_content["candidate"])
                candidate.sdpMid = ice_content["sdpMid"]
                candidate.sdpMLineIndex = ice_content["sdpMLineIndex"]
                await self.connection.addIceCandidate(candidate)
            except Exception:
                logging.warning("Signalling Error: failed to add ice candidate")

    async def on_answer(self, answer_sdp: str) -> None:
        """Handle received SDP answer from remote peer.

        Processes the answer and triggers ICE negotiation. Includes
        proper state validation and error handling.

        Args:
            answer_sdp: SDP answer string from the remote peer
        """
        if self.enabled_manager.is_disabled():
            logger.info("answer to closed connection")
            return

        async with self.handle_answer_lock:
            if self.received_answer_event.is_set():
                return

            if self.connection.signalingState != "have-local-offer":
                logger.info("Received answer before offer.")
                return
            try:
                answer = RTCSessionDescription(answer_sdp, type="answer")
                self.fix_mid_ordering("before answer")
                await self.connection.setRemoteDescription(answer)
                await self.force_ice_negotiation()
                self.received_answer_event.set()

            except Exception:
                logger.warning("Signalling Error: failed to set remote description")

    async def send_offer(self) -> None:
        """Send SDP offer to remote peer.

        Creates and sends an SDP offer through the signaling server.
        Includes proper state validation and error handling with retry logic.
        """
        if self.enabled_manager.is_disabled():
            logger.warning("Cannot send offer from closed connection")
            return

        async with self.send_offer_lock:
            if self.has_sent_offer:
                return

            if self.connection.signalingState != "stable":
                logger.warning("Not ready to send offer")
                return

            self.fix_mid_ordering("before offer")
            try:
                await self.connection.setLocalDescription(
                    await self.connection.createOffer()
                )
                await self.send_handshake_message(
                    MessageType.SDP_OFFER, self.connection.localDescription.sdp
                )

                await asyncio.wait_for(self.received_answer_event.wait(), timeout=30)
                self.has_sent_offer = True
            except asyncio.TimeoutError:
                # waiting 30 seconds for answer -> assuming peer is gone
                self.close()
            except Exception:
                logger.info("Signalling Error: Failed to send offer")

    async def _on_close(self) -> None:
        """Handles the connection close event."""
        await self.connection.close()
        for source in self.event_sources:
            source.remove_listener(
                source.STATE_UPDATED_EVENT, self.data_channel_callback[source.mid]
            )

    def close(self) -> None:
        """Close the peer-to-peer connection gracefully.

        Closes the WebRTC connection, removes event listeners, and
        triggers the cleanup callback. Ensures proper resource cleanup.
        """
        self.enabled_manager.disable()
