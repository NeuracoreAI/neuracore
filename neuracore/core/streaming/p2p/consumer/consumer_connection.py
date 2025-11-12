"""WebRTC peer-to-peer connection management.

This module provides WebRTC-based peer-to-peer connections for real-time
streaming of video and data channels between robot clients and viewers.
Handles SDP negotiation, ICE candidate exchange, and connection lifecycle.
"""

import asyncio
import json
import logging
from typing import List, Optional, Union

from aiohttp import ClientSession
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCIceGatherer,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from neuracore_types import (
    HandshakeMessage,
    MessageType,
    OpenConnectionDetails,
    RobotStreamTrack,
    SyncPoint,
    VideoFormat,
)

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.consumer.ice_models import IceConfig
from neuracore.core.streaming.p2p.consumer.sync_point_parser import (
    merge_sync_points,
    parse_sync_point,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager

logger = logging.getLogger(__name__)


class PierToPierConsumerConnection:
    """WebRTC peer-to-peer connection for streaming robot sensor data.

    Manages the complete lifecycle of a WebRTC connection including SDP
    negotiation, ICE candidate exchange, video tracks, and data channels.
    """

    def __init__(
        self,
        connection_id: str,
        local_stream_id: str,
        remote_stream_id: str,
        ice_config: IceConfig,
        connection_details: OpenConnectionDetails,
        expected_tracks: List[RobotStreamTrack],
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
            ice_config: the details of servers for ICE.
            connection_details: the configuration of the connection established.
            expected_tracks: The tracks that are expected to be received from the
                remote peer on this connection.
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
        self.expected_tracks = expected_tracks

        if connection_details.video_format != VideoFormat.NEURACORE_CUSTOM:
            raise ValueError(
                "Only custom video format supported to avoid dual encoding."
            )

        self.client_session = client_session

        self.connection = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[
                    RTCIceServer(**server.model_dump(mode="json"))
                    for server in ice_config.iceServers
                ]
            )
        )
        self.received_offer_event = asyncio.Event()
        self.handle_offer_lock = asyncio.Lock()
        self.handle_ice_lock = asyncio.Lock()

        self.org_id = org_id or get_current_org()
        self.auth = auth or get_auth()
        self.loop = loop or get_running_loop()
        self.enabled_manager = enabled_manager or EnabledManager(True, loop=self.loop)
        self.enabled_manager.add_listener(EnabledManager.DISABLED, self._on_close)

        # mid/label -> channel
        self.data_channels: dict[str, RTCDataChannel] = {}
        # track id -> channel
        self.connected_data_channels: dict[str, RTCDataChannel] = {}

        @self.connection.on("connectionstatechange")
        def on_connectionstatechange() -> None:
            if self.connection.connectionState in ("closed", "failed"):
                self.close()

        @self.connection.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            self.data_channels[channel.label] = channel
            track = next(
                (track for track in self.expected_tracks if track.mid == channel.label),
                None,
            )
            if track is not None:
                self._connect_data_channel(channel, track)

        self.latest_data = SyncPoint()

    def fully_connected(self) -> bool:
        """Get whether all expected remote tracks are connected.

        Returns:
            True if all expected tracks nodes are connected, False otherwise.
        """
        connected_tracks_ids = set(self.connected_data_channels.keys())

        return all(track.id in connected_tracks_ids for track in self.expected_tracks)

    def _connect_data_channel(
        self, data_channel: RTCDataChannel, track: RobotStreamTrack
    ) -> None:
        """Starts listening to updates from a datachannel.

        Args:
            data_channel: The data channel to start listening to.
            track: The track that this data channel is for.
        """
        assert data_channel.label == track.mid, "Incorrect data channel for track"

        @data_channel.on("message")
        def on_message(message: Union[bytes, str]) -> None:
            assert isinstance(message, str), "Only string messages supported."
            self.latest_data = merge_sync_points(
                self.latest_data, parse_sync_point(message, track)
            )
            self.connected_data_channels[track.id] = data_channel

    def get_latest_data(self) -> SyncPoint:
        """Get the latest data  provided on this connection.

        Each track provided by this node is agglomerated in this sync point.

        you are not guaranteed to get fresh data when called will just provide
        whatever was logged last.

        Returns:
            SyncPoint: the latest data provided on this connection.
        """
        return self.latest_data

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
        tracks: dict[str, MediaStreamTrack] = {}
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
            await self.received_offer_event.wait()

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

    async def on_offer(self, offer_sdp: str) -> None:
        """Handle received SDP answer from remote peer.

        Processes the answer and triggers ICE negotiation. Includes
        proper state validation and error handling.

        Args:
            offer_sdp: SDP answer string from the remote peer
        """
        if self.enabled_manager.is_disabled():
            logger.info("offer to closed connection")
            return

        async with self.handle_offer_lock:
            if self.received_offer_event.is_set():
                return

            if self.connection.signalingState != "stable":
                logger.info("Received offer twice")
                return
            try:
                offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
                self.fix_mid_ordering("before offer")
                await self.connection.setRemoteDescription(offer)
                self.fix_mid_ordering("after offer")
                answer = await self.connection.createAnswer()
                self.fix_mid_ordering("after answer")
                await self.connection.setLocalDescription(answer)
                await self.send_handshake_message(MessageType.SDP_ANSWER, answer.sdp)
                self.received_offer_event.set()
            except Exception:
                logger.warning("Signalling Error: failed to set remote description")

    async def _on_close(self) -> None:
        """Handles the connection close event."""
        await self.connection.close()

    def close(self) -> None:
        """Close the peer-to-peer connection gracefully.

        Closes the WebRTC connection, removes event listeners, and
        triggers the cleanup callback. Ensures proper resource cleanup.
        """
        self.enabled_manager.disable()
