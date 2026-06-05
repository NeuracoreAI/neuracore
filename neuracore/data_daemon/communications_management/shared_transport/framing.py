"""Binary framing for self-describing video transport packets.

One packet is ``magic (4B) + struct header (8B) + JSON metadata + raw chunk``.
This framing is shared by the producer (which writes packets into iceoryx2
slots) and the daemon (which parses packets copied out of those slots). It is
deliberately transport-agnostic so the same bytes could travel over any
zero-copy channel.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping

from neuracore.data_daemon.const import (
    VIDEO_TRANSPORT_PACKET_HEADER_FORMAT,
    VIDEO_TRANSPORT_PACKET_HEADER_SIZE,
    VIDEO_TRANSPORT_PACKET_MAGIC,
)


class PacketTooLarge(ValueError):
    """Raised when an encoded frame cannot fit in a single transport slot."""


def build_video_transport_packet(
    metadata: Mapping[str, object],
    chunk: bytes | bytearray | memoryview,
) -> bytes:
    """Build one self-describing transport packet."""
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload = bytes(chunk)
    return (
        struct.pack(
            VIDEO_TRANSPORT_PACKET_HEADER_FORMAT,
            VIDEO_TRANSPORT_PACKET_MAGIC,
            len(metadata_bytes),
            len(payload),
        )
        + metadata_bytes
        + payload
    )


def build_video_transport_packet_metadata(
    metadata: Mapping[str, object],
    chunk: bytes | bytearray | memoryview,
) -> tuple[bytes, int]:
    """Return serialized metadata plus total packet length without copying chunk."""
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    chunk_len = len(chunk)
    packet_length = VIDEO_TRANSPORT_PACKET_HEADER_SIZE + len(metadata_bytes) + chunk_len
    return metadata_bytes, packet_length


def parse_video_transport_packet(packet: bytes) -> tuple[dict[str, object], bytes]:
    """Parse one self-describing packet, returning its metadata and chunk."""
    metadata, chunk_start, chunk_end = parse_video_transport_packet_view(
        memoryview(packet)
    )
    return metadata, packet[chunk_start:chunk_end]


def parse_video_transport_packet_view(
    packet: memoryview,
) -> tuple[dict[str, object], int, int]:
    """Parse one packet view without copying the payload chunk."""
    if len(packet) < VIDEO_TRANSPORT_PACKET_HEADER_SIZE:
        raise ValueError("Transport packet shorter than record header")
    magic, metadata_len, chunk_len = struct.unpack(
        VIDEO_TRANSPORT_PACKET_HEADER_FORMAT,
        packet[:VIDEO_TRANSPORT_PACKET_HEADER_SIZE],
    )
    if magic != VIDEO_TRANSPORT_PACKET_MAGIC:
        raise ValueError("Transport packet missing video transport magic")
    expected = VIDEO_TRANSPORT_PACKET_HEADER_SIZE + metadata_len + chunk_len
    if len(packet) < expected:
        raise ValueError("Transport packet shorter than declared lengths")
    if len(packet) > expected:
        raise ValueError("Transport packet contains trailing bytes")
    metadata_start = VIDEO_TRANSPORT_PACKET_HEADER_SIZE
    chunk_start = metadata_start + metadata_len
    metadata = json.loads(packet[metadata_start:chunk_start].tobytes().decode("utf-8"))
    return metadata, chunk_start, expected
