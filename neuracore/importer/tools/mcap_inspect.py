#!/usr/bin/env python3
"""Inspect MCAP files to understand topics, schemas, and encodings."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcap.decoder import DecoderFactory
from mcap.reader import make_reader
from mcap.well_known import MessageEncoding

try:
    from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory

    _HAS_PROTOBUF = True
except Exception:  # noqa: BLE001
    ProtobufDecoderFactory = None
    _HAS_PROTOBUF = False

try:
    from mcap_ros1.decoder import DecoderFactory as Ros1DecoderFactory

    _HAS_ROS1 = True
except Exception:  # noqa: BLE001
    Ros1DecoderFactory = None
    _HAS_ROS1 = False

try:
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    _HAS_ROS2 = True
except Exception:  # noqa: BLE001
    Ros2DecoderFactory = None
    _HAS_ROS2 = False


class JsonDecoderFactory(DecoderFactory):
    """Decode JSON-encoded MCAP messages into Python objects."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a JSON decoder when the encoding matches."""
        if (message_encoding or "").lower() != MessageEncoding.JSON:
            return None

        def _decode(data: bytes) -> Any:
            text = (
                data.decode("utf-8")
                if isinstance(data, (bytes, bytearray, memoryview))
                else data
            )
            return json.loads(text)

        return _decode


class TextDecoderFactory(DecoderFactory):
    """Decode UTF-8 text-encoded MCAP messages."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a text decoder when the encoding matches."""
        if (message_encoding or "").lower() not in {"text", "utf-8", "utf8"}:
            return None

        def _decode(data: bytes) -> str:
            return (
                data.decode("utf-8")
                if isinstance(data, (bytes, bytearray, memoryview))
                else str(data)
            )

        return _decode


def _ns_to_iso(ns: int | None) -> str:
    if ns is None:
        return "n/a"
    seconds = ns / 1e9
    return f"{seconds:.3f}s ({dt.datetime.utcfromtimestamp(seconds).isoformat()}Z)"


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _build_decoder_factories() -> list[DecoderFactory]:
    factories: list[DecoderFactory] = [JsonDecoderFactory(), TextDecoderFactory()]
    if _HAS_PROTOBUF and ProtobufDecoderFactory is not None:
        factories.append(ProtobufDecoderFactory())
    if _HAS_ROS1 and Ros1DecoderFactory is not None:
        factories.append(Ros1DecoderFactory())
    if _HAS_ROS2 and Ros2DecoderFactory is not None:
        factories.append(Ros2DecoderFactory())
    return factories


def _decoder_for(
    factories: list[DecoderFactory], message_encoding: str, schema: Any
) -> Any | None:
    for factory in factories:
        decoder = factory.decoder_for(message_encoding, schema)
        if decoder is not None:
            return decoder
    return None


def _summarize_summary(summary: Any) -> dict[str, Any]:
    stats = _safe_get(summary, "statistics")
    channels = _safe_get(summary, "channels", {}) or {}
    schemas = _safe_get(summary, "schemas", {}) or {}
    metadata = _safe_get(summary, "metadata", {}) or {}

    summary_dict: dict[str, Any] = {
        "statistics": {},
        "channels": [],
        "metadata_keys": list(metadata.keys()) if isinstance(metadata, dict) else [],
    }

    if stats is not None:
        summary_dict["statistics"] = {
            "message_count": _safe_get(stats, "message_count"),
            "message_start_time": _safe_get(stats, "message_start_time"),
            "message_end_time": _safe_get(stats, "message_end_time"),
        }
        if _safe_get(stats, "channel_message_counts") is not None:
            summary_dict["statistics"]["channel_message_counts"] = dict(
                _safe_get(stats, "channel_message_counts")
            )

    for channel_id, channel in channels.items():
        schema = schemas.get(channel.schema_id) if schemas else None
        summary_dict["channels"].append({
            "id": channel_id,
            "topic": channel.topic,
            "message_encoding": channel.message_encoding,
            "schema_id": channel.schema_id,
            "schema_name": getattr(schema, "name", None),
            "schema_encoding": getattr(schema, "encoding", None),
            "schema_data_len": len(getattr(schema, "data", b"") or b""),
        })

    return summary_dict


def inspect_mcap(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect an MCAP file and return a structured summary."""
    path = Path(args.mcap)
    if not path.exists():
        raise SystemExit(f"MCAP file not found: {path}")

    output: dict[str, Any] = {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "summary": None,
        "scan": None,
        "decoders": {
            "protobuf": _HAS_PROTOBUF,
            "ros1": _HAS_ROS1,
            "ros2": _HAS_ROS2,
        },
    }

    factories = _build_decoder_factories()

    with path.open("rb") as f:
        reader = make_reader(f)
        try:
            summary = reader.get_summary()
        except Exception:  # noqa: BLE001
            summary = None

        if summary is not None:
            output["summary"] = _summarize_summary(summary)

        if args.scan or args.sample_per_topic:
            topics = set(args.topics or [])
            if not topics and summary and summary.channels:
                topics = {channel.topic for channel in summary.channels.values()}

            max_messages = args.max_messages
            sample_limit = max(0, args.sample_per_topic)
            samples: dict[str, Any] = {}
            counts: dict[str, int] = defaultdict(int)
            start_times: dict[str, int] = {}
            end_times: dict[str, int] = {}

            decoder_cache: dict[int, Any] = {}
            processed = 0

            for schema, channel, message in reader.iter_messages(
                topics=list(topics) if topics else None, log_time_order=True
            ):
                processed += 1
                counts[channel.topic] += 1
                if channel.topic not in start_times:
                    start_times[channel.topic] = message.log_time
                end_times[channel.topic] = message.log_time

                if sample_limit > 0 and channel.topic not in samples:
                    decoder = decoder_cache.get(message.channel_id)
                    if decoder is None:
                        decoder = _decoder_for(
                            factories, channel.message_encoding, schema
                        )
                        if decoder is not None:
                            decoder_cache[message.channel_id] = decoder

                    if decoder is not None:
                        try:
                            decoded = decoder(message.data)
                            if isinstance(decoded, dict):
                                samples[channel.topic] = {
                                    "type": "dict",
                                    "keys": sorted(decoded.keys()),
                                }
                            else:
                                samples[channel.topic] = {
                                    "type": type(decoded).__name__,
                                    "repr": repr(decoded)[:200],
                                }
                        except Exception as exc:  # noqa: BLE001
                            samples[channel.topic] = {"error": str(exc)}
                    else:
                        samples[channel.topic] = {
                            "error": f"No decoder for {channel.message_encoding}"
                        }

                if max_messages and processed >= max_messages:
                    break
                if sample_limit > 0 and topics and len(samples) == len(topics):
                    if not args.scan:
                        break

            topics_summary: list[dict[str, Any]] = []
            scan = {
                "processed_messages": processed,
                "topics": topics_summary,
                "samples": samples,
            }
            for topic, count in sorted(counts.items()):
                start_ns = start_times.get(topic)
                end_ns = end_times.get(topic)
                duration = None
                if start_ns is not None and end_ns is not None:
                    duration = (end_ns - start_ns) / 1e9 if end_ns >= start_ns else None
                topics_summary.append({
                    "topic": topic,
                    "count": count,
                    "start_time": start_ns,
                    "end_time": end_ns,
                    "duration_s": duration,
                })

            output["scan"] = scan

    return output


def main() -> None:
    """CLI entrypoint for MCAP inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect an MCAP file for topics, schemas, and encodings."
    )
    parser.add_argument("mcap", type=str, help="Path to an MCAP file")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan messages to compute per-topic counts and time ranges.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after reading N messages while scanning (0 = no limit).",
    )
    parser.add_argument(
        "--sample-per-topic",
        type=int,
        default=1,
        help="Decode a sample message per topic when possible (0 disables).",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Optional list of topics to focus on.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON.",
    )

    args = parser.parse_args()
    report = inspect_mcap(args)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"File: {report['file']}")
    print(f"Size: {report['size_bytes']} bytes")
    print(
        "Decoders: protobuf={protobuf} ros1={ros1} ros2={ros2}".format(
            **report["decoders"]
        )
    )

    summary = report.get("summary") or {}
    if summary:
        stats = summary.get("statistics") or {}
        print("\nSummary statistics:")
        print(f"  message_count: {stats.get('message_count')}")
        print(f"  message_start_time: {_ns_to_iso(stats.get('message_start_time'))}")
        print(f"  message_end_time:   {_ns_to_iso(stats.get('message_end_time'))}")

        print("\nChannels:")
        for channel in summary.get("channels", []):
            print(
                "  - topic={topic} encoding={message_encoding} "
                "schema={schema_name}".format(**channel)
            )

    scan = report.get("scan") or {}
    if scan:
        print("\nScan results:")
        print(f"  processed_messages: {scan.get('processed_messages')}")
        for topic_info in scan.get("topics", []):
            start_ns = topic_info.get("start_time")
            end_ns = topic_info.get("end_time")
            duration = topic_info.get("duration_s")
            print(
                f"  - {topic_info['topic']}: count={topic_info['count']} "
                f"duration={duration}s "
                f"start={_ns_to_iso(start_ns)} end={_ns_to_iso(end_ns)}"
            )
        if scan.get("samples"):
            print("\nSamples:")
            for topic, sample in scan["samples"].items():
                print(f"  - {topic}: {sample}")


if __name__ == "__main__":
    main()
