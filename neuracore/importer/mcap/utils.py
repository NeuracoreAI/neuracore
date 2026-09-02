"""MCAP reader utilities for the Neuracore importer.

Thin adapter over Foxglove's ``mcap`` package: decodes messages and normalises
them into the shape expected by Neuracore's mapping pipeline.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import logging
import pkgutil
import time
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mcap.decoder import DecoderFactory
from mcap.reader import McapReader
from mcap.well_known import MessageEncoding
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.importer.data_config import DataFormat
from neuracore_types.nc_data import DatasetImportConfig
from neuracore_types.nc_data.nc_data import MappingItem

from neuracore.core.utils.depth_utils import MAX_DEPTH
from neuracore.importer.core.exceptions import ImportError


def split_topic_path(source: str) -> tuple[str, list[str]]:
    """Split a source path into topic and nested field components."""
    value = source.strip()
    if not value:
        raise ImportError("Source must include a topic.")

    topic, sep, subpath = value.partition(".")
    if not topic:
        raise ImportError(f"Invalid source '{source}': topic segment is empty.")

    path = [part for part in subpath.split(".") if part] if sep else []
    return topic, path


def resolve_path(data: Any, path: list[str]) -> Any:
    """Resolve a nested path against dict/object/list payloads."""
    current = data
    for part in path:
        current = _resolve_path_part(current, part)
    return current


def _resolve_path_part(data: Any, part: str) -> Any:
    """Resolve one path segment from dicts, objects, or indexable containers."""
    if isinstance(data, dict):
        if part in data:
            return data[part]
        if part.isdigit():
            numeric_key = int(part)
            if numeric_key in data:
                return data[numeric_key]
        raise ImportError(f"Key '{part}' not found while resolving message path.")

    if hasattr(data, part):
        return getattr(data, part)

    if part.isdigit():
        index = int(part)
        try:
            return data[index]
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                f"Index {index} is unavailable while resolving message path: {exc}"
            ) from exc

    try:
        return data[part]
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"Failed to resolve '{part}' from payload: {exc}") from exc


@dataclass(frozen=True, slots=True)
class TopicConfig:
    """Resolved topic configuration for one mapping entry group."""

    data_type: DataType
    import_config: Any
    source_path: list[str]
    mapping_item: MappingItem | None = None
    item_base_path: list[str] | None = None


TopicMap = dict[str, list[TopicConfig]]


def build_topic_map(dataset_config: DatasetImportConfig) -> TopicMap:
    """Build topic lookup tables from dataset mapping configuration."""
    topic_map: TopicMap = {}

    for data_type, import_config in dataset_config.data_import_config.items():
        source = (import_config.source or "").strip()
        mapping = list(import_config.mapping)

        absolute_items = [
            item
            for item in mapping
            if item.source_name and item.source_name.startswith("/")
        ]
        relative_items = [
            item
            for item in mapping
            if not (item.source_name and item.source_name.startswith("/"))
        ]

        if relative_items:
            if not source:
                raise ImportError(
                    f"Missing source for data type '{data_type.value}'. "
                    "Relative mapping entries require a base source path."
                )
            topic, subpath = split_topic_path(source)
            topic_map.setdefault(topic, []).append(
                TopicConfig(
                    data_type=data_type,
                    import_config=_copy_import_config_with_mapping(
                        import_config,
                        relative_items,
                    ),
                    source_path=subpath,
                )
            )

        for item in absolute_items:
            item_topic, item_subpath = split_topic_path(item.source_name)
            topic_map.setdefault(item_topic, []).append(
                TopicConfig(
                    data_type=data_type,
                    import_config=import_config,
                    source_path=[],
                    mapping_item=item,
                    item_base_path=item_subpath,
                )
            )

    if not topic_map:
        raise ImportError("No data_import_config entries found for MCAP import.")

    return topic_map


def _copy_import_config_with_mapping(
    import_config: Any,
    mapping: list[MappingItem],
) -> Any:
    """Clone an import config while replacing mapping entries."""
    mapping_copy = list(mapping)
    if hasattr(import_config, "model_copy"):
        return import_config.model_copy(update={"mapping": mapping_copy})
    cloned = copy(import_config)
    setattr(cloned, "mapping", mapping_copy)
    return cloned


def get_mcap_topics(topic_map: TopicMap) -> list[str]:
    """Return all configured MCAP topics in deterministic order."""
    return sorted(topic_map)


@dataclass(frozen=True, slots=True)
class MCAPSourceEvent:
    """One extracted MCAP source message ready for _log_data."""

    data_type: DataType
    source_data: Any
    item: MappingItem
    format: DataFormat
    timestamp: float
    source_topic: str = ""


@dataclass(frozen=True, slots=True)
class DecodedMCAPMessage:
    """Decoded MCAP message plus the metadata needed by Neuracore.

    topic: MCAP channel topic name. In Neuracore importer config this is the
        source topic used to select mapping rules.
    log_time_ns: MCAP message log timestamp in nanoseconds.
    publish_time_ns: MCAP message publish timestamp in nanoseconds.
    timestamp_ns: Resolved timestamp in nanoseconds, preferring log time.
    data: Decoded message data returned by the MCAP decoder factory. This is
        the source object Neuracore mappings read from.
    """

    topic: str
    log_time_ns: int
    publish_time_ns: int
    timestamp_ns: int
    data: Any


def read_mcap_header(reader: McapReader) -> Any | None:
    """Return the MCAP header when available."""
    try:
        return reader.get_header()
    except Exception:  # noqa: BLE001
        return None


def read_mcap_summary(reader: McapReader) -> Any | None:
    """Return the MCAP summary when available."""
    try:
        return reader.get_summary()
    except Exception:  # noqa: BLE001
        return None


def log_mcap_header(header: Any | None, logger: logging.Logger) -> None:
    """Log basic MCAP header details for diagnostics."""
    if header is None:
        return
    profile = getattr(header, "profile", "") or "<empty>"
    library = getattr(header, "library", "") or "<empty>"
    logger.debug(
        f"MCAP header | profile={profile} | library={library}",
    )


def log_mcap_summary_details(summary: Any | None, logger: logging.Logger) -> None:
    """Log non-message record counts that this importer does not process."""
    if summary is None:
        return
    attachment_count = len(getattr(summary, "attachment_indexes", []) or [])
    metadata_count = len(getattr(summary, "metadata_indexes", []) or [])
    if attachment_count > 0 or metadata_count > 0:
        logger.debug(
            f"MCAP includes {attachment_count} attachment(s) and {metadata_count} "
            "metadata record(s); the importer currently processes message records "
            "only.",
        )


def resolve_timestamp_ns(
    *,
    log_time_ns: int,
    publish_time_ns: int,
) -> int:
    """Resolve message timestamp in nanoseconds from log/publish time."""
    if log_time_ns > 0:
        return log_time_ns
    if publish_time_ns > 0:
        return publish_time_ns
    return time.time_ns()


def iter_decoded_mcap_messages(
    reader: McapReader,
    topics: list[str],
) -> Iterator[DecodedMCAPMessage]:
    """Yield decoded MCAP messages in log-time order for configured topics.

    The MCAP reader yields:
    schema: message type information used by the decoder.
    channel: topic metadata such as topic name, message encoding, and schema id.
    raw_message: the MCAP message record with timestamps and serialized bytes.
    decoded_data: raw_message.data after the decoder factory has decoded it.
    """
    for _schema, channel, raw_message, decoded_data in reader.iter_decoded_messages(
        topics=topics,
        log_time_order=True,
    ):
        log_time_ns = int(getattr(raw_message, "log_time", 0) or 0)
        publish_time_ns = int(getattr(raw_message, "publish_time", 0) or 0)
        yield DecodedMCAPMessage(
            topic=str(getattr(channel, "topic", "") or ""),
            log_time_ns=log_time_ns,
            publish_time_ns=publish_time_ns,
            timestamp_ns=resolve_timestamp_ns(
                log_time_ns=log_time_ns,
                publish_time_ns=publish_time_ns,
            ),
            data=decoded_data,
        )


def estimate_total_messages(summary: Any | None, topics: list[str]) -> int | None:
    """Estimate total message count from MCAP summary statistics."""
    if (
        summary is None
        or not getattr(summary, "statistics", None)
        or not summary.statistics.channel_message_counts
    ):
        return None

    counts = summary.statistics.channel_message_counts
    total = 0
    for channel_id, channel in summary.channels.items():
        if channel.topic in topics:
            total += int(counts.get(channel_id, 0))
    return total if total > 0 else None


def topic_schema_names(summary: Any | None, topics: list[str]) -> dict[str, str]:
    """Map each requested topic to its schema type name in one MCAP summary.

    Read from the summary section, so no message is decoded. Topics absent from
    the file, and channels carrying no schema, are left out of the result.

    Args:
        summary: The MCAP summary, or None when the file has no summary section.
        topics: The topics to look up.

    Returns:
        dict[str, str]: Schema type name keyed by topic.
    """
    if summary is None or not getattr(summary, "channels", None):
        return {}

    schemas = getattr(summary, "schemas", None) or {}
    requested = set(topics)
    names: dict[str, str] = {}
    for channel in summary.channels.values():
        topic = getattr(channel, "topic", "")
        if topic not in requested:
            continue
        schema = schemas.get(getattr(channel, "schema_id", 0))
        schema_name = getattr(schema, "name", "") if schema is not None else ""
        if schema_name:
            names[topic] = schema_name
    return names


def validate_requested_topics(summary: Any | None, topics: list[str]) -> None:
    """Validate configured topics against the MCAP summary when available."""
    if summary is None or not getattr(summary, "channels", None) or not topics:
        return

    available_topics = {channel.topic for channel in summary.channels.values()}
    missing = sorted(topic for topic in topics if topic not in available_topics)
    if not missing:
        return

    shown_available = ", ".join(sorted(available_topics)[:20])
    raise ImportError(
        "Configured topic(s) not present in MCAP: "
        f"{', '.join(missing)}. "
        f"Available topics include: {shown_available}"
    )


try:
    import cbor2

    HAS_CBOR = True
except Exception:  # noqa: BLE001
    cbor2 = None
    HAS_CBOR = False

try:
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.message import Message as ProtobufMessage

    HAS_PROTOBUF_RUNTIME = True
except Exception:  # noqa: BLE001
    FileDescriptorSet = None
    MessageToDict = None
    ProtobufMessage = None
    HAS_PROTOBUF_RUNTIME = False

try:
    from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory

    HAS_PROTOBUF_FACTORY = True
except Exception:  # noqa: BLE001
    ProtobufDecoderFactory = None
    HAS_PROTOBUF_FACTORY = False

try:
    from mcap_ros1.decoder import DecoderFactory as Ros1DecoderFactory

    HAS_ROS1_FACTORY = True
except Exception:  # noqa: BLE001
    Ros1DecoderFactory = None
    HAS_ROS1_FACTORY = False

try:
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    HAS_ROS2_FACTORY = True
except Exception:  # noqa: BLE001
    Ros2DecoderFactory = None
    HAS_ROS2_FACTORY = False

try:
    from PIL import Image

    HAS_PIL = True
except Exception:  # noqa: BLE001
    Image = None
    HAS_PIL = False

try:
    import av

    # Keep libav output out of the importer logs; decode failures are
    # reported by this module instead.
    av.logging.set_level(av.logging.ERROR)
    HAS_AV = True
except Exception:  # noqa: BLE001
    av = None
    HAS_AV = False

_DISCOVERED_DECODER_FACTORY_CLASSES: list[type[DecoderFactory]] | None = None


def _to_bytes(data: Any) -> bytes:
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


class JSONDecoderFactory(DecoderFactory):
    """Decode ``json`` message-encoded payloads."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a decoder when the channel encoding is JSON."""
        if (message_encoding or "").lower() != MessageEncoding.JSON.lower():
            return None

        def _decode(data: bytes) -> Any:
            return json.loads(_to_bytes(data).decode("utf-8"))

        return _decode


class TextDecoderFactory(DecoderFactory):
    """Decode UTF-8 text payloads."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a decoder for common UTF-8 text encodings."""
        if (message_encoding or "").lower() not in {"text", "utf-8", "utf8"}:
            return None

        def _decode(data: bytes) -> str:
            return _to_bytes(data).decode("utf-8")

        return _decode


class CborDecoderFactory(DecoderFactory):
    """Decode ``cbor`` payloads when ``cbor2`` is installed."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a decoder when CBOR is available and requested."""
        if (message_encoding or "").lower() != MessageEncoding.CBOR.lower():
            return None
        if not HAS_CBOR or cbor2 is None:
            return None

        def _decode(data: bytes) -> Any:
            return cbor2.loads(_to_bytes(data))

        return _decode


# Some MCAP writers emit Schema records with a few stray bytes appended to the
# serialised FileDescriptorSet. Those records are recoverable by truncating the
# trailing garbage; this is the largest overrun we will try to undo.
MAX_TRAILING_SCHEMA_BYTES = 64


# FileDescriptorSet has exactly one field: `file`, number 1, length-delimited.
_PROTOBUF_FDS_FILE_TAG = 0x0A


def _read_varint(data: bytes, offset: int) -> tuple[int | None, int]:
    """Read a protobuf varint, returning (value, next offset) or (None, offset)."""
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            return None, offset
    return None, offset


def _file_descriptor_set_length(data: bytes) -> int:
    """Length of the leading run of well-formed ``FileDescriptorProto`` records.

    Walking the wire format finds the exact byte where a truncated
    ``FileDescriptorSet`` stops being well formed. Probing with
    ``FileDescriptorSet.FromString`` cannot: trailing junk is often itself valid
    wire format (any byte in ``0x08..0x7f`` opens a varint field), and protobuf
    silently keeps such bytes as unknown fields instead of rejecting them.
    """
    offset = 0
    while offset < len(data):
        tag, after_tag = _read_varint(data, offset)
        if tag != _PROTOBUF_FDS_FILE_TAG:
            break
        length, after_length = _read_varint(data, after_tag)
        if length is None or after_length + length > len(data):
            break
        offset = after_length + length
    return offset


def repair_protobuf_schema_data(data: bytes) -> bytes | None:
    """Recover a usable ``FileDescriptorSet`` from a malformed schema blob.

    Some MCAP writers append a few stray bytes after the serialised descriptor
    set. This drops that tail.

    Args:
        data: Raw bytes of an MCAP protobuf Schema record.

    Returns:
        The longest prefix of ``data`` that is a well-formed, non-empty
        ``FileDescriptorSet``, or None when there is no such prefix or when more
        than ``MAX_TRAILING_SCHEMA_BYTES`` would have to be discarded. Healthy
        input is returned unchanged.
    """
    if not HAS_PROTOBUF_RUNTIME or FileDescriptorSet is None:
        return None

    end = _file_descriptor_set_length(data)
    if end == 0 or len(data) - end > MAX_TRAILING_SCHEMA_BYTES:
        return None

    candidate = data[:end]
    try:
        descriptor_set = FileDescriptorSet.FromString(candidate)
    except Exception:  # noqa: BLE001 - malformed beyond a trailing-byte overrun
        return None
    return candidate if descriptor_set.file else None


class TolerantProtobufDecoderFactory(DecoderFactory):
    """Protobuf decoder factory that survives slightly malformed schemas.

    ``mcap_protobuf``'s factory raises when a Schema record does not parse as a
    ``FileDescriptorSet``, and that exception escapes
    ``McapReader.iter_decoded_messages`` and aborts the whole episode. This
    wrapper first tries the upstream factory, then retries once against a
    repaired copy of the schema, and finally gives up quietly so that the
    raw-byte fallback can still apply.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Wrap the upstream protobuf decoder factory.

        Args:
            logger: Optional logger used to report repaired schemas.
        """
        self._inner = ProtobufDecoderFactory() if HAS_PROTOBUF_FACTORY else None
        self._logger = logger
        self._reported_schema_ids: set[int] = set()

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a protobuf decoder, repairing the schema record if needed."""
        if self._inner is None:
            return None
        try:
            return self._inner.decoder_for(message_encoding, schema)
        except Exception as exc:  # noqa: BLE001 - malformed schema record
            if schema is None or not getattr(schema, "data", None):
                return None
            repaired = repair_protobuf_schema_data(schema.data)
            if repaired is None:
                self._report(
                    f"MCAP schema '{getattr(schema, 'name', '<unnamed>')}' "
                    f"(id={getattr(schema, 'id', '?')}) could not be parsed as a "
                    f"protobuf FileDescriptorSet: {exc}. Falling back to raw bytes.",
                    schema,
                )
                return None
            try:
                decoder = self._inner.decoder_for(
                    message_encoding,
                    dataclasses.replace(schema, data=repaired),
                )
            except Exception as retry_exc:  # noqa: BLE001 - repair did not help
                self._report(
                    f"MCAP schema '{getattr(schema, 'name', '<unnamed>')}' "
                    f"(id={getattr(schema, 'id', '?')}) stayed unusable after "
                    f"repair: {retry_exc}. Falling back to raw bytes.",
                    schema,
                )
                return None
            dropped = len(schema.data) - len(repaired)
            self._report(
                f"Repaired malformed MCAP protobuf schema "
                f"'{getattr(schema, 'name', '<unnamed>')}' "
                f"(id={getattr(schema, 'id', '?')}) by dropping "
                f"{dropped} trailing byte(s).",
                schema,
            )
            return decoder

    def _report(self, message: str, schema: Any) -> None:
        """Log ``message`` once per schema id."""
        schema_id = getattr(schema, "id", None)
        if schema_id in self._reported_schema_ids:
            return
        if schema_id is not None:
            self._reported_schema_ids.add(schema_id)
        if self._logger is not None:
            self._logger.warning(message)


class RawPassthroughDecoderFactory(DecoderFactory):
    """Final fallback factory that returns raw bytes.

    Register this last so all format-specific factories get first chance.
    """

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a fallback decoder that passes bytes through unchanged."""

        def _decode(data: bytes) -> bytes:
            return _to_bytes(data)

        return _decode


def _iter_candidate_decoder_modules() -> set[str]:
    modules = {
        module.name
        for module in pkgutil.iter_modules()
        if module.name.startswith("mcap_")
    }
    try:
        for distribution in importlib_metadata.distributions():
            name = (distribution.metadata["Name"] or "").strip().lower()
            if name.startswith("mcap-"):
                modules.add(name.replace("-", "_"))
    except Exception:  # noqa: BLE001
        pass
    return modules


def _load_decoder_factory_class(module_name: str) -> type[DecoderFactory] | None:
    try:
        decoder_module = importlib.import_module(f"{module_name}.decoder")
    except Exception:  # noqa: BLE001
        return None

    decoder_factory_cls = getattr(decoder_module, "DecoderFactory", None)
    if (
        decoder_factory_cls is None
        or not isinstance(decoder_factory_cls, type)
        or not issubclass(decoder_factory_cls, DecoderFactory)
    ):
        return None
    return decoder_factory_cls


def _discover_decoder_factory_classes(
    logger: logging.Logger | None = None,
) -> list[type[DecoderFactory]]:
    global _DISCOVERED_DECODER_FACTORY_CLASSES

    if _DISCOVERED_DECODER_FACTORY_CLASSES is not None:
        return list(_DISCOVERED_DECODER_FACTORY_CLASSES)

    classes: list[type[DecoderFactory]] = []
    for module_name in sorted(_iter_candidate_decoder_modules()):
        factory_cls = _load_decoder_factory_class(module_name)
        if factory_cls is None:
            continue
        classes.append(factory_cls)

    _DISCOVERED_DECODER_FACTORY_CLASSES = classes

    if logger is not None:
        logger.info(f"Discovered {len(classes)} MCAP decoder plugin class(es).")

    return list(_DISCOVERED_DECODER_FACTORY_CLASSES)


def discover_decoder_factories(
    logger: logging.Logger | None = None,
) -> list[DecoderFactory]:
    """Discover and instantiate optional MCAP decoder factories."""
    factories: list[DecoderFactory] = []
    for factory_cls in _discover_decoder_factory_classes(logger=logger):
        try:
            factories.append(factory_cls())
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.debug(
                    "Failed to instantiate discovered MCAP decoder factory "
                    f"'{factory_cls}': {exc}",
                    exc_info=True,
                )
    return factories


def list_decoder_factories(
    *,
    enable_discovery: bool = False,
    include_raw_fallback: bool = True,
    logger: logging.Logger | None = None,
) -> list[DecoderFactory]:
    """Build decoder factories used by ``make_reader(..., decoder_factories=...)``."""
    factories: list[DecoderFactory] = [
        JSONDecoderFactory(),
        TextDecoderFactory(),
        CborDecoderFactory(),
    ]

    if HAS_PROTOBUF_FACTORY and ProtobufDecoderFactory is not None:
        factories.append(TolerantProtobufDecoderFactory(logger=logger))
    if HAS_ROS1_FACTORY and Ros1DecoderFactory is not None:
        factories.append(Ros1DecoderFactory())
    if HAS_ROS2_FACTORY and Ros2DecoderFactory is not None:
        factories.append(Ros2DecoderFactory())

    if enable_discovery:
        seen = {factory.__class__ for factory in factories}
        for factory in discover_decoder_factories(logger=logger):
            if factory.__class__ in seen:
                continue
            factories.append(factory)
            seen.add(factory.__class__)

    if include_raw_fallback:
        factories.append(RawPassthroughDecoderFactory())

    if logger is not None:
        factory_names = [
            f"{factory.__class__.__module__}.{factory.__class__.__qualname__}"
            for factory in factories
        ]
        logger.debug(
            f"Configured MCAP decoder factories: {factory_names}",
        )

    return factories


def _channel_has_decoder_support(
    message_encoding: str,
    schema: Any | None,
    decoder_factories: list[DecoderFactory],
) -> bool:
    for factory in decoder_factories:
        if isinstance(factory, RawPassthroughDecoderFactory):
            continue
        try:
            if factory.decoder_for(message_encoding, schema) is not None:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def validate_channel_decoder_support(
    summary: Any | None,
    topics: list[str],
    decoder_factories: list[DecoderFactory],
    logger: logging.Logger,
) -> None:
    """Warn when configured topics lack non-raw decoder support."""
    if summary is None or not getattr(summary, "channels", None):
        return

    schemas = getattr(summary, "schemas", {}) or {}
    for channel in summary.channels.values():
        if channel.topic not in topics:
            continue

        encoding = str(channel.message_encoding or "")
        schema = schemas.get(getattr(channel, "schema_id", 0), None)

        if encoding.lower() == MessageEncoding.CBOR.lower() and not HAS_CBOR:
            logger.warning(
                f"MCAP topic '{channel.topic}' uses CBOR but cbor2 is unavailable.",
            )

        if _channel_has_decoder_support(encoding, schema, decoder_factories):
            continue

        logger.warning(
            f"No structured decoder available for topic '{channel.topic}' "
            f"(encoding={encoding or '<empty>'}). Using raw-byte fallback.",
        )


def _read_field(data: Any, name: str) -> Any:
    """Read a field from mapping-like or object-like message payloads."""
    if isinstance(data, dict):
        return data.get(name)
    if hasattr(data, name):
        return getattr(data, name)
    return None


def _is_byte_list(data: Any) -> bool:
    """Return True for integer sequences that likely represent bytes."""
    return isinstance(data, (list, tuple)) and bool(data) and isinstance(data[0], int)


def _is_raw_image_message(message: Any) -> bool:
    """Check whether a message payload exposes raw image metadata fields."""
    return (
        _read_field(message, "height") is not None
        and _read_field(message, "width") is not None
        and _read_field(message, "encoding") is not None
    )


def _is_compressed_image_message(message: Any) -> bool:
    """Check whether a message payload exposes compressed image bytes."""
    return _read_field(message, "data") is not None


class SkipMCAPMessage(Exception):
    """Raised when a message carries no usable payload and must be skipped.

    Unlike :class:`~neuracore.importer.core.exceptions.ImportError`, this is not
    a failure: inter-frame video streams legitimately produce no image until the
    first keyframe arrives, and those messages are dropped rather than aborting
    the episode.
    """


# ``CompressedImage.format`` values that denote an H.264 elementary stream
# rather than a still-image container PIL can open.
_H264_FORMAT_HINTS = ("h264", "h.264", "avc")

# NAL unit type of a Sequence Parameter Set; decoding cannot start before one.
_H264_NAL_TYPE_SPS = 7


def is_h264_image_format(format_value: Any) -> bool:
    """Return True when a ``CompressedImage.format`` names an H.264 stream."""
    if not format_value:
        return False
    text = str(format_value).lower()
    return any(hint in text for hint in _H264_FORMAT_HINTS)


def _h264_nal_unit_type(payload: bytes) -> int | None:
    """Return the NAL unit type of the first NAL unit in an Annex-B payload."""
    if payload.startswith(b"\x00\x00\x00\x01"):
        offset = 4
    elif payload.startswith(b"\x00\x00\x01"):
        offset = 3
    else:
        offset = 0
    if len(payload) <= offset:
        return None
    return payload[offset] & 0x1F


class H264StreamDecoder:
    """Stateful per-topic H.264 decoder over PyAV.

    One instance decodes exactly one topic's stream in log order. Packets
    arriving before the first Sequence Parameter Set are dropped, because an
    inter-frame stream cannot be decoded from the middle.
    """

    def __init__(self) -> None:
        """Create a fresh H.264 decoding context."""
        if not HAS_AV or av is None:
            raise ImportError(
                "Decoding H.264 image topics requires PyAV. "
                "Install with `pip install neuracore[import]`."
            )
        self._codec_context = av.CodecContext.create("h264", "r")
        self._started = False
        self.skipped_before_keyframe = 0

    def decode(self, payload: bytes) -> np.ndarray | None:
        """Decode one packet.

        Args:
            payload: Annex-B H.264 bytes for a single message.

        Returns:
            An ``(H, W, 3)`` uint8 RGB frame, or None when this packet yields no
            frame (before the first keyframe, or on a recoverable decode error).
        """
        if not payload:
            return None
        if not self._started:
            if _h264_nal_unit_type(payload) != _H264_NAL_TYPE_SPS:
                self.skipped_before_keyframe += 1
                return None
            self._started = True
        try:
            frames = self._codec_context.decode(av.packet.Packet(payload))
        except Exception:  # noqa: BLE001 - a damaged packet must not kill the run
            return None
        if not frames:
            return None
        return frames[0].to_ndarray(format="rgb24")


@dataclass(slots=True)
class MCAPImageContext:
    """Per-message state needed to decode video image topics.

    ``decoders`` is owned by the importer and lives for one episode so that
    inter-frame streams stay continuous. ``frame_cache`` is scoped to a single
    message so that several mapping entries on one topic share one decode -- a
    second ``decode()`` of the same packet would corrupt the stream state.
    """

    topic: str
    decoders: dict[str, H264StreamDecoder]
    frame_cache: dict[str, np.ndarray | None] = field(default_factory=dict)

    def decode_h264(self, payload: bytes) -> np.ndarray | None:
        """Decode (or return the cached decode of) this message's frame."""
        if self.topic in self.frame_cache:
            return self.frame_cache[self.topic]
        decoder = self.decoders.get(self.topic)
        if decoder is None:
            decoder = H264StreamDecoder()
            self.decoders[self.topic] = decoder
        frame = decoder.decode(payload)
        self.frame_cache[self.topic] = frame
        return frame


def _decode_raw_image(
    data_type: DataType,
    data: Any,
    message: Any,
    *,
    logger: logging.Logger,
) -> np.ndarray:
    """Decode ``sensor_msgs/Image``-style payloads."""
    height = _read_field(message, "height")
    width = _read_field(message, "width")
    encoding = _read_field(message, "encoding")
    step = _read_field(message, "step")
    is_bigendian = _read_field(message, "is_bigendian")

    if height is None or width is None or encoding is None:
        raise ImportError(
            "Raw image decoding requires height, width, and encoding fields."
        )

    encoding_name = str(encoding).lower().split(";", maxsplit=1)[0].strip()
    enc_map = {
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16uc1": (np.uint16, 1),
        "32fc1": (np.float32, 1),
        "64fc1": (np.float64, 1),
    }
    if encoding_name not in enc_map:
        raise ImportError(f"Unsupported image encoding '{encoding_name}'.")

    dtype, channels = enc_map[encoding_name]
    buffer = _read_bytes(data)

    bytes_per_pixel = np.dtype(dtype).itemsize * channels
    row_step = int(step) if step else int(width) * bytes_per_pixel
    row_elements = row_step // np.dtype(dtype).itemsize
    expected_len = row_step * int(height)
    actual_len = len(buffer)
    if actual_len != expected_len:
        relation = "too small" if actual_len < expected_len else "too large"
        raise ImportError(
            "Image buffer size mismatch "
            f"({relation}: expected {expected_len} bytes, got {actual_len})."
        )

    array = np.frombuffer(buffer[:expected_len], dtype=dtype).reshape(
        int(height), row_elements
    )
    if channels == 1:
        array = array[:, : int(width)]
    else:
        array = array[:, : int(width) * channels].reshape(
            int(height),
            int(width),
            channels,
        )

    if is_bigendian and np.dtype(dtype).itemsize > 1:
        array = array.byteswap()

    return _drop_alpha_channel(data_type, array, logger=logger)


def _decode_compressed_image(
    data_type: DataType,
    message: Any,
    *,
    logger: logging.Logger,
    image_context: MCAPImageContext | None = None,
) -> np.ndarray:
    """Decode ``sensor_msgs/CompressedImage``-style payloads.

    Still-image containers (PNG/JPEG/...) go through PIL. Payloads whose
    ``format`` field names an H.264 stream are handed to the per-topic video
    decoder in ``image_context`` instead.
    """
    raw = _read_field(message, "data")
    if raw is None:
        raise ImportError("Compressed image decoding requires data field.")
    if is_h264_image_format(_read_field(message, "format")):
        return _decode_h264_image(
            data_type,
            raw,
            logger=logger,
            image_context=image_context,
        )
    return __decode_compressed_image_bytes(data_type, raw, logger=logger)


def _decode_h264_image(
    data_type: DataType,
    raw: Any,
    *,
    logger: logging.Logger,
    image_context: MCAPImageContext | None,
) -> np.ndarray:
    """Decode one frame of an H.264 image topic.

    Raises:
        SkipMCAPMessage: When this packet yields no frame, which is expected for
            every packet before the stream's first keyframe.
        ImportError: When H.264 decoding is not usable for this data type or no
            decoder registry was supplied.
    """
    if data_type != DataType.RGB_IMAGES:
        raise ImportError(
            f"H.264 payloads are only supported for {DataType.RGB_IMAGES.value}; "
            f"got {data_type.value}. Hue-encoded H.264 depth is not supported."
        )
    if image_context is None:
        raise ImportError(
            "H.264 image topics require a per-episode video decoder registry. "
            "Pass 'video_decoders' to iter_mcap_source_events()."
        )
    frame = image_context.decode_h264(_read_bytes(raw))
    if frame is None:
        raise SkipMCAPMessage(
            f"No decodable H.264 frame for topic '{image_context.topic}' yet."
        )
    return _drop_alpha_channel(data_type, frame, logger=logger)


def read_image_data(
    data_type: DataType,
    data: Any,
    message: Any,
    *,
    logger: logging.Logger,
    image_context: MCAPImageContext | None = None,
) -> Any:
    """Read image payloads as arrays; keep non-image values untouched."""
    if data_type not in {DataType.RGB_IMAGES, DataType.DEPTH_IMAGES}:
        return data

    if isinstance(data, np.ndarray):
        return _drop_alpha_channel(data_type, data, logger=logger)

    if _is_raw_image_message(message):
        return _decode_raw_image(data_type, data, message, logger=logger)

    if _is_compressed_image_message(message):
        return _decode_compressed_image(
            data_type,
            message,
            logger=logger,
            image_context=image_context,
        )

    if isinstance(data, (list, tuple)) and data and not _is_byte_list(data):
        array = np.array(data)
        if array.ndim >= 2:
            return _drop_alpha_channel(data_type, array, logger=logger)

    if isinstance(data, (bytes, bytearray, memoryview, str)) or _is_byte_list(data):
        try:
            return __decode_compressed_image_bytes(data_type, data, logger=logger)
        except ImportError:
            pass

    raise ImportError(
        "Image mapping resolved to unsupported payload type "
        f"{type(data).__name__}. Configure mapping to point to image bytes/data."
    )


def __decode_compressed_image_bytes(
    data_type: DataType,
    data: Any,
    *,
    logger: logging.Logger,
    has_pil: bool = HAS_PIL,
    image_module: Any = Image,
) -> np.ndarray:
    """Decode compressed image bytes into a numpy array."""
    if not has_pil or image_module is None:
        raise ImportError(
            "Compressed image decoding requires pillow. "
            "Install with `pip install neuracore[import]`."
        )

    buffer = _read_bytes(data)
    try:
        with image_module.open(io.BytesIO(buffer)) as image:
            if data_type == DataType.RGB_IMAGES and image.mode != "RGB":
                image = image.convert("RGB")
            array = np.array(image)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"Failed decoding compressed image: {exc}") from exc

    return _drop_alpha_channel(data_type, array, logger=logger)


def _read_bytes(data: Any) -> bytes:
    """Read bytes from bytes-like values, base64 strings, or integer lists."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        return _decode_base64_bytes(data)
    if _is_byte_list(data):
        return bytes(data)
    raise ImportError("Image payload is not a byte buffer.")


def _drop_alpha_channel(
    data_type: DataType,
    array: np.ndarray,
    *,
    logger: logging.Logger,
) -> np.ndarray:
    """Normalize image arrays into Neuracore-friendly shape and dtype."""
    if data_type == DataType.RGB_IMAGES and array.ndim == 3 and array.shape[2] == 4:
        logger.warning("Dropping alpha channel for RGB image import.")
        array = array[:, :, :3]

    if data_type == DataType.DEPTH_IMAGES and array.dtype not in (
        np.float16,
        np.float32,
        np.float64,
    ):
        array = array.astype(np.float32, copy=False)

    return array


def _decode_base64_bytes(value: str) -> bytes:
    """Decode a base64 string into bytes."""
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ImportError(
            "Image payload is a string but not valid base64-encoded bytes."
        ) from exc


def convert_decoded_mcap_data(decoded_data: Any) -> Any:
    """Convert decoded MCAP data into values the importer can read.

    Protobuf objects are kept as-is to preserve ``bytes`` fields used by image
    extraction. Non-protobuf decoded data is converted recursively.
    """
    if (
        HAS_PROTOBUF_RUNTIME
        and ProtobufMessage is not None
        and isinstance(decoded_data, ProtobufMessage)
    ):
        return decoded_data
    return to_python_types(decoded_data)


def _apply_item_indexing(source_data: Any, item: MappingItem) -> Any:
    """Apply a mapping item's ``index``/``index_range`` to an extracted value."""
    if item.index is not None:
        return source_data[item.index]
    if item.index_range is not None:
        return source_data[item.index_range.start : item.index_range.end]
    return source_data


def iter_mcap_source_events(
    topic: str,
    decoded_data: Any,
    *,
    topic_map: TopicMap,
    logger: logging.Logger,
    timestamp: float,
    video_decoders: dict[str, H264StreamDecoder] | None = None,
) -> Iterator[MCAPSourceEvent]:
    """Yield source events for each mapping config, ready for _log_data.

    Args:
        topic: MCAP channel topic this message arrived on.
        decoded_data: The decoded message.
        topic_map: Topic lookup built by :func:`build_topic_map`.
        logger: Logger for decode diagnostics.
        timestamp: Neuracore timestamp for this message.
        video_decoders: Per-episode registry of H.264 decoders keyed by topic,
            owned by the importer so that inter-frame streams stay continuous.
            Required only when a configured image topic carries H.264 payloads.
    """
    configs = topic_map.get(topic, [])
    if not configs:
        return

    image_context = (
        MCAPImageContext(topic=topic, decoders=video_decoders)
        if video_decoders is not None
        else None
    )

    for config in configs:
        if config.mapping_item is not None:
            base = decoded_data
            if config.item_base_path:
                base = resolve_path(base, config.item_base_path)

            try:
                source_data = read_image_data(
                    config.data_type,
                    base,
                    decoded_data,
                    logger=logger,
                    image_context=image_context,
                )
            except SkipMCAPMessage as skip:
                logger.debug(f"Skipping message on '{topic}': {skip}")
                continue
            if not _is_language_text(config.data_type, config.import_config):
                source_data = to_numpy(source_data)
                # An absolute source_name selects the topic and field path; any
                # index/index_range then applies to the resolved value. Note the
                # relative branch below instead treats source_name as exclusive
                # of index/index_range.
                source_data = _apply_item_indexing(source_data, config.mapping_item)

            yield MCAPSourceEvent(
                data_type=config.data_type,
                source_data=source_data,
                item=config.mapping_item,
                format=config.import_config.format,
                timestamp=timestamp,
                source_topic=topic,
            )
            continue

        base = resolve_path(decoded_data, config.source_path)
        for item in config.import_config.mapping:
            if item.source_name:
                source_data = resolve_path(base, item.source_name.split("."))
            elif item.index is not None:
                source_data = base[item.index]
            elif item.index_range is not None:
                source_data = base[item.index_range.start : item.index_range.end]
            else:
                source_data = base

            try:
                source_data = read_image_data(
                    config.data_type,
                    source_data,
                    decoded_data,
                    logger=logger,
                    image_context=image_context,
                )
            except SkipMCAPMessage as skip:
                logger.debug(f"Skipping message on '{topic}': {skip}")
                continue
            if not _is_language_text(config.data_type, config.import_config):
                source_data = to_numpy(source_data)

            yield MCAPSourceEvent(
                data_type=config.data_type,
                source_data=source_data,
                item=item,
                format=config.import_config.format,
                timestamp=timestamp,
                source_topic=topic,
            )


_PRIMITIVE_PYTHON_TYPES = frozenset(
    {bool, int, float, str, bytes, bytearray, type(None)}
)


def to_python_types(value: Any) -> Any:
    """Recursively convert message payload objects to plain Python values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        return value

    if (
        HAS_PROTOBUF_RUNTIME
        and ProtobufMessage is not None
        and isinstance(value, ProtobufMessage)
    ):
        if MessageToDict is None:
            return repr(value)
        return to_python_types(MessageToDict(value, preserving_proto_field_name=True))

    if isinstance(value, dict):
        if all(type(v) in _PRIMITIVE_PYTHON_TYPES for v in value.values()):
            return value
        return {str(key): to_python_types(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        lst = value if isinstance(value, list) else list(value)
        if lst and all(type(e) in _PRIMITIVE_PYTHON_TYPES for e in lst):
            return lst
        return [to_python_types(item) for item in lst]

    if hasattr(value, "__dict__"):
        attrs = {
            name: getattr(value, name)
            for name in vars(value)
            if not str(name).startswith("_")
        }
        if attrs:
            return {key: to_python_types(item) for key, item in attrs.items()}

    slots = getattr(type(value), "__slots__", None)
    if slots:
        names = [slots] if isinstance(slots, str) else list(slots)
        out: dict[str, Any] = {}
        for name in names:
            if not isinstance(name, str) or name.startswith("_"):
                continue
            try:
                out[name] = to_python_types(getattr(value, name))
            except Exception:  # noqa: BLE001
                continue
        if out:
            return out

    return repr(value)


_PROTOBUF_NUMERIC_FIELD_TYPES: frozenset[int] = frozenset()
if HAS_PROTOBUF_RUNTIME:
    from google.protobuf.descriptor import FieldDescriptor as _FieldDescriptor

    _PROTOBUF_NUMERIC_FIELD_TYPES = frozenset({
        _FieldDescriptor.TYPE_DOUBLE,
        _FieldDescriptor.TYPE_FLOAT,
        _FieldDescriptor.TYPE_INT32,
        _FieldDescriptor.TYPE_INT64,
        _FieldDescriptor.TYPE_UINT32,
        _FieldDescriptor.TYPE_UINT64,
        _FieldDescriptor.TYPE_SINT32,
        _FieldDescriptor.TYPE_SINT64,
        _FieldDescriptor.TYPE_FIXED32,
        _FieldDescriptor.TYPE_FIXED64,
        _FieldDescriptor.TYPE_SFIXED32,
        _FieldDescriptor.TYPE_SFIXED64,
        _FieldDescriptor.TYPE_BOOL,
    })


def _is_repeated_field(descriptor: Any) -> bool:
    """Return True when a protobuf field descriptor describes a repeated field.

    ``FieldDescriptor.is_repeated`` is a bool property on protobuf >= 5.27 and
    absent on older releases, where the (now deprecated) ``label`` is the only
    option. Cover both so this works across the protobuf versions pulled in by
    the ``[import]`` extra's transitive dependencies.
    """
    repeated = getattr(descriptor, "is_repeated", None)
    if repeated is None:
        return bool(descriptor.label == descriptor.LABEL_REPEATED)
    if callable(repeated):
        return bool(repeated())
    return bool(repeated)


def flatten_numeric_protobuf(message: Any) -> list[float] | None:
    """Flatten a purely numeric protobuf message into a list of floats.

    Singular numeric fields are appended in declaration order and singular
    sub-messages are recursed into, so ``foxglove.Pose`` yields
    ``[x, y, z, qx, qy, qz, qw]`` -- the layout Neuracore's
    ``POSITION_ORIENTATION`` / ``QUATERNION`` / ``XYZW`` pose format expects --
    and ``foxglove.Vector3`` yields ``[x, y, z]``.

    Args:
        message: A protobuf message instance.

    Returns:
        The flattened values, or None when the message contains anything that is
        not a singular numeric field or numeric sub-message (a repeated field, or
        a string/bytes/enum field). Returning None lets callers leave the
        message untouched rather than inventing a numeric interpretation.
    """
    if not HAS_PROTOBUF_RUNTIME or ProtobufMessage is None:
        return None
    if not isinstance(message, ProtobufMessage):
        return None

    values: list[float] = []
    for descriptor in message.DESCRIPTOR.fields:
        if _is_repeated_field(descriptor):
            return None
        if descriptor.type == descriptor.TYPE_MESSAGE:
            nested = flatten_numeric_protobuf(getattr(message, descriptor.name))
            if nested is None:
                return None
            values.extend(nested)
        elif descriptor.type in _PROTOBUF_NUMERIC_FIELD_TYPES:
            values.append(float(getattr(message, descriptor.name)))
        else:
            return None
    return values or None


def to_numpy(data: Any) -> Any:
    """Convert numeric Python values to numpy values for transform compatibility."""
    if hasattr(data, "numpy"):
        return data.numpy()
    if isinstance(data, np.ndarray):
        return data
    flattened = flatten_numeric_protobuf(data)
    if flattened is not None:
        return np.asarray(flattened, dtype=np.float64)
    if isinstance(data, (list, tuple)):
        return np.array(data)
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return np.float64(data)
    return data


def clip_depth(
    data: Any,
    logger: logging.Logger | None = None,
) -> Any:
    """Clip depth arrays to the backend-accepted meter range."""
    if not isinstance(data, np.ndarray):
        return data
    float32 = data.astype(np.float32, copy=False)
    needs_clip = (
        np.any(np.isnan(float32))
        or np.any(np.isinf(float32))
        or float32.size > 0
        and (float(float32.min()) < 0.0 or float(float32.max()) > MAX_DEPTH)
    )
    if needs_clip and logger is not None:
        logger.warning(
            f"Depth values outside valid range [0, {MAX_DEPTH:.1f} m] — clipping."
        )
    clipped = np.nan_to_num(float32, nan=0.0, posinf=MAX_DEPTH, neginf=0.0)
    return np.clip(clipped, 0.0, MAX_DEPTH).astype(np.float16, copy=False)


def _is_language_text(data_type: DataType, import_config: Any) -> bool:
    """Return True when the mapping describes plain language text."""
    return bool(
        data_type == DataType.LANGUAGE
        and import_config.format.language_type == LanguageConfig.STRING
    )
