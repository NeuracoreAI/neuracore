"""MCAP reader utilities for the Neuracore importer.

Thin adapter over Foxglove's ``mcap`` package: decodes messages and normalises
them into the shape expected by Neuracore's mapping pipeline.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import logging
import pkgutil
import time
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass
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
    timestamp_seconds: Resolved timestamp in seconds, preferring log time.
    data: Decoded message data returned by the MCAP decoder factory. This is
        the source object Neuracore mappings read from.
    """

    topic: str
    log_time_ns: int
    publish_time_ns: int
    timestamp_seconds: float
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


def resolve_timestamp_seconds(
    *,
    log_time_ns: int,
    publish_time_ns: int,
) -> float:
    """Resolve message timestamp from log/publish time nanoseconds."""
    if log_time_ns > 0:
        return log_time_ns / 1e9
    if publish_time_ns > 0:
        return publish_time_ns / 1e9
    return time.time()


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
            timestamp_seconds=resolve_timestamp_seconds(
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
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.message import Message as ProtobufMessage

    HAS_PROTOBUF_RUNTIME = True
except Exception:  # noqa: BLE001
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
        factories.append(ProtobufDecoderFactory())
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
) -> np.ndarray:
    """Decode ``sensor_msgs/CompressedImage`` payloads."""
    raw = _read_field(message, "data")
    if raw is None:
        raise ImportError("Compressed image decoding requires data field.")
    return __decode_compressed_image_bytes(data_type, raw, logger=logger)


def read_image_data(
    data_type: DataType,
    data: Any,
    message: Any,
    *,
    logger: logging.Logger,
) -> Any:
    """Read image payloads as arrays; keep non-image values untouched."""
    if data_type not in {DataType.RGB_IMAGES, DataType.DEPTH_IMAGES}:
        return data

    if isinstance(data, np.ndarray):
        return _drop_alpha_channel(data_type, data, logger=logger)

    if _is_raw_image_message(message):
        return _decode_raw_image(data_type, data, message, logger=logger)

    if _is_compressed_image_message(message):
        return _decode_compressed_image(data_type, message, logger=logger)

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


def iter_mcap_source_events(
    topic: str,
    decoded_data: Any,
    *,
    topic_map: TopicMap,
    logger: logging.Logger,
    timestamp: float,
) -> Iterator[MCAPSourceEvent]:
    """Yield source events for each mapping config, ready for _log_data."""
    configs = topic_map.get(topic, [])
    if not configs:
        return

    for config in configs:
        if config.mapping_item is not None:
            base = decoded_data
            if config.item_base_path:
                base = resolve_path(base, config.item_base_path)

            source_data = read_image_data(
                config.data_type,
                base,
                decoded_data,
                logger=logger,
            )
            if not _is_language_text(config.data_type, config.import_config):
                source_data = to_numpy(source_data)

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

            source_data = read_image_data(
                config.data_type,
                source_data,
                decoded_data,
                logger=logger,
            )
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


def to_numpy(data: Any) -> Any:
    """Convert numeric Python values to numpy values for transform compatibility."""
    if hasattr(data, "numpy"):
        return data.numpy()
    if isinstance(data, np.ndarray):
        return data
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
