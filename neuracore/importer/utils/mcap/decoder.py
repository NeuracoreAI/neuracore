"""MCAP decoding and transform helpers.

This module intentionally centralizes MCAP decode functionality in one place:
- construct ``decoder_factories`` for ``mcap.reader.make_reader(...)``
- warn about unsupported channel encodings
- coerce ROS-style image payloads into numpy arrays
- map decoded topic messages into ``CachedMessage`` records

The importer now relies on MCAP's native ``iter_decoded_messages()`` API rather
than a custom registry layer.
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
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from mcap.decoder import DecoderFactory
from mcap.well_known import MessageEncoding
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig

from neuracore.core.utils.depth_utils import MAX_DEPTH
from neuracore.importer.core.exceptions import ImportError

from .cache import CachedMessage
from .paths import resolve_path
from .topics import TopicMapper

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
            name = (distribution.metadata.get("Name") or "").strip().lower()
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
        logger.info("Discovered %s MCAP decoder plugin class(es).", len(classes))

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
                    "Failed to instantiate discovered MCAP decoder factory '%s': %s",
                    factory_cls,
                    exc,
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
        logger.debug(
            "Configured MCAP decoder factories: %s",
            [f"{f.__class__.__module__}.{f.__class__.__qualname__}" for f in factories],
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
                "MCAP topic '%s' uses CBOR but cbor2 is unavailable.",
                channel.topic,
            )

        if _channel_has_decoder_support(encoding, schema, decoder_factories):
            continue

        logger.warning(
            "No structured decoder available for topic '%s' (encoding=%s). "
            "Using raw-byte fallback.",
            channel.topic,
            encoding or "<empty>",
        )


class ImageDecoder:
    """Decode ROS-style raw and compressed image payloads into numpy arrays."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        has_pil: bool = HAS_PIL,
        image_module: Any = Image,
    ) -> None:
        """Configure image decoding helpers and optional PIL dependency handles."""
        self._logger = logger
        self._has_pil = has_pil
        self._image_module = image_module

    @staticmethod
    def get_field(data: Any, name: str) -> Any:
        """Read a field from mapping-like or object-like decoded payloads."""
        if isinstance(data, dict):
            return data.get(name)
        if hasattr(data, name):
            return getattr(data, name)
        return None

    @staticmethod
    def looks_like_bytes(data: Any) -> bool:
        """Return True for integer sequences that likely represent bytes."""
        return (
            isinstance(data, (list, tuple)) and bool(data) and isinstance(data[0], int)
        )

    def has_raw_image_fields(self, message: Any) -> bool:
        """Check whether a decoded message exposes raw image metadata fields."""
        return (
            self.get_field(message, "height") is not None
            and self.get_field(message, "width") is not None
            and self.get_field(message, "encoding") is not None
        )

    def has_compressed_image_fields(self, message: Any) -> bool:
        """Check whether a decoded message exposes compressed image bytes."""
        return self.get_field(message, "data") is not None

    def decode_raw_image(
        self,
        data_type: DataType,
        data: Any,
        message: Any,
    ) -> np.ndarray:
        """Decode ``sensor_msgs/Image``-style payloads."""
        height = self.get_field(message, "height")
        width = self.get_field(message, "width")
        encoding = self.get_field(message, "encoding")
        step = self.get_field(message, "step")
        is_bigendian = self.get_field(message, "is_bigendian")

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
        buffer = self._buffer_from_any(data)

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
            array = array.byteswap().newbyteorder()

        return self._normalize_image_shape(data_type, array)

    def decode_compressed_image(self, data_type: DataType, message: Any) -> np.ndarray:
        """Decode ``sensor_msgs/CompressedImage`` payloads."""
        raw = self.get_field(message, "data")
        if raw is None:
            raise ImportError("Compressed image decoding requires data field.")
        return self._decode_compressed_bytes(data_type, raw)

    def coerce_message_data(self, data_type: DataType, data: Any, message: Any) -> Any:
        """Convert image-like values into arrays; keep other data untouched."""
        if data_type not in {DataType.RGB_IMAGES, DataType.DEPTH_IMAGES}:
            return data

        if isinstance(data, np.ndarray):
            return self._normalize_image_shape(data_type, data)

        if self.has_raw_image_fields(message):
            return self.decode_raw_image(data_type, data, message)

        if self.has_compressed_image_fields(message):
            return self.decode_compressed_image(data_type, message)

        if isinstance(data, (list, tuple)) and data and not self.looks_like_bytes(data):
            array = np.array(data)
            if array.ndim >= 2:
                return self._normalize_image_shape(data_type, array)

        if isinstance(
            data, (bytes, bytearray, memoryview, str)
        ) or self.looks_like_bytes(data):
            try:
                return self._decode_compressed_bytes(data_type, data)
            except ImportError:
                pass

        raise ImportError(
            "Image mapping resolved to unsupported payload type "
            f"{type(data).__name__}. Configure mapping to point to image bytes/data."
        )

    def _decode_compressed_bytes(self, data_type: DataType, data: Any) -> np.ndarray:
        if not self._has_pil or self._image_module is None:
            raise ImportError(
                "Compressed image decoding requires pillow. "
                "Install with `pip install neuracore[import]`."
            )

        buffer = self._buffer_from_any(data)
        try:
            with self._image_module.open(io.BytesIO(buffer)) as image:
                if data_type == DataType.RGB_IMAGES and image.mode != "RGB":
                    image = image.convert("RGB")
                array = np.array(image)
        except Exception as exc:  # noqa: BLE001
            raise ImportError(f"Failed decoding compressed image: {exc}") from exc

        return self._normalize_image_shape(data_type, array)

    def _buffer_from_any(self, data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        if isinstance(data, str):
            return self._decode_base64_bytes(data)
        if self.looks_like_bytes(data):
            return bytes(data)
        raise ImportError("Image payload is not a byte buffer.")

    def _normalize_image_shape(
        self, data_type: DataType, array: np.ndarray
    ) -> np.ndarray:
        if data_type == DataType.RGB_IMAGES and array.ndim == 3 and array.shape[2] == 4:
            self._logger.warning("Dropping alpha channel for RGB image import.")
            array = array[:, :, :3]

        if data_type == DataType.DEPTH_IMAGES and array.dtype not in (
            np.float16,
            np.float32,
            np.float64,
        ):
            array = array.astype(np.float32, copy=False)

        return array

    @staticmethod
    def _decode_base64_bytes(value: str) -> bytes:
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ImportError(
                "Image payload is a string but not valid base64-encoded bytes."
            ) from exc


class MCAPMessageDecoder:
    """Map decoded MCAP topic payloads to transformed Neuracore cache events."""

    def __init__(
        self,
        topic_mapper: TopicMapper,
        prepare_log_data: Callable[..., Any],
        image_decoder: ImageDecoder,
    ) -> None:
        """Bind topic mapping and transformation helpers for message conversion."""
        self._topic_mapper = topic_mapper
        self._prepare_log_data = prepare_log_data
        self._image_decoder = image_decoder

    def normalize_decoded_message(self, decoded: Any) -> Any:
        """Normalize MCAP decoded payloads into plain Python structures.

        Protobuf objects are kept as-is to preserve ``bytes`` fields used by image
        extraction. Non-protobuf payloads are converted recursively.
        """
        if (
            HAS_PROTOBUF_RUNTIME
            and ProtobufMessage is not None
            and isinstance(decoded, ProtobufMessage)
        ):
            return decoded
        return self._convert_to_plain_types(decoded)

    def transform_message(
        self,
        topic: str,
        decoded: Any,
        *,
        timestamp: float,
        log_time_ns: int,
    ) -> list[CachedMessage]:
        """Transform one decoded topic message into one or more cache events."""
        return list(
            self.iter_transformed_messages(
                topic,
                decoded,
                timestamp=timestamp,
                log_time_ns=log_time_ns,
            )
        )

    def iter_transformed_messages(
        self,
        topic: str,
        decoded: Any,
        *,
        timestamp: float,
        log_time_ns: int,
    ) -> Iterator[CachedMessage]:
        """Yield transformed events one-by-one to avoid per-message list allocation."""
        configs = self._topic_mapper.get_configs_for_topic(topic)
        if not configs:
            return

        for config in configs:
            if config.mapping_item is not None:
                base = decoded
                if config.item_base_path:
                    base = resolve_path(base, config.item_base_path)

                source_data = self._image_decoder.coerce_message_data(
                    config.data_type,
                    base,
                    decoded,
                )
                if not self._is_language_string(config.data_type, config.import_config):
                    source_data = self._to_numpy(source_data)

                transformed_data = self._prepare_log_data(
                    data_type=config.data_type,
                    source_data=source_data,
                    item=config.mapping_item,
                    format=config.import_config.format,
                )
                if config.data_type == DataType.DEPTH_IMAGES:
                    transformed_data = self._sanitize_depth_image(transformed_data)
                yield CachedMessage(
                    data_type=config.data_type.value,
                    name=config.mapping_item.name,
                    timestamp=timestamp,
                    log_time_ns=log_time_ns,
                    transformed_data=transformed_data,
                    source_topic=topic,
                )
                continue

            base = resolve_path(decoded, config.source_path)
            for item in config.import_config.mapping:
                if item.source_name:
                    source_data = resolve_path(base, item.source_name.split("."))
                elif item.index is not None:
                    source_data = base[item.index]
                elif item.index_range is not None:
                    source_data = base[item.index_range.start : item.index_range.end]
                else:
                    source_data = base

                source_data = self._image_decoder.coerce_message_data(
                    config.data_type,
                    source_data,
                    decoded,
                )
                if not self._is_language_string(config.data_type, config.import_config):
                    source_data = self._to_numpy(source_data)

                transformed_data = self._prepare_log_data(
                    data_type=config.data_type,
                    source_data=source_data,
                    item=item,
                    format=config.import_config.format,
                )
                if config.data_type == DataType.DEPTH_IMAGES:
                    transformed_data = self._sanitize_depth_image(transformed_data)
                yield CachedMessage(
                    data_type=config.data_type.value,
                    name=item.name,
                    timestamp=timestamp,
                    log_time_ns=log_time_ns,
                    transformed_data=transformed_data,
                    source_topic=topic,
                )

    def _convert_to_plain_types(self, value: Any) -> Any:
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
            return self._convert_to_plain_types(
                MessageToDict(value, preserving_proto_field_name=True)
            )

        if isinstance(value, dict):
            return {
                str(key): self._convert_to_plain_types(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._convert_to_plain_types(item) for item in value]

        if hasattr(value, "__dict__"):
            attrs = {
                name: getattr(value, name)
                for name in vars(value)
                if not str(name).startswith("_")
            }
            if attrs:
                return {
                    key: self._convert_to_plain_types(item)
                    for key, item in attrs.items()
                }

        slots = getattr(type(value), "__slots__", None)
        if slots:
            names = [slots] if isinstance(slots, str) else list(slots)
            out: dict[str, Any] = {}
            for name in names:
                if not isinstance(name, str) or name.startswith("_"):
                    continue
                try:
                    out[name] = self._convert_to_plain_types(getattr(value, name))
                except Exception:  # noqa: BLE001
                    continue
            if out:
                return out

        return repr(value)

    def _to_numpy(self, data: Any) -> Any:
        if hasattr(data, "numpy"):
            return data.numpy()
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            return np.array(data)
        return data

    @staticmethod
    def _sanitize_depth_image(data: Any) -> Any:
        """Clip depth arrays to the backend-accepted meter range."""
        if not isinstance(data, np.ndarray):
            return data
        clipped = np.nan_to_num(
            data.astype(np.float32, copy=False),
            nan=0.0,
            posinf=MAX_DEPTH,
            neginf=0.0,
        )
        return np.clip(clipped, 0.0, MAX_DEPTH).astype(np.float16, copy=False)

    @staticmethod
    def _is_language_string(data_type: DataType, import_config: Any) -> bool:
        return bool(
            data_type == DataType.LANGUAGE
            and import_config.format.language_type == LanguageConfig.STRING
        )


__all__ = [
    "CborDecoderFactory",
    "ImageDecoder",
    "JSONDecoderFactory",
    "MCAPMessageDecoder",
    "RawPassthroughDecoderFactory",
    "TextDecoderFactory",
    "discover_decoder_factories",
    "list_decoder_factories",
    "validate_channel_decoder_support",
]
