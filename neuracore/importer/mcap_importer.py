"""Importer for MCAP datasets."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import io
import json
import logging
import multiprocessing as mp
import os
import pkgutil
import struct
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from copy import copy
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from mcap.decoder import DecoderFactory
from mcap.reader import make_reader
from mcap.well_known import MessageEncoding
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.importer.mcap import StagedRecord, TopicImportConfig
from neuracore_types.nc_data import DatasetImportConfig
from neuracore_types.nc_data.nc_data import MappingItem
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import neuracore as nc
from neuracore.core.robot import JointInfo
from neuracore.importer.core.base import (
    ImportItem,
    NeuracoreDatasetImporter,
    WorkerError,
    get_shared_console,
)
from neuracore.importer.core.exceptions import ImportError, UploaderError

MODULE_LOGGER = logging.getLogger(__name__)
RECORDING_RETRY_SLEEP_SECONDS = 0.2
RECORDING_POLL_SLEEP_SECONDS = 0.05
MAX_TO_PLAIN_DEPTH = 64
MAX_TO_PLAIN_REPR_CHARS = 2048


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    """Parse float env vars defensively with a lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        MODULE_LOGGER.warning("Invalid %s=%r; using default %.3f.", name, raw, default)
        return default
    if parsed < minimum:
        MODULE_LOGGER.warning(
            "Invalid %s=%r (< %.3f); using default %.3f.",
            name,
            raw,
            minimum,
            default,
        )
        return default
    return parsed


def env_int(name: str, default: int, minimum: int = 0) -> int:
    """Parse int env vars defensively with a lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        MODULE_LOGGER.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return default
    if parsed < minimum:
        MODULE_LOGGER.warning(
            "Invalid %s=%r (< %s); using default %s.",
            name,
            raw,
            minimum,
            default,
        )
        return default
    return parsed


def env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean env vars with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


try:  # Optional decoder dependencies.
    from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory

    HAS_PROTOBUF = True
except Exception:  # noqa: BLE001
    ProtobufDecoderFactory = None
    HAS_PROTOBUF = False

try:
    from mcap_ros1.decoder import DecoderFactory as Ros1DecoderFactory

    HAS_ROS1 = True
except Exception:  # noqa: BLE001
    Ros1DecoderFactory = None
    HAS_ROS1 = False

try:
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    HAS_ROS2 = True
except Exception:  # noqa: BLE001
    Ros2DecoderFactory = None
    HAS_ROS2 = False

try:
    from PIL import Image

    HAS_PIL = True
except Exception:  # noqa: BLE001
    Image = None
    HAS_PIL = False

try:
    import cbor2

    HAS_CBOR = True
except Exception:  # noqa: BLE001
    cbor2 = None
    HAS_CBOR = False


class JsonDecoderFactory(DecoderFactory):
    """Decode JSON-encoded MCAP messages into Python objects."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a JSON decoder when the encoding matches."""
        if (message_encoding or "").lower() != MessageEncoding.JSON.lower():
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


class CborDecoderFactory(DecoderFactory):
    """Decode CBOR-encoded MCAP messages into Python objects."""

    def decoder_for(self, message_encoding: str, schema: Any | None) -> Any | None:
        """Return a CBOR decoder when the encoding matches."""
        if (message_encoding or "").lower() != MessageEncoding.CBOR.lower():
            return None
        if not HAS_CBOR or cbor2 is None:
            return None

        def _decode(data: bytes) -> Any:
            payload = (
                data.tobytes()
                if isinstance(data, memoryview)
                else bytes(data) if isinstance(data, (bytes, bytearray)) else data
            )
            return cbor2.loads(payload)

        return _decode


FuncType = TypeVar("FuncType", bound=Callable[..., Any])


def retry_with_backoff(
    max_attempts: int,
    backoff_seconds: float,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, int, Exception], None] | None = None,
) -> Callable[[FuncType], FuncType]:
    """Retry a callable with fixed backoff and optional retry hook."""

    def decorator(func: FuncType) -> FuncType:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_error = exc
                    if attempt >= max_attempts:
                        break
                    if on_retry is not None:
                        on_retry(attempt, max_attempts, exc)
                    time.sleep(backoff_seconds)
            if last_error is None:
                raise RuntimeError("retry_with_backoff exhausted without error.")
            raise last_error

        return wrapper  # type: ignore[return-value]

    return decorator


class ImageDecoder:
    """Decode raw/compressed ROS-style image payloads."""

    def __init__(
        self, logger: logging.Logger, has_pil: bool, image_module: Any
    ) -> None:
        """Initialize image decoding dependencies and logger handles."""
        self._logger = logger
        self._has_pil = has_pil
        self._image_module = image_module

    @staticmethod
    def get_field(data: Any, name: str) -> Any:
        """Fetch a field from dicts or objects."""
        if isinstance(data, dict):
            return data.get(name)
        if hasattr(data, name):
            return getattr(data, name)
        return None

    @staticmethod
    def looks_like_bytes(data: Any) -> bool:
        """Detect list-like byte buffers."""
        return (
            isinstance(data, (list, tuple)) and bool(data) and isinstance(data[0], int)
        )

    def has_raw_image_fields(self, message: Any) -> bool:
        """Check for fields needed to decode raw images."""
        return (
            self.get_field(message, "height") is not None
            and self.get_field(message, "width") is not None
            and self.get_field(message, "encoding") is not None
        )

    def has_compressed_image_fields(self, message: Any) -> bool:
        """Check for fields needed to decode compressed images."""
        return self.get_field(message, "data") is not None

    def decode_raw_image(
        self, data_type: DataType, data: Any, message: Any
    ) -> np.ndarray:
        """Decode RawImage-like messages into numpy arrays."""
        height = self.get_field(message, "height")
        width = self.get_field(message, "width")
        encoding = self.get_field(message, "encoding")
        step = self.get_field(message, "step")
        is_bigendian = self.get_field(message, "is_bigendian")

        if height is None or width is None or encoding is None:
            raise ImportError(
                "Raw image decoding requires height, width, and encoding fields."
            )

        encoding = str(encoding).lower()
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
        if encoding not in enc_map:
            raise ImportError(f"Unsupported image encoding '{encoding}'.")

        dtype, channels = enc_map[encoding]
        if isinstance(data, (bytes, bytearray, memoryview)):
            buffer = bytes(data)
        elif self.looks_like_bytes(data):
            buffer = bytes(data)
        else:
            raise ImportError("Image data is not a byte buffer.")

        bytes_per_pixel = np.dtype(dtype).itemsize * channels
        row_step = int(step) if step else int(width) * bytes_per_pixel
        row_elems = row_step // np.dtype(dtype).itemsize
        expected_len = row_step * int(height)

        actual_len = len(buffer)
        if actual_len != expected_len:
            relation = "too small" if actual_len < expected_len else "too large"
            raise ImportError(
                "Image buffer size mismatch "
                f"({relation}: expected {expected_len} bytes, got {actual_len})."
            )

        array = np.frombuffer(buffer[:expected_len], dtype=dtype).reshape(
            int(height), row_elems
        )
        if channels == 1:
            array = array[:, : int(width)]
        else:
            array = array[:, : int(width) * channels].reshape(
                int(height), int(width), channels
            )

        if is_bigendian and np.dtype(dtype).itemsize > 1:
            array = array.byteswap().newbyteorder()

        if data_type == DataType.RGB_IMAGES and channels == 4:
            self._logger.warning("Dropping alpha channel for RGB image import.")
            array = array[:, :, :3]

        return array

    def decode_compressed_image(self, data_type: DataType, message: Any) -> np.ndarray:
        """Decode CompressedImage-like messages into numpy arrays."""
        if not self._has_pil or self._image_module is None:
            raise ImportError(
                "Compressed image decoding requires pillow. "
                "Install with `pip install neuracore[import]`."
            )
        raw = self.get_field(message, "data")
        if raw is None:
            raise ImportError("Compressed image decoding requires data field.")
        raw_bytes = bytes(raw)
        try:
            with self._image_module.open(io.BytesIO(raw_bytes)) as img:
                if data_type == DataType.RGB_IMAGES and img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img)
                if data_type == DataType.DEPTH_IMAGES and arr.dtype not in (
                    np.float16,
                    np.float32,
                ):
                    arr = arr.astype(np.float32, copy=False)
        except Exception as exc:  # noqa: BLE001
            raise ImportError(f"Failed decoding compressed image: {exc}") from exc
        return arr

    def coerce_message_data(self, data_type: DataType, data: Any, message: Any) -> Any:
        """Convert image payloads into numpy-friendly forms."""
        if data_type not in {DataType.RGB_IMAGES, DataType.DEPTH_IMAGES}:
            return data
        if not (
            isinstance(data, (bytes, bytearray, memoryview))
            or self.looks_like_bytes(data)
        ):
            return data
        if self.has_raw_image_fields(message):
            return self.decode_raw_image(data_type, data, message)
        if self.has_compressed_image_fields(message):
            return self.decode_compressed_image(data_type, message)
        return data


class McapMessageDecoder:
    """Decode MCAP messages with per-channel cache and safe plain conversion."""

    def __init__(
        self,
        decoder_factories: list[DecoderFactory],
        logger: logging.Logger,
        *,
        max_to_plain_depth: int,
        max_to_plain_repr_chars: int,
        decoder_cache: dict[int, Any] | None = None,
        raw_channel_ids: set[int] | None = None,
    ) -> None:
        """Initialize decoder factories, cache state, and plain-data limits."""
        self._decoder_factories = decoder_factories
        self._logger = logger
        self.decoder_cache: dict[int, Any] = (
            decoder_cache if decoder_cache is not None else {}
        )
        self.raw_channel_ids: set[int] = (
            raw_channel_ids if raw_channel_ids is not None else set()
        )
        self._max_to_plain_depth = max_to_plain_depth
        self._max_to_plain_repr_chars = max_to_plain_repr_chars

    def reset_state(self) -> None:
        """Clear per-channel caches before reading a new MCAP file."""
        self.decoder_cache.clear()
        self.raw_channel_ids.clear()

    def has_decoder_for(self, message_encoding: str, schema: Any | None) -> bool:
        """Check whether any decoder factory can decode this channel."""
        for factory in self._decoder_factories:
            try:
                decoder = factory.decoder_for(message_encoding, schema)
            except Exception:  # noqa: BLE001
                continue
            if decoder is not None:
                return True
        return False

    def validate_decoder_support(self, summary: Any | None, topics: list[str]) -> None:
        """Warn when channels in scope have no decoder support."""
        if summary is None or not summary.channels:
            return
        schemas = getattr(summary, "schemas", {}) or {}
        for channel in summary.channels.values():
            if channel.topic not in topics:
                continue
            encoding = (channel.message_encoding or "").lower()
            schema = schemas.get(getattr(channel, "schema_id", 0), None)

            if encoding == MessageEncoding.CBOR.lower() and not HAS_CBOR:
                self._logger.warning(
                    "MCAP channel '%s' uses CBOR encoding but cbor2 is not installed. "
                    "Falling back to raw payload dictionaries.",
                    channel.topic,
                )
                continue

            if self.has_decoder_for(channel.message_encoding, schema):
                continue

            self._logger.warning(
                "No decoder currently available for topic '%s' (encoding=%s). "
                "Using raw payload fallback for this channel.",
                channel.topic,
                channel.message_encoding or "<empty>",
            )

    def build_raw_payload(
        self, schema: Any, channel: Any, message: Any
    ) -> dict[str, Any]:
        """Wrap undecodable messages in a structured raw payload dict."""
        schema_payload = None
        if schema is not None:
            schema_payload = {
                "id": getattr(schema, "id", None),
                "name": getattr(schema, "name", None),
                "encoding": getattr(schema, "encoding", None),
            }
        return {
            "data": bytes(message.data),
            "topic": channel.topic,
            "message_encoding": channel.message_encoding,
            "schema": schema_payload,
            "log_time_ns": getattr(message, "log_time", None),
            "publish_time_ns": getattr(message, "publish_time", None),
        }

    def decode_message(self, schema: Any, channel: Any, message: Any) -> Any | None:
        """Decode a message using decoder factories with fallback behavior."""
        if message.channel_id in self.raw_channel_ids:
            return self.build_raw_payload(schema, channel, message)

        cached = self.decoder_cache.get(message.channel_id)
        if cached is not None:
            try:
                return cached(message.data)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Cached decoder failed for topic '%s' (encoding=%s): %s. "
                    "Falling back to raw payloads for this channel.",
                    channel.topic,
                    channel.message_encoding,
                    exc,
                )
                self.decoder_cache.pop(message.channel_id, None)
                self.raw_channel_ids.add(message.channel_id)
                return self.build_raw_payload(schema, channel, message)

        for factory in self._decoder_factories:
            try:
                decoder = factory.decoder_for(channel.message_encoding, schema)
            except Exception:  # noqa: BLE001
                continue
            if decoder is None:
                continue
            try:
                decoded = decoder(message.data)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Decoder '%s' failed for topic '%s' (encoding=%s): %s. "
                    "Falling back to raw payloads for this channel.",
                    f"{decoder.__class__.__module__}.{decoder.__class__.__qualname__}",
                    channel.topic,
                    channel.message_encoding,
                    exc,
                )
                self.raw_channel_ids.add(message.channel_id)
                return self.build_raw_payload(schema, channel, message)
            self.decoder_cache[message.channel_id] = decoder
            return decoded

        self.raw_channel_ids.add(message.channel_id)
        self._logger.debug(
            "No decoder available for topic '%s' (encoding=%s). "
            "Using raw payload fallback.",
            channel.topic,
            channel.message_encoding,
        )
        return self.build_raw_payload(schema, channel, message)

    def to_plain_data(
        self,
        value: Any,
        seen: set[int] | None = None,
        depth: int = 0,
    ) -> Any:
        """Convert decoded objects into pickle-safe plain Python values."""
        if depth >= self._max_to_plain_depth:
            return "<max-depth-exceeded>"
        if seen is None:
            seen = set()
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, np.ndarray):
            return value

        value_id = id(value)
        if value_id in seen:
            return f"<circular-reference:{type(value).__name__}>"
        seen.add(value_id)
        try:
            if isinstance(value, dict):
                return {
                    str(k): self.to_plain_data(v, seen=seen, depth=depth + 1)
                    for k, v in value.items()
                }
            if isinstance(value, (list, tuple, set)):
                return [
                    self.to_plain_data(v, seen=seen, depth=depth + 1) for v in value
                ]

            if hasattr(value, "ListFields") and hasattr(value, "DESCRIPTOR"):
                out: dict[str, Any] = {}
                descriptor = getattr(value, "DESCRIPTOR", None)
                fields = getattr(descriptor, "fields", None)
                if fields is not None:
                    for field in fields:
                        name = getattr(field, "name", None)
                        if not name:
                            continue
                        try:
                            out[str(name)] = self.to_plain_data(
                                getattr(value, name),
                                seen=seen,
                                depth=depth + 1,
                            )
                        except Exception:  # noqa: BLE001
                            continue
                    return out
                try:
                    for field, field_value in value.ListFields():
                        out[str(field.name)] = self.to_plain_data(
                            field_value,
                            seen=seen,
                            depth=depth + 1,
                        )
                except Exception:  # noqa: BLE001
                    pass
                if out:
                    return out

            slots = getattr(type(value), "__slots__", None)
            if slots is not None:
                names = [slots] if isinstance(slots, str) else list(slots)
                out = {}
                for name in names:
                    if not isinstance(name, str) or name.startswith("_"):
                        continue
                    try:
                        out[name] = self.to_plain_data(
                            getattr(value, name),
                            seen=seen,
                            depth=depth + 1,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                if out:
                    return out

            if hasattr(value, "__dict__"):
                out = {
                    name: self.to_plain_data(
                        field_value,
                        seen=seen,
                        depth=depth + 1,
                    )
                    for name, field_value in vars(value).items()
                    if not name.startswith("_")
                }
                if out:
                    return out

            try:
                return [
                    self.to_plain_data(v, seen=seen, depth=depth + 1) for v in value
                ]
            except Exception:  # noqa: BLE001
                pass

            text = repr(value)
            if len(text) > self._max_to_plain_repr_chars:
                truncated_chars = len(text) - self._max_to_plain_repr_chars
                return (
                    text[: self._max_to_plain_repr_chars]
                    + f"...<truncated {truncated_chars} chars>"
                )
            return text
        finally:
            seen.discard(value_id)


class RecordingSessionManager:
    """Manage Neuracore recording sessions with rotation and retries."""

    def __init__(
        self,
        logger: logging.Logger,
        refresh_active_dataset: Callable[[], None],
        max_recording_seconds: float,
        state_timeout_seconds: float,
        start_retry_count: int,
    ) -> None:
        """Initialize retry and timeout controls for recording lifecycle."""
        self._logger = logger
        self._refresh_active_dataset = refresh_active_dataset
        self._max_recording_seconds = max_recording_seconds
        self._state_timeout_seconds = state_timeout_seconds
        self._start_retry_count = start_retry_count
        self._recording_started_at = 0.0

    def _wait_for_recording_state(self, target_state: bool) -> bool:
        timeout_seconds = max(0.0, self._state_timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            current = nc.is_recording()
            if current == target_state:
                return True
            now = time.monotonic()
            if timeout_seconds == 0.0 or now >= deadline:
                return False
            time.sleep(min(RECORDING_POLL_SLEEP_SECONDS, max(0.0, deadline - now)))

    def wait_for_state(self, target_state: bool) -> bool:
        """Wait for recording state to reach the target value."""
        return self._wait_for_recording_state(target_state)

    def _on_start_retry(self, attempt: int, max_attempts: int, exc: Exception) -> None:
        self._logger.warning(
            "Failed to start recording (attempt %s/%s): %s. "
            "Retrying after forcing a stop.",
            attempt,
            max_attempts,
            exc,
        )
        if nc.is_recording():
            nc.stop_recording(wait=False)
            self._wait_for_recording_state(False)

    def start(self) -> float:
        """Start recording and return wall-clock start time."""
        self._refresh_active_dataset()
        start_impl = retry_with_backoff(
            max_attempts=self._start_retry_count,
            backoff_seconds=RECORDING_RETRY_SLEEP_SECONDS,
            on_retry=self._on_start_retry,
        )(nc.start_recording)
        try:
            start_impl()
        except Exception as exc:  # noqa: BLE001
            raise ImportError(f"Failed to start recording: {exc}") from exc
        self._recording_started_at = time.monotonic()
        return self._recording_started_at

    def rotate(self) -> float:
        """Rotate recording sessions and return new start time."""
        if nc.is_recording():
            nc.stop_recording(wait=False)
            if not self._wait_for_recording_state(False):
                self._logger.warning(
                    (
                        "Recording state did not clear within %.1fs; "
                        "forcing blocking stop."
                    ),
                    self._state_timeout_seconds,
                )
                if nc.is_recording():
                    nc.stop_recording(wait=True)
        return self.start()

    def rotate_if_needed(self) -> bool:
        """Rotate the active recording when max runtime is reached."""
        if self._max_recording_seconds <= 0:
            return False
        if (
            time.monotonic() - self._recording_started_at
        ) < self._max_recording_seconds:
            return False
        self.rotate()
        return True

    def stop(self) -> None:
        """Stop recording if currently active."""
        if nc.is_recording():
            nc.stop_recording(wait=True)


class MCAPDatasetImporter(NeuracoreDatasetImporter):
    """Importer for MCAP datasets."""

    def __init__(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        dataset_dir: Path,
        dataset_config: DatasetImportConfig,
        joint_info: dict[str, JointInfo] = {},
        dry_run: bool = False,
        suppress_warnings: bool = False,
        *,
        max_workers: int | None = 1,
        skip_on_error: str = "episode",
    ) -> None:
        """Initialize the MCAP dataset importer."""
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_config=dataset_config,
            output_dataset_name=output_dataset_name,
            max_workers=max_workers,
            skip_on_error=skip_on_error,
            joint_info=joint_info,
            dry_run=dry_run,
            suppress_warnings=suppress_warnings,
        )
        self.dataset_name = input_dataset_name
        self.dataset_dir = Path(dataset_dir)
        self.mcap_files = self._discover_mcap_files(self.dataset_dir)
        self._topic_map = self._build_topic_map()
        self._decoder_factories = self._build_decoder_factories()
        self._decoder_cache: dict[int, Any] = {}
        self._raw_channel_ids: set[int] = set()
        self._max_recording_seconds = env_float(
            "NEURACORE_IMPORT_MAX_RECORDING_SECONDS", 270.0, minimum=0.0
        )
        self._recording_state_timeout_seconds = env_float(
            "NEURACORE_IMPORT_RECORDING_STATE_TIMEOUT_SECONDS", 5.0, minimum=0.0
        )
        self._recording_start_retry_count = env_int(
            "NEURACORE_IMPORT_RECORDING_START_RETRY_COUNT", 3, minimum=1
        )
        stage_dir = os.getenv("NEURACORE_MCAP_STAGE_DIR", "").strip()
        self._stage_dir: Path | None = (
            Path(stage_dir).expanduser() if stage_dir else None
        )
        if self._stage_dir is not None:
            self._stage_dir.mkdir(parents=True, exist_ok=True)
        self._stage_heartbeat_seconds = env_float(
            "NEURACORE_MCAP_STAGE_HEARTBEAT_SECONDS", 15.0, minimum=0.0
        )
        self._inline_progress_every = env_int(
            "NEURACORE_MCAP_INLINE_PROGRESS_EVERY", 2000, minimum=1
        )
        self._reuse_main_process_session = env_bool(
            "NEURACORE_MCAP_REUSE_MAIN_PROCESS_SESSION",
            default=True,
        )
        self._to_plain_max_depth = env_int(
            "NEURACORE_MCAP_TO_PLAIN_MAX_DEPTH", MAX_TO_PLAIN_DEPTH, minimum=1
        )
        self._to_plain_max_repr_chars = env_int(
            "NEURACORE_MCAP_TO_PLAIN_MAX_REPR_CHARS",
            MAX_TO_PLAIN_REPR_CHARS,
            minimum=64,
        )
        self._image_decoder = ImageDecoder(self.logger, HAS_PIL, Image)
        self._message_decoder = McapMessageDecoder(
            decoder_factories=self._decoder_factories,
            logger=self.logger,
            max_to_plain_depth=self._to_plain_max_depth,
            max_to_plain_repr_chars=self._to_plain_max_repr_chars,
            decoder_cache=self._decoder_cache,
            raw_channel_ids=self._raw_channel_ids,
        )
        self._recording_session = RecordingSessionManager(
            logger=self.logger,
            refresh_active_dataset=self._refresh_active_dataset,
            max_recording_seconds=self._max_recording_seconds,
            state_timeout_seconds=self._recording_state_timeout_seconds,
            start_retry_count=self._recording_start_retry_count,
        )
        self._active_dataset_id: str | None = None
        self.logger.info(
            "Initialized MCAP importer for '%s' (files=%s, topics=%s, root=%s)",
            self.dataset_name,
            len(self.mcap_files),
            len(self._topic_map),
            self.dataset_dir,
        )
        # `all` aborts immediately, while `episode`/`step` continue at item level.
        self.continue_on_error = self.skip_on_error != "all"
        self.logger.info(
            (
                "MCAP importer runtime options: mode=decode-first "
                "max_recording_seconds=%.1f "
                "recording_state_timeout_seconds=%.1f stage_dir=%s"
            ),
            self._max_recording_seconds,
            self._recording_state_timeout_seconds,
            str(self._stage_dir) if self._stage_dir is not None else "<system-temp>",
        )

    def __getstate__(self) -> dict:
        """Drop worker-local handles when pickling for multiprocessing."""
        state = self.__dict__.copy()
        return state

    def build_work_items(self) -> list[ImportItem]:
        """Build work items for the dataset importer."""
        return [
            ImportItem(index=i, description=path.name, metadata={"path": str(path)})
            for i, path in enumerate(self.mcap_files)
        ]

    def get_message_decoder(self) -> McapMessageDecoder:
        """Build a decoder lazily for tests that instantiate via __new__."""
        decoder = getattr(self, "_message_decoder", None)
        if decoder is None:
            logger = getattr(self, "logger", MODULE_LOGGER)
            decoder = McapMessageDecoder(
                decoder_factories=getattr(self, "_decoder_factories", []),
                logger=logger,
                max_to_plain_depth=getattr(
                    self, "_to_plain_max_depth", MAX_TO_PLAIN_DEPTH
                ),
                max_to_plain_repr_chars=getattr(
                    self,
                    "_to_plain_max_repr_chars",
                    MAX_TO_PLAIN_REPR_CHARS,
                ),
                decoder_cache=getattr(self, "_decoder_cache", {}),
                raw_channel_ids=getattr(self, "_raw_channel_ids", set()),
            )
            self._message_decoder = decoder
        self._decoder_cache = decoder.decoder_cache
        self._raw_channel_ids = decoder.raw_channel_ids
        return decoder

    def get_image_decoder(self) -> ImageDecoder:
        """Build an image decoder lazily for tests that instantiate via __new__."""
        image_decoder = getattr(self, "_image_decoder", None)
        if image_decoder is None:
            logger = getattr(self, "logger", MODULE_LOGGER)
            image_decoder = ImageDecoder(logger, HAS_PIL, Image)
            self._image_decoder = image_decoder
        return image_decoder

    def get_recording_session(self) -> RecordingSessionManager:
        """Build recording session manager lazily for tests using __new__."""
        session = getattr(self, "_recording_session", None)
        if session is None:
            logger = getattr(self, "logger", MODULE_LOGGER)
            session = RecordingSessionManager(
                logger=logger,
                refresh_active_dataset=self._refresh_active_dataset,
                max_recording_seconds=getattr(self, "_max_recording_seconds", 270.0),
                state_timeout_seconds=getattr(
                    self,
                    "_recording_state_timeout_seconds",
                    5.0,
                ),
                start_retry_count=getattr(self, "_recording_start_retry_count", 3),
            )
            self._recording_session = session
        return session

    def prepare_worker(
        self, worker_id: int, chunk: Sequence[ImportItem] | None = None
    ) -> None:
        """Prepare worker state and Neuracore connections."""
        if (
            mp.current_process().name == "MainProcess"
            and self._reuse_main_process_session
        ):
            self.logger.info(
                "[worker %s] Reusing main-process Neuracore session.",
                worker_id,
            )
        else:
            super().prepare_worker(worker_id, chunk)
        self._active_dataset_id = None

    def upload_all(self) -> None:
        """Run MCAP imports inline for single-worker stability and observability."""
        items = list(self.build_work_items())
        if not items:
            self.logger.info("No upload items found; nothing to do.")
            return

        self.logger.info(
            "Preparing import -> dataset=%s | source_dir=%s | items=%s | "
            "continue_on_error=%s",
            self.output_dataset_name,
            self.dataset_dir,
            len(items),
            self.continue_on_error,
        )
        worker_count = self._resolve_worker_count(len(items))
        cpu_count = os.cpu_count()
        self.logger.info(
            "Scheduling %s items across %s worker(s) "
            "(min=%s max=%s cpu=%s progress_interval=%s)",
            len(items),
            worker_count,
            self.min_workers,
            self.max_workers if self.max_workers is not None else "auto",
            cpu_count if cpu_count is not None else "unknown",
            self.progress_interval,
        )

        if worker_count != 1:
            super().upload_all()
            return

        worker_id = 0
        self._worker_id = worker_id
        self._progress_queue = None
        self.worker_errors = []

        try:
            self.prepare_worker(worker_id, items)
            self.logger.info(
                "[worker %s] Starting chunk (%s items): %s → %s",
                worker_id,
                len(items),
                items[0].index,
                items[-1].index,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            err = WorkerError(
                worker_id=worker_id,
                item_index=None,
                message=str(exc),
                traceback=tb,
            )
            self.worker_errors = [err]
            self._log_worker_error(worker_id, None, str(exc))
            self._report_errors(self.worker_errors)
            raise UploaderError("Upload aborted due to worker setup failure.") from exc

        for local_index, item in enumerate(items):
            try:
                self.upload(item)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                err = WorkerError(
                    worker_id=worker_id,
                    item_index=item.index,
                    message=str(exc),
                    traceback=tb,
                )
                self.worker_errors.append(err)
                self._log_worker_error(worker_id, item.index, str(exc))
                if self.continue_on_error:
                    self.logger.warning(
                        "[worker %s] Continuing after failure on item %s "
                        "(continue_on_error=True).",
                        worker_id,
                        item.index,
                    )
                    continue
                self._report_errors(self.worker_errors)
                raise UploaderError("Upload aborted due to worker errors.") from exc

            if (local_index + 1) % self.progress_interval == 0 or (
                local_index + 1 == len(items)
            ):
                self.logger.info(
                    "[worker %s] processed %s/%s (item index=%s)",
                    worker_id,
                    local_index + 1,
                    len(items),
                    item.index,
                )

        self._report_errors(self.worker_errors)
        if self.worker_errors and not self.continue_on_error:
            raise UploaderError("Upload aborted due to worker errors.")

    def _refresh_active_dataset(self) -> None:
        """Refresh active dataset context for recording start calls."""
        if self._active_dataset_id:
            try:
                dataset = nc.get_dataset(id=self._active_dataset_id)
                self._active_dataset_id = dataset.id
                return
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Cached active dataset id '%s' is unavailable (%s). "
                    "Falling back to dataset name '%s'.",
                    self._active_dataset_id,
                    exc,
                    self.output_dataset_name,
                )

        retries = env_int("NEURACORE_IMPORT_DATASET_RESOLVE_RETRIES", 8, minimum=1)
        backoff_seconds = env_float(
            "NEURACORE_IMPORT_DATASET_RESOLVE_BACKOFF_SECONDS", 0.25, minimum=0.0
        )

        def _on_retry(attempt: int, max_attempts: int, exc: Exception) -> None:
            self.logger.warning(
                "Failed to resolve dataset by name '%s' (attempt %s/%s): %s. "
                "Retrying in %.2fs.",
                self.output_dataset_name,
                attempt,
                max_attempts,
                exc,
                backoff_seconds,
            )

        resolver = retry_with_backoff(
            max_attempts=retries,
            backoff_seconds=backoff_seconds,
            on_retry=_on_retry,
        )(lambda: nc.get_dataset(self.output_dataset_name))
        try:
            dataset = resolver()
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Unable to resolve active dataset context for recording: "
                f"name='{self.output_dataset_name}' id='{self._active_dataset_id}'. "
                f"Original error: {exc}"
            ) from exc
        self._active_dataset_id = dataset.id

    def _start_recording_session(self) -> float:
        """Start a recording session and return its wall-clock start time."""
        return self.get_recording_session().start()

    def _rotate_recording_session(self) -> float:
        """Rotate recording sessions and wait for local state to clear."""
        return self.get_recording_session().rotate()

    def _wait_for_recording_state(self, target_state: bool) -> bool:
        """Wait for recording state to reach the target value."""
        return self.get_recording_session().wait_for_state(target_state)

    def _resolve_timestamp_seconds(self, message: Any) -> float:
        """Resolve message timestamp in seconds with robust fallbacks."""
        log_time_ns = int(getattr(message, "log_time", 0) or 0)
        if log_time_ns > 0:
            return log_time_ns / 1e9
        publish_time_ns = int(getattr(message, "publish_time", 0) or 0)
        if publish_time_ns > 0:
            return publish_time_ns / 1e9
        return time.time()

    def upload(self, item: ImportItem) -> None:
        """Upload a single MCAP file as an episode."""
        file_path = Path(item.metadata.get("path", "")) if item.metadata else None
        if file_path is None or not file_path.exists():
            raise ImportError(f"MCAP file not found for item {item.index}.")

        worker_label = (
            f"worker {self._worker_id}" if self._worker_id is not None else "worker 0"
        )
        self.logger.info("[%s] Importing MCAP file: %s", worker_label, file_path)

        topics = list(self._topic_map.keys())
        total_messages = None
        self.get_message_decoder().reset_state()

        with file_path.open("rb") as f:
            reader = make_reader(f)
            summary = None
            try:
                summary = reader.get_summary()
            except Exception:  # noqa: BLE001 - summary is optional
                summary = None

            if (
                summary
                and summary.statistics
                and summary.statistics.channel_message_counts
            ):
                total_messages = self._estimate_total_messages(summary, topics)

            self._validate_requested_topics(summary, topics)
            self._validate_decoder_support(summary, topics)

            self._upload_with_decode_staging(
                reader=reader,
                topics=topics,
                item=item,
                total_messages=total_messages,
                worker_label=worker_label,
                file_name=file_path.name,
            )

        self.logger.info(
            "[%s] Completed MCAP import for %s", worker_label, file_path.name
        )

    def _upload_with_decode_staging(
        self,
        reader: Any,
        topics: list[str],
        item: ImportItem,
        total_messages: int | None,
        worker_label: str,
        file_name: str,
    ) -> None:
        """Decode MCAP records to disk first, then start recording and replay."""
        temp_dir_args: dict[str, Any] = {"prefix": "neuracore_mcap_stage_"}
        if self._stage_dir is not None:
            temp_dir_args["dir"] = str(self._stage_dir)
        progress_ctx = (
            self._inline_progress_context()
            if self._progress_queue is None
            else nullcontext(None)
        )
        with tempfile.TemporaryDirectory(
            **temp_dir_args
        ) as tmp_dir, progress_ctx as progress:
            staged_path = Path(tmp_dir) / "decoded_records.bin"
            decode_task_id: TaskID | None = None
            replay_task_id: TaskID | None = None
            rich_progress_active = progress is not None
            if progress is not None:
                decode_task_id = progress.add_task(
                    f"{file_name} (decode)",
                    total=total_messages,
                    completed=0,
                )
            staged_count = self._stage_decoded_records(
                reader=reader,
                topics=topics,
                staged_path=staged_path,
                worker_label=worker_label,
                file_name=file_name,
                total_messages=total_messages,
                emit_heartbeat_logs=not rich_progress_active,
                progress_callback=(
                    (
                        lambda count: self._update_progress(
                            progress, decode_task_id, count
                        )
                    )
                    if progress is not None and decode_task_id is not None
                    else None
                ),
            )

            if staged_count == 0:
                self.logger.warning(
                    "[%s] No messages matched configured topics for %s",
                    worker_label,
                    file_name,
                )
                return

            staged_bytes = staged_path.stat().st_size
            self.logger.info(
                "[%s] Finished staging %s -> %s message(s), %.2f MiB",
                worker_label,
                file_name,
                staged_count,
                staged_bytes / (1024 * 1024),
            )
            if progress is not None and decode_task_id is not None:
                progress.update(
                    decode_task_id,
                    total=max(staged_count, total_messages or 0) or staged_count,
                    completed=staged_count,
                    refresh=True,
                )
                replay_task_id = progress.add_task(
                    f"{file_name} (replay)",
                    total=staged_count,
                    completed=0,
                )

            processed = 0
            recording_started = False
            session = self.get_recording_session()
            heartbeat_interval = (
                0.0
                if rich_progress_active
                else env_float("NEURACORE_IMPORT_HEARTBEAT_SECONDS", 15.0, 0.0)
            )
            last_heartbeat_time = time.monotonic()
            last_heartbeat_count = 0
            try:
                if nc.is_recording():
                    self.logger.warning(
                        "Detected an already-active recording before replay; "
                        "stopping it to start a fresh session."
                    )
                    session.stop()
                session.start()
                recording_started = True

                for record in self._iter_staged_records(staged_path):
                    topic = record.topic
                    timestamp = record.timestamp
                    log_time_ns = record.log_time_ns
                    payload = record.payload
                    if not nc.is_recording():
                        self.logger.warning(
                            "Recording became inactive mid-replay; "
                            "starting a fresh recording and continuing."
                        )
                        session.start()
                        recording_started = True

                    if session.rotate_if_needed():
                        self.logger.info(
                            (
                                "Rotated recording after %.1fs to avoid backend "
                                "TTL expiry."
                            ),
                            self._max_recording_seconds,
                        )

                    try:
                        self._record_step({topic: payload}, timestamp)
                    except Exception as exc:  # noqa: BLE001
                        raise ImportError(
                            "Failed replaying staged MCAP message "
                            f"(topic={topic}, index={processed + 1}, "
                            f"log_time_ns={log_time_ns}): {exc}"
                        ) from exc

                    processed += 1
                    if progress is not None and replay_task_id is not None:
                        self._update_progress(progress, replay_task_id, processed)
                    if heartbeat_interval > 0:
                        now = time.monotonic()
                        if now - last_heartbeat_time >= heartbeat_interval:
                            interval = max(now - last_heartbeat_time, 1e-6)
                            delta = processed - last_heartbeat_count
                            rate = delta / interval
                            total_label = (
                                str(total_messages)
                                if total_messages is not None
                                else str(staged_count)
                            )
                            self.logger.info(
                                "[%s] %s replay progress: %s/%s messages (%.1f msg/s)",
                                worker_label,
                                file_name,
                                processed,
                                total_label,
                                rate,
                            )
                            last_heartbeat_time = now
                            last_heartbeat_count = processed
                    if processed % self.progress_interval == 0:
                        self._emit_progress(
                            item.index,
                            step=processed,
                            total_steps=total_messages or staged_count,
                            episode_label=item.description,
                        )
                        if (
                            not rich_progress_active
                            and self._progress_queue is None
                            and (
                                processed % self._inline_progress_every == 0
                                or processed == staged_count
                            )
                        ):
                            self.logger.info(
                                "[%s] %s progress: %s/%s messages",
                                worker_label,
                                file_name,
                                processed,
                                total_messages or staged_count,
                            )
            finally:
                if recording_started:
                    session.stop()

    def _stage_decoded_records(
        self,
        reader: Any,
        topics: list[str],
        staged_path: Path,
        worker_label: str,
        file_name: str,
        total_messages: int | None,
        emit_heartbeat_logs: bool = True,
        progress_callback: Callable[[int], None] | None = None,
    ) -> int:
        """Decode messages and stream staged records to disk."""
        self.logger.info(
            "[%s] Pre-decoding %s to disk before recording (staging=%s)",
            worker_label,
            file_name,
            staged_path,
        )
        staged_count = 0
        last_heartbeat_time = time.monotonic()
        last_heartbeat_count = 0

        with staged_path.open("wb") as staged_file:
            for schema, channel, message in reader.iter_messages(
                topics=topics, log_time_order=True
            ):
                decoded = self._decode_message(schema, channel, message)
                timestamp = self._resolve_timestamp_seconds(message)
                plain_payload = self._to_plain_data(decoded)
                self._write_staged_record(
                    staged_file,
                    topic=channel.topic,
                    timestamp=timestamp,
                    log_time_ns=int(getattr(message, "log_time", 0) or 0),
                    payload=plain_payload,
                )
                staged_count += 1
                if progress_callback is not None:
                    progress_callback(staged_count)

                if emit_heartbeat_logs and self._stage_heartbeat_seconds > 0:
                    now = time.monotonic()
                    if now - last_heartbeat_time >= self._stage_heartbeat_seconds:
                        interval = max(now - last_heartbeat_time, 1e-6)
                        delta = staged_count - last_heartbeat_count
                        rate = delta / interval
                        total_label = (
                            str(total_messages)
                            if total_messages is not None
                            else "unknown"
                        )
                        self.logger.info(
                            "[%s] %s decode progress: %s/%s messages (%.1f msg/s)",
                            worker_label,
                            file_name,
                            staged_count,
                            total_label,
                            rate,
                        )
                        last_heartbeat_time = now
                        last_heartbeat_count = staged_count
        return staged_count

    def _inline_progress_context(self) -> Progress:
        """Build a rich progress renderer for inline decode/replay flow."""
        return Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None, complete_style="green", pulse_style="cyan"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
            transient=True,
            console=get_shared_console(),
        )

    def _update_progress(
        self, progress: Progress, task_id: TaskID | None, completed: int
    ) -> None:
        """Best-effort progress updates to avoid import interruption."""
        if task_id is None:
            return
        try:
            progress.update(task_id, completed=completed, refresh=True)
        except Exception:  # noqa: BLE001
            self.logger.debug("Failed to update inline progress bar.", exc_info=True)

    def _write_staged_record(
        self,
        staged_file: Any,
        topic: str,
        timestamp: float,
        log_time_ns: int,
        payload: Any,
    ) -> None:
        """Write a single staged record as length-prefixed pickle bytes."""
        blob = StagedRecord(
            topic=str(topic),
            timestamp=float(timestamp),
            log_time_ns=int(log_time_ns),
            payload=payload,
        ).to_pickle_bytes()
        staged_file.write(struct.pack("<Q", len(blob)))
        staged_file.write(blob)

    def _iter_staged_records(self, staged_path: Path) -> Iterator[StagedRecord]:
        """Yield staged records from disk in write order."""
        with staged_path.open("rb") as staged_file:
            while True:
                size_buf = staged_file.read(8)
                if not size_buf:
                    break
                if len(size_buf) != 8:
                    raise ImportError("Corrupt staged MCAP record size header.")
                size = struct.unpack("<Q", size_buf)[0]
                blob = staged_file.read(size)
                if len(blob) != size:
                    raise ImportError("Corrupt staged MCAP record payload.")
                try:
                    yield StagedRecord.from_pickle_bytes(blob)
                except ValueError as exc:
                    raise ImportError(str(exc)) from exc

    def _to_plain_data(
        self,
        value: Any,
        seen: set[int] | None = None,
        depth: int = 0,
    ) -> Any:
        """Convert decoded objects into pickle-safe plain Python values."""
        return self.get_message_decoder().to_plain_data(
            value=value,
            seen=seen,
            depth=depth,
        )

    def _record_step(self, step_data: dict, timestamp: float) -> None:
        """Record a single step to Neuracore."""
        for topic, configs in self._topic_map.items():
            if topic not in step_data:
                continue
            message_data = step_data[topic]
            for config in configs:
                if config.mapping_item is not None:
                    base = message_data
                    if config.item_base_path:
                        base = self._resolve_path(base, config.item_base_path)

                    source_data = self._coerce_message_data(
                        config.data_type, base, message_data
                    )

                    if not (
                        config.data_type == DataType.LANGUAGE
                        and config.import_config.format.language_type
                        == LanguageConfig.STRING
                    ):
                        source_data = self._to_numpy(source_data)

                    self._log_data(
                        config.data_type,
                        source_data,
                        config.mapping_item,
                        config.import_config.format,
                        timestamp,
                    )
                    continue

                base = self._resolve_path(message_data, config.source_path)
                for item in config.import_config.mapping:
                    if item.source_name:
                        source_data = self._resolve_path(
                            base, item.source_name.split(".")
                        )
                    elif item.index is not None:
                        source_data = base[item.index]
                    elif item.index_range is not None:
                        source_data = base[
                            item.index_range.start : item.index_range.end
                        ]
                    else:
                        source_data = base

                    source_data = self._coerce_message_data(
                        config.data_type, source_data, message_data
                    )

                    if not (
                        config.data_type == DataType.LANGUAGE
                        and config.import_config.format.language_type
                        == LanguageConfig.STRING
                    ):
                        source_data = self._to_numpy(source_data)

                    self._log_data(
                        config.data_type,
                        source_data,
                        item,
                        config.import_config.format,
                        timestamp,
                    )

    def _discover_mcap_files(self, dataset_dir: Path) -> list[Path]:
        """Find MCAP files in the dataset directory or treat file path as input."""
        if dataset_dir.is_file():
            if dataset_dir.suffix.lower() != ".mcap":
                raise ImportError(
                    f"Expected an MCAP file, got '{dataset_dir.name}' instead."
                )
            return [dataset_dir]

        if not dataset_dir.exists():
            raise ImportError(f"Dataset path does not exist: {dataset_dir}")

        mcap_files = sorted(dataset_dir.rglob("*.mcap"))
        if not mcap_files:
            raise ImportError(
                f"No MCAP files found under '{dataset_dir}'. "
                "Provide a .mcap file or a directory containing MCAP files."
            )
        return mcap_files

    def _build_topic_map(self) -> dict[str, list[TopicImportConfig]]:
        """Map topics to import configurations for quick lookup."""
        topic_map: dict[str, list[TopicImportConfig]] = {}
        for data_type, import_config in self.dataset_config.data_import_config.items():
            source = (import_config.source or "").strip()

            absolute_topic_items = [
                item
                for item in import_config.mapping
                if item.source_name and item.source_name.startswith("/")
            ]
            relative_items = [
                item
                for item in import_config.mapping
                if not (item.source_name and item.source_name.startswith("/"))
            ]

            if relative_items:
                if not source:
                    raise ImportError(
                        f"Missing source for data type '{data_type.value}'. "
                        "Relative mapping entries require a base source path."
                    )
                topic, subpath = self._split_source(source)
                topic_map.setdefault(topic, []).append(
                    TopicImportConfig(
                        data_type=data_type,
                        import_config=self._copy_import_config_with_mapping(
                            import_config, relative_items
                        ),
                        source_path=subpath,
                    )
                )

            for item in absolute_topic_items:
                item_topic, item_subpath = self._split_source(item.source_name)
                topic_map.setdefault(item_topic, []).append(
                    TopicImportConfig(
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
        self, import_config: Any, mapping: list[MappingItem]
    ) -> Any:
        """Clone an import config while replacing its mapping list."""
        mapping_copy = list(mapping)
        if hasattr(import_config, "model_copy"):
            return import_config.model_copy(update={"mapping": mapping_copy})
        cloned = copy(import_config)
        setattr(cloned, "mapping", mapping_copy)
        return cloned

    def _build_decoder_factories(self) -> list[DecoderFactory]:
        """Create decoder factories based on available dependencies."""
        factories: list[DecoderFactory] = []
        seen_factory_types: set[str] = set()

        def _add(factory: DecoderFactory) -> None:
            key = f"{factory.__class__.__module__}." f"{factory.__class__.__qualname__}"
            if key in seen_factory_types:
                return
            seen_factory_types.add(key)
            factories.append(factory)

        _add(JsonDecoderFactory())
        _add(TextDecoderFactory())
        _add(CborDecoderFactory())
        if HAS_PROTOBUF and ProtobufDecoderFactory is not None:
            _add(ProtobufDecoderFactory())
        if HAS_ROS1 and Ros1DecoderFactory is not None:
            _add(Ros1DecoderFactory())
        if HAS_ROS2 and Ros2DecoderFactory is not None:
            _add(Ros2DecoderFactory())
        for factory in self._discover_additional_decoder_factories():
            _add(factory)
        return factories

    def _discover_additional_decoder_factories(self) -> list[DecoderFactory]:
        """Discover installed MCAP decoder plugins dynamically."""
        factories: list[DecoderFactory] = []
        module_names = {
            module.name
            for module in pkgutil.iter_modules()
            if module.name.startswith("mcap_")
        }
        try:
            for dist in importlib_metadata.distributions():
                dist_name = (dist.metadata.get("Name") or "").strip().lower()
                if not dist_name.startswith("mcap-"):
                    continue
                module_names.add(dist_name.replace("-", "_"))
        except Exception:  # noqa: BLE001
            self.logger.debug(
                "Failed while scanning installed distributions.", exc_info=True
            )

        for module_name in sorted(module_names):
            decoder_module_name = f"{module_name}.decoder"
            try:
                decoder_module = importlib.import_module(decoder_module_name)
            except Exception:  # noqa: BLE001
                continue
            decoder_factory_cls = getattr(decoder_module, "DecoderFactory", None)
            if (
                decoder_factory_cls is None
                or not isinstance(decoder_factory_cls, type)
                or not issubclass(decoder_factory_cls, DecoderFactory)
            ):
                continue
            try:
                factories.append(decoder_factory_cls())
            except Exception:  # noqa: BLE001
                self.logger.debug(
                    "Failed to initialize decoder factory from %s",
                    decoder_module_name,
                    exc_info=True,
                )
        return factories

    def _validate_decoder_support(self, summary: Any | None, topics: list[str]) -> None:
        """Ensure required decoders are available for the requested topics."""
        self.get_message_decoder().validate_decoder_support(summary, topics)

    def _has_decoder_for(self, message_encoding: str, schema: Any | None) -> bool:
        """Check whether any configured decoder factory can decode this channel."""
        return self.get_message_decoder().has_decoder_for(message_encoding, schema)

    def _validate_requested_topics(
        self, summary: Any | None, topics: list[str]
    ) -> None:
        """Ensure configured topics exist in the MCAP summary when available."""
        if summary is None or not summary.channels or not topics:
            return
        available_topics = {channel.topic for channel in summary.channels.values()}
        missing = sorted(topic for topic in topics if topic not in available_topics)
        if missing:
            shown_available = ", ".join(sorted(available_topics)[:20])
            raise ImportError(
                "Configured topic(s) not present in MCAP: "
                f"{', '.join(missing)}. "
                f"Available topics include: {shown_available}"
            )

    def _decode_message(self, schema: Any, channel: Any, message: Any) -> Any | None:
        """Decode a message using the available decoder factories."""
        decoder = self.get_message_decoder()
        decoded = decoder.decode_message(schema, channel, message)
        self._decoder_cache = decoder.decoder_cache
        self._raw_channel_ids = decoder.raw_channel_ids
        return decoded

    def _build_raw_payload(
        self, schema: Any, channel: Any, message: Any
    ) -> dict[str, Any]:
        """Wrap undecodable messages in a structured raw payload dict."""
        return self.get_message_decoder().build_raw_payload(schema, channel, message)

    def _estimate_total_messages(self, summary: Any, topics: list[str]) -> int | None:
        """Estimate message count for requested topics using summary statistics."""
        if not summary.statistics or not summary.statistics.channel_message_counts:
            return None
        channel_counts = summary.statistics.channel_message_counts
        total = 0
        for channel_id, channel in summary.channels.items():
            if channel.topic in topics:
                total += int(channel_counts.get(channel_id, 0))
        return total if total > 0 else None

    def _split_source(self, source: str) -> tuple[str, list[str]]:
        """Split a source into topic and nested field path."""
        source = source.strip()
        if not source:
            raise ImportError("Source must include a topic.")
        topic, sep, subpath = source.partition(".")
        if not topic:
            raise ImportError("Source must include a topic.")
        path = [part for part in subpath.split(".") if part] if sep else []
        return topic, path

    def _resolve_path(self, data: Any, path: list[str]) -> Any:
        """Resolve a dotted path into nested data structures."""
        current = data
        for part in path:
            current = self._resolve_part(current, part)
        return current

    def _resolve_part(self, data: Any, part: str) -> Any:
        """Resolve a single path part from dicts, objects, or sequences."""
        if isinstance(data, dict):
            if part in data:
                return data[part]
            if part.isdigit():
                return data[int(part)]
            raise ImportError(f"Key '{part}' not found in message dict.")

        if hasattr(data, part):
            return getattr(data, part)

        if part.isdigit():
            idx = int(part)
            try:
                return data[idx]
            except Exception as exc:  # noqa: BLE001
                raise ImportError(f"Index {idx} not available: {exc}") from exc

        try:
            return data[part]
        except Exception as exc:  # noqa: BLE001
            raise ImportError(f"Failed to access '{part}': {exc}") from exc

    def _to_numpy(self, data: Any) -> Any:
        """Convert data to numpy if possible."""
        if hasattr(data, "numpy"):
            return data.numpy()
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            return np.array(data)
        return data

    def _coerce_message_data(self, data_type: DataType, data: Any, message: Any) -> Any:
        """Convert MCAP message payloads into numpy-friendly forms."""
        return self.get_image_decoder().coerce_message_data(data_type, data, message)

    def _looks_like_bytes(self, data: Any) -> bool:
        """Detect list-like byte buffers."""
        return self.get_image_decoder().looks_like_bytes(data)

    def _decode_raw_image(
        self, data_type: DataType, data: Any, message: Any
    ) -> np.ndarray:
        """Decode RawImage-like messages into numpy arrays."""
        return self.get_image_decoder().decode_raw_image(data_type, data, message)

    def _has_raw_image_fields(self, message: Any) -> bool:
        """Check for fields needed to decode raw images."""
        return self.get_image_decoder().has_raw_image_fields(message)

    def _has_compressed_image_fields(self, message: Any) -> bool:
        """Check for fields needed to decode compressed images."""
        return self.get_image_decoder().has_compressed_image_fields(message)

    def _decode_compressed_image(self, data_type: DataType, message: Any) -> np.ndarray:
        """Decode CompressedImage-like messages into numpy arrays."""
        return self.get_image_decoder().decode_compressed_image(data_type, message)

    def _get_field(self, data: Any, name: str) -> Any:
        """Fetch a field from dicts or objects."""
        return self.get_image_decoder().get_field(data, name)
