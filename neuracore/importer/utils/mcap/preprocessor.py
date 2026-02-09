"""Phase-1 preprocessor for MCAP import.

Preprocessing is intentionally separated from logging because backend recording
sessions expire after ~5 minutes. This phase performs all decode/transform work
without an active recording session and writes events to a disk cache.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcap.decoder import DecoderFactory
from mcap.reader import make_reader

from neuracore.importer.core.exceptions import ImportError

from .cache import MessageCache
from .decoder import MCAPMessageDecoder, validate_channel_decoder_support
from .progress import ProgressReporter
from .topics import TopicMapper


@dataclass(frozen=True, slots=True)
class PreprocessStats:
    """Summary metrics produced by preprocessing."""

    message_count: int
    event_count: int
    cache_size_bytes: int
    duration_seconds: float


class MessagePreprocessor:
    """Decode MCAP messages and stage transformed events into a cache file."""

    def __init__(
        self,
        mcap_file: Path,
        topic_mapper: TopicMapper,
        message_decoder: MCAPMessageDecoder,
        decoder_factories: list[DecoderFactory],
        progress_reporter: ProgressReporter,
        *,
        logger: logging.Logger,
        on_message_error: Callable[[int, str, int, Exception], bool] | None = None,
    ) -> None:
        """Initialize preprocessing dependencies for one MCAP input file."""
        self._mcap_file = mcap_file
        self._topic_mapper = topic_mapper
        self._message_decoder = message_decoder
        self._decoder_factories = list(decoder_factories)
        self._progress = progress_reporter
        self._logger = logger
        self._on_message_error = on_message_error
        self._decoder_cache: dict[int, Any] = {}
        self._raw_channel_ids: set[int] = set()

    def preprocess_to_cache(self, cache_file: Path) -> PreprocessStats:
        """Run preprocessing and write transformed messages to a cache file."""
        started_at = time.monotonic()
        topics = self._topic_mapper.get_all_topics()
        message_count = 0
        event_count = 0

        with self._mcap_file.open("rb") as input_stream:
            reader = make_reader(
                input_stream,
                decoder_factories=self._decoder_factories,
            )
            try:
                header = reader.get_header()
            except Exception:  # noqa: BLE001
                header = None
            if header is not None:
                self._logger.debug(
                    "MCAP header | profile=%s | library=%s",
                    getattr(header, "profile", "") or "<empty>",
                    getattr(header, "library", "") or "<empty>",
                )

            summary = None
            try:
                summary = reader.get_summary()
            except Exception:  # noqa: BLE001
                summary = None
            if summary is not None:
                attachment_count = len(getattr(summary, "attachment_indexes", []) or [])
                metadata_count = len(getattr(summary, "metadata_indexes", []) or [])
                if attachment_count > 0 or metadata_count > 0:
                    self._logger.debug(
                        "MCAP includes %s attachment(s) and %s metadata record(s); "
                        "the importer currently processes message records only.",
                        attachment_count,
                        metadata_count,
                    )

            self._validate_requested_topics(summary, topics)
            validate_channel_decoder_support(
                summary=summary,
                topics=topics,
                decoder_factories=self._decoder_factories,
                logger=self._logger,
            )

            total_messages = self._estimate_total_messages(summary, topics) or 0
            self._progress.start_phase("preprocess", total_messages)

            try:
                with MessageCache(cache_file, mode="wb") as cache:
                    for schema, channel, message in reader.iter_messages(
                        topics=topics,
                        log_time_order=True,
                    ):
                        message_count += 1
                        timestamp = self._resolve_timestamp_seconds(message)
                        log_time_ns = int(getattr(message, "log_time", 0) or 0)
                        topic = str(getattr(channel, "topic", "") or "")

                        try:
                            decoded_message = self._decode_message(
                                schema=schema,
                                channel=channel,
                                message=message,
                            )
                            decoded = self._message_decoder.normalize_decoded_message(
                                decoded_message
                            )
                            for (
                                event
                            ) in self._message_decoder.iter_transformed_messages(
                                topic,
                                decoded,
                                timestamp=timestamp,
                                log_time_ns=log_time_ns,
                            ):
                                cache.write_message(event)
                                event_count += 1
                        except Exception as exc:  # noqa: BLE001
                            handled = False
                            if self._on_message_error is not None:
                                handled = bool(
                                    self._on_message_error(
                                        message_count,
                                        topic,
                                        log_time_ns,
                                        exc,
                                    )
                                )
                            if handled:
                                self._progress.update(message_count)
                                continue
                            raise ImportError(
                                "Failed preprocessing MCAP message "
                                f"(topic={topic}, index={message_count}, "
                                f"log_time_ns={log_time_ns}): {exc}"
                            ) from exc

                        self._progress.update(message_count)
            finally:
                self._progress.finish_phase()

        cache_size = cache_file.stat().st_size if cache_file.exists() else 0
        duration = time.monotonic() - started_at
        return PreprocessStats(
            message_count=message_count,
            event_count=event_count,
            cache_size_bytes=cache_size,
            duration_seconds=duration,
        )

    def _estimate_total_messages(
        self,
        summary: Any | None,
        topics: list[str],
    ) -> int | None:
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

    def _validate_requested_topics(
        self, summary: Any | None, topics: list[str]
    ) -> None:
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

    @staticmethod
    def _resolve_timestamp_seconds(message: Any) -> float:
        """Resolve message timestamp from log/publish time nanoseconds."""
        log_time_ns = int(getattr(message, "log_time", 0) or 0)
        if log_time_ns > 0:
            return log_time_ns / 1e9

        publish_time_ns = int(getattr(message, "publish_time", 0) or 0)
        if publish_time_ns > 0:
            return publish_time_ns / 1e9

        return time.time()

    def _channel_key(self, message: Any, channel: Any) -> int:
        channel_id = getattr(message, "channel_id", None)
        if isinstance(channel_id, int):
            return channel_id
        return int(id(channel))

    def _build_raw_payload(
        self, schema: Any, channel: Any, message: Any
    ) -> dict[str, Any]:
        schema_payload = None
        if schema is not None:
            schema_payload = {
                "id": getattr(schema, "id", None),
                "name": getattr(schema, "name", None),
                "encoding": getattr(schema, "encoding", None),
            }
        return {
            "data": bytes(getattr(message, "data", b"")),
            "topic": str(getattr(channel, "topic", "") or ""),
            "message_encoding": str(getattr(channel, "message_encoding", "") or ""),
            "schema": schema_payload,
            "log_time_ns": int(getattr(message, "log_time", 0) or 0),
            "publish_time_ns": int(getattr(message, "publish_time", 0) or 0),
        }

    def _decode_message(self, schema: Any, channel: Any, message: Any) -> Any:
        channel_key = self._channel_key(message, channel)
        if channel_key in self._raw_channel_ids:
            return self._build_raw_payload(schema, channel, message)

        cached_decoder = self._decoder_cache.get(channel_key)
        if cached_decoder is not None:
            try:
                return cached_decoder(message.data)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Cached decoder failed for topic '%s' (encoding=%s): %s. "
                    "Falling back to raw payloads for this channel.",
                    getattr(channel, "topic", "<unknown>"),
                    getattr(channel, "message_encoding", "<unknown>"),
                    exc,
                )
                self._decoder_cache.pop(channel_key, None)
                self._raw_channel_ids.add(channel_key)
                return self._build_raw_payload(schema, channel, message)

        encoding = str(getattr(channel, "message_encoding", "") or "")
        for factory in self._decoder_factories:
            try:
                decoder = factory.decoder_for(encoding, schema)
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
                    getattr(channel, "topic", "<unknown>"),
                    encoding or "<empty>",
                    exc,
                )
                self._raw_channel_ids.add(channel_key)
                return self._build_raw_payload(schema, channel, message)
            self._decoder_cache[channel_key] = decoder
            return decoded

        self._raw_channel_ids.add(channel_key)
        return self._build_raw_payload(schema, channel, message)
