"""MCAP importer configuration.

This module centralizes MCAP-specific runtime settings so the importer has a
single source of truth. The 5-minute backend recording TTL is modeled as
constants in this config and consumed by session rotation logic.

MCAP replay rotates recordings at Neuracore's warning threshold minus a small
buffer to reduce warning noise while preserving margin before hard expiry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from neuracore.core.streaming.recording_state_manager import RecordingStateManager

    DEFAULT_BACKEND_RECORDING_TTL_SECONDS = int(
        RecordingStateManager.MAX_RECORDING_DURATION_S
    )
    # Rotate slightly before Neuracore's local 4.5-minute warning timer to avoid
    # noisy warning logs and keep margin before 5-minute expiry.
    SESSION_ROTATION_WARNING_BUFFER_SECONDS = 10
    DEFAULT_SESSION_ROTATION_SECONDS = max(
        1,
        int(RecordingStateManager.RECORDING_EXPIRY_WARNING)
        - SESSION_ROTATION_WARNING_BUFFER_SECONDS,
    )
except Exception:  # noqa: BLE001
    DEFAULT_BACKEND_RECORDING_TTL_SECONDS = 300
    SESSION_ROTATION_WARNING_BUFFER_SECONDS = 10
    DEFAULT_SESSION_ROTATION_SECONDS = 260

# Cap replay throughput to avoid overwhelming local daemon buffers.
# Keep this reasonably high so imports remain practical for image-heavy logs.
DEFAULT_REPLAY_MAX_BYTES_PER_SECOND = 256 * 1024 * 1024
# Emit MCAP progress updates continuously by default.
DEFAULT_PROGRESS_EMIT_INTERVAL = 1


SkipOnError = Literal["all", "episode", "step"]


@dataclass(frozen=True, slots=True)
class MCAPImportConfig:
    """Configuration for MCAP import behavior.

    TTL defaults are sourced from Neuracore's recording state manager constants
    when available and rotation defaults to warning threshold minus a 10-second
    buffer. The importer still only reads one MCAP-specific environment
    variables: `NEURACORE_MCAP_STAGE_DIR` and
    `NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL`.
    """

    BACKEND_RECORDING_TTL_SECONDS: int = DEFAULT_BACKEND_RECORDING_TTL_SECONDS
    SESSION_ROTATION_SECONDS: int = DEFAULT_SESSION_ROTATION_SECONDS
    REPLAY_MAX_BYTES_PER_SECOND: int = DEFAULT_REPLAY_MAX_BYTES_PER_SECOND
    progress_emit_interval: int = DEFAULT_PROGRESS_EMIT_INTERVAL
    stage_dir: Path | None = None
    skip_on_error: SkipOnError = "episode"
    enable_decoder_discovery: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        skip_on_error: SkipOnError = "episode",
        enable_decoder_discovery: bool = False,
    ) -> MCAPImportConfig:
        """Build config from environment.

        Reads `NEURACORE_MCAP_STAGE_DIR`
        and `NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL`.
        """
        stage_dir_raw = os.getenv("NEURACORE_MCAP_STAGE_DIR", "").strip()
        stage_dir = Path(stage_dir_raw).expanduser() if stage_dir_raw else None
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)

        progress_emit_interval_raw = os.getenv(
            "NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL",
            str(DEFAULT_PROGRESS_EMIT_INTERVAL),
        ).strip()
        try:
            progress_emit_interval = int(progress_emit_interval_raw)
        except ValueError as exc:
            raise ValueError(
                "NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL must be a positive integer"
            ) from exc
        if progress_emit_interval <= 0:
            raise ValueError(
                "NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL must be a positive integer"
            )

        if skip_on_error not in {"all", "episode", "step"}:
            raise ValueError("skip_on_error must be one of: 'all', 'episode', 'step'")

        return cls(
            progress_emit_interval=progress_emit_interval,
            stage_dir=stage_dir,
            skip_on_error=skip_on_error,
            enable_decoder_discovery=enable_decoder_discovery,
        )
