"""Shared Typer option annotations for the data daemon CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from neuracore.data_daemon.config_manager.helpers import parse_bytes

ProfileNameCreateArgument = Annotated[
    str,
    typer.Argument(help="Profile name."),
]

ProfileNameUpdateArgument = Annotated[
    str | None,
    typer.Argument(help="Profile name to update."),
]

ProfileNameGetArgument = Annotated[
    str | None,
    typer.Argument(help="Profile name to get."),
]


ProfileNameDeleteArgument = Annotated[
    str,
    typer.Argument(help="Profile name to delete."),
]

StorageLimitOption = Annotated[
    int | None,
    typer.Option(
        "--storage-limit",
        "--storage_limit",
        parser=parse_bytes,
        help="Storage limit in bytes.",
    ),
]

BandwidthLimitOption = Annotated[
    int | None,
    typer.Option(
        "--bandwidth-limit",
        "--bandwidth_limit",
        parser=parse_bytes,
        help="Bandwidth limit in bytes per second.",
    ),
]

StoragePathOption = Annotated[
    str | None,
    typer.Option(
        "--storage-path",
        "--storage_path",
        "--path_to_store_record",
        help="Path where records should be stored.",
    ),
]

NumThreadsOption = Annotated[
    int | None,
    typer.Option(
        "--num-threads",
        "--num_threads",
        help="Number of worker threads.",
    ),
]

MaxConcurrentUploadsOption = Annotated[
    int | None,
    typer.Option(
        "--max-concurrent-uploads",
        "--max_concurrent_uploads",
        help="Maximum number of traces uploaded concurrently.",
    ),
]

KeepWakelockOption = Annotated[
    bool | None,
    typer.Option(
        "--wakelock/--no-wakelock",
        help="Keep a wakelock while uploading.",
    ),
]

OfflineOption = Annotated[
    bool | None,
    typer.Option(
        "--offline/--online",
        help="Run in offline mode.",
    ),
]

ApiKeyOption = Annotated[
    str | None,
    typer.Option(
        "--api-key",
        "--api_key",
        help="API key used for authenticating the daemon.",
    ),
]

CurrentOrgIdOption = Annotated[
    str | None,
    typer.Option(
        "--current-org-id",
        "--current_org_id",
        help="Active organisation ID for scoping daemon operations.",
    ),
]

LaunchProfileOption = Annotated[
    str | None,
    typer.Option(
        "--profile",
        help="Profile name to launch (from ~/.neuracore/data_daemon/profiles).",
    ),
]

BackgroundOption = Annotated[
    bool,
    typer.Option(
        "--background",
        help="Run the daemon in the background without terminal output.",
        is_flag=True,
    ),
]
