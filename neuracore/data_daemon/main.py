"""Main entry point for the Neuracore data daemon CLI."""

from __future__ import annotations

import typer

from neuracore.data_daemon.config_manager.args_handler import (
    profile_app,
    run_install,
    run_launch,
    run_status,
    run_stop,
    run_uninstall,
)

app = typer.Typer(
    add_completion=False,
    help="Neuracore Data Daemon CLI.",
)

app.command("launch")(run_launch)
app.command("stop")(run_stop)
app.command("status")(run_status)
app.command("install")(run_install)
app.command("uninstall")(run_uninstall)

app.add_typer(profile_app, name="profile")


def main() -> None:
    """CLI entrypoint for neuracore data-daemon."""
    app()


if __name__ == "__main__":
    main()
