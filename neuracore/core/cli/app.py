"""Neuracore CLI entry point."""

import typer

from neuracore import __version__
from neuracore.core.cli.generate_api_key import run as login
from neuracore.core.cli.launch_server import run as launch_server
from neuracore.core.cli.select_current_org import run as select_org

app = typer.Typer(add_completion=False, help="Neuracore command line interface.")


def _version_callback(value: bool) -> bool:
    if value:
        typer.echo(__version__)
        raise typer.Exit()
    return value


@app.callback()
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the neuracore version and exit.",
        callback=_version_callback,
        is_eager=True,
        is_flag=True,
    ),
) -> None:
    """Handle global CLI option for --version."""
    return None


app.command("login")(login)
app.command("select-org")(select_org)
app.command("launch-server")(launch_server)


def main() -> None:
    """CLI entrypoint for the neuracore command."""
    app()


if __name__ == "__main__":
    main()
