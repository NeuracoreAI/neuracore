"""Runner entrypoint for the Neuracore data daemon."""

from __future__ import annotations

import atexit
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import (
    get_daemon_db_path,
    get_daemon_pid_path,
    is_debug_mode,
)
from neuracore.data_daemon.lifecycle.daemon_os_control import install_signal_handlers
from neuracore.data_daemon.lifecycle.runtime_recovery import shutdown
from neuracore.data_daemon.runtime import DaemonContext, DaemonRuntime

logger = logging.getLogger(__name__)


def _configure_root_logging() -> None:
    """Configure root logging to the daemon log file when one is requested.

    Without this, a background-spawned daemon's log lines (SSL/auth/upload
    retries, org mismatches) end up in the void — only the launching parent's
    stderr capture window would see anything, and that closes seconds after
    startup. We route everything to a rotated file under ~/.neuracore/logs/.
    """
    log_path_env = os.environ.get("NEURACORE_DAEMON_LOG_PATH")
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = logging.INFO

    if not log_path_env:
        logging.basicConfig(level=level, format=fmt)
        return

    try:
        log_path = Path(log_path_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger()
        root.setLevel(level)
        # Avoid duplicate stream handlers if the runtime imports re-configure.
        root.handlers = [file_handler]
        # Mirror to stderr (which the parent captured during startup poll).
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(stream_handler)
    except OSError:
        logging.basicConfig(level=level, format=fmt)
        logging.getLogger(__name__).warning(
            "Could not open daemon log file %s; falling back to stderr-only",
            log_path_env,
        )


def main() -> None:
    """Runner entrypoint for the Neuracore data daemon.

    This function initializes the daemon runtime, starts it, and then waits for
    a signal to stop. The daemon is stopped when the function returns.

    Environment variables affecting this function:

    NEURACORE_DAEMON_PID_PATH
        Path to the pid file for the daemon.

    NEURACORE_DAEMON_DB_PATH
        Path to the SQLite database file for the daemon's state.

    The daemon will exit with a status code of 1 if the socket at
    NEURACORE_DAEMON_SOCKET_PATH already exists.

    The daemon will shut down when it receives a SIGINT or SIGTERM signal.
    """
    debug_mode = is_debug_mode()
    profiler = None
    if debug_mode:
        import pyinstrument

        profiler = pyinstrument.Profiler()
    if profiler:
        profiler.start()
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    runtime = DaemonRuntime(
        db_path=db_path,
        pid_path=pid_path,
        socket_paths=(SOCKET_PATH,),
    )
    cleaned_up = False

    def shutdown_runtime() -> None:
        """Run the standard daemon shutdown path at most once."""
        nonlocal cleaned_up
        if cleaned_up:
            return
        runtime.shutdown()
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH,),
            db_path=db_path,
        )
        cleaned_up = True

    try:
        # Make SIGTERM raise KeyboardInterrupt
        install_signal_handlers()

        context = runtime.initialize()

        if not isinstance(context, DaemonContext):
            logger.error("Failed to start daemon")
            return

        def on_exit() -> None:
            """Inform user of daemon exit event."""
            runtime.shutdown()

        atexit.register(on_exit)
        logger.info("Daemon starting main loop...")
        try:
            runtime.run_forever()
        except Exception:
            logger.exception("Fatal error while daemon main loop was running")
            shutdown_runtime()
            raise

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except SystemExit:
        pass
    finally:
        shutdown_runtime()
        if profiler:
            profiler.stop()
            profiler.write_html("profile-daemon-main.html")
        print("Daemon stopped.")


if __name__ == "__main__":
    _configure_root_logging()
    main()
