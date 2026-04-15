"""Runner entrypoint for the Neuracore data daemon."""

from __future__ import annotations

import atexit
import logging

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

    try:
        install_signal_handlers(lambda _signum: None)
        context = runtime.initialize()

        if not isinstance(context, DaemonContext):
            logger.error("Failed to start daemon")
            return

        def on_exit() -> None:
            """Inform user of daemon exit event."""
            runtime.shutdown()

        atexit.register(on_exit)
        logger.info("Daemon starting main loop...")
        runtime.run_forever()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except SystemExit:
        pass
    finally:
        runtime.shutdown()
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH,),
            db_path=db_path,
        )
        if profiler:
            profiler.stop()
            profiler.write_html("profile-daemon-main.html")
        print("Daemon stopped.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
