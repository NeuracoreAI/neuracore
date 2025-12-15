"""This module provides a global enable manager for live data.

This streamlines disabling all live data producers useful during testing and
validation.
"""

from neuracore.core.const import CONSUME_LIVE_DATA, PROVIDE_LIVE_DATA
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager

_global_provide_live_data_manager: EnabledManager | None = None


def get_provide_live_data_enabled_manager() -> EnabledManager:
    """Get the global enabled manager for providing live data.

    Returns:
        EnabledManager: The global enabled manager for providing live data.
    """
    global _global_provide_live_data_manager
    if not _global_provide_live_data_manager:
        _global_provide_live_data_manager = EnabledManager(
            PROVIDE_LIVE_DATA, loop=get_running_loop()
        )
    return _global_provide_live_data_manager


_global_consume_live_data_manager: EnabledManager | None = None


def get_consume_live_data_enabled_manager() -> EnabledManager:
    """Get the global enabled manager for consuming live data.

    Returns:
        EnabledManager: The global enabled manager for consuming live data.
    """
    global _global_consume_live_data_manager
    if not _global_consume_live_data_manager:
        _global_consume_live_data_manager = EnabledManager(
            CONSUME_LIVE_DATA, loop=get_running_loop()
        )
    return _global_consume_live_data_manager
