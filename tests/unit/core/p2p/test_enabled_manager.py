"""EnabledManager listeners may use the manager that notifies them."""

import asyncio
import threading

from neuracore.core.streaming.p2p.enabled_manager import EnabledManager


def _returns_within(action, seconds: float = 2.0) -> bool:
    done = threading.Event()

    def run() -> None:
        action()
        done.set()

    threading.Thread(target=run, daemon=True).start()
    return done.wait(seconds)


def test_a_listener_can_disable_and_read_its_manager() -> None:
    loop = asyncio.new_event_loop()
    manager = EnabledManager(True, loop=loop)
    seen: list[bool] = []

    def on_disabled() -> None:
        seen.append(manager.is_enabled())
        manager.disable()

    manager.add_listener(EnabledManager.DISABLED, on_disabled)

    assert _returns_within(manager.disable)
    assert seen == [False]
    loop.close()
