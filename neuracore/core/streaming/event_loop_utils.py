import asyncio
import threading


def get_running_loop():
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()
        return loop
