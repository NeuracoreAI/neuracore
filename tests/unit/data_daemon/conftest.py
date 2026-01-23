"""Shared fixtures for data_daemon tests."""

import asyncio

import pytest_asyncio

import neuracore.data_daemon.event_emitter as em_module
from neuracore.data_daemon.event_emitter import init_emitter


@pytest_asyncio.fixture(scope="function", autouse=True)
async def initialize_emitter():
    """Initialize the emitter for each test with the current event loop."""
    loop = asyncio.get_event_loop()

    em_module._emitter = None

    emitter = init_emitter(loop=loop)

    yield emitter

    emitter.remove_all_listeners()

    em_module._emitter = None
