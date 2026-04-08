"""Shared fixtures for data_daemon tests."""

# cspell:ignore getfixturevalue
from collections.abc import Generator

import pytest
from pytest import FixtureRequest
from pytest_asyncio.plugin import Runner

from neuracore.data_daemon.event_emitter import Emitter, init_emitter


@pytest.fixture(scope="function", autouse=True)
def emitter(request: FixtureRequest) -> Generator[Emitter, None, None]:
    """Create a fresh Emitter bound to pytest-asyncio's function runner loop."""
    runner = request.getfixturevalue("_function_scoped_runner")
    assert isinstance(runner, Runner)
    test_emitter = init_emitter(loop=runner.get_loop())
    yield test_emitter
    test_emitter.remove_all_listeners()
