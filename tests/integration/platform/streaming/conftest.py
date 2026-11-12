"""Fixtures of the streaming integration tests."""

import multiprocessing
import os
import time
import uuid
from collections.abc import Generator

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    delete_cloud_robot,
)
from tests.integration.platform.streaming.nodes import (
    REMOTE_JOINT_NAME,
    SIGNALLING_URL_VARIABLE,
    Provider,
    ProviderFactory,
    regional_node,
)

READY_POLL_SECONDS = 1


@pytest.fixture(scope="module")
def regions() -> list[str]:
    """Canonical URLs of the signalling regions of the target."""
    assert (
        SIGNALLING_URL_VARIABLE not in os.environ
    ), f"{SIGNALLING_URL_VARIABLE} must be unset: the tests choose the region."
    from neuracore.core.config.get_current_org import get_current_org
    from tests.integration.platform.streaming.consumer import region_endpoints

    nc.login()
    endpoints = region_endpoints(get_current_org())
    assert endpoints, "The target has no signalling regions: regional mode is off."
    return endpoints


@pytest.fixture
def robot_name() -> Generator[str, None, None]:
    """A unique robot name, deleted after the test."""
    name = f"streaming-{uuid.uuid4().hex[:8]}"
    yield name
    delete_cloud_robot(name)


@pytest.fixture
def start_provider() -> Generator[ProviderFactory, None, None]:
    """Start provider nodes, and stop them after the test."""
    from tests.integration.platform.streaming.consumer import WAIT_SECONDS

    context = multiprocessing.get_context("spawn")
    providers: list[Provider] = []

    def start(
        robot_name: str,
        instance: int,
        signalling_url: str | None,
        joint_name: str = REMOTE_JOINT_NAME,
    ) -> Provider:
        ready_event = context.Event()
        stop_event = context.Event()
        process = context.Process(
            target=regional_node,
            args=(
                robot_name,
                instance,
                signalling_url,
                ready_event,
                stop_event,
                joint_name,
            ),
        )
        process.start()
        provider = Provider(process=process, stop_event=stop_event)
        providers.append(provider)
        deadline = time.monotonic() + WAIT_SECONDS
        while not ready_event.wait(READY_POLL_SECONDS):
            assert (
                process.is_alive()
            ), f"Provider node exited with code {process.exitcode} before it was ready."
            assert time.monotonic() < deadline, "Provider node was not ready in time."
        return provider

    yield start
    for provider in providers:
        provider.stop()
