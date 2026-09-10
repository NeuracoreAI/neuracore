"""Who opens and closes a recording window, behind one interface.

A case's producer decides what is logged; this decides who brackets it. Both
report on the same wall clock, so the boundary classifier reads one vocabulary
whichever combination ran.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_session import thread_local_session
from neuracore.data_daemon.bridge import RecordingStateUnavailableError
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CONTROL_REMOTE,
    MAX_TIME_TO_START_S,
    REMOTE_CONTROL_REQUEST_TIMEOUT_S,
    REMOTE_GATE_POLL_INTERVAL_S,
    REMOTE_START_ANNOUNCEMENT_SLA_S,
    REMOTE_STOP_PROPAGATION_SLA_S,
    STOP_PUBLISH_SKEW_S,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextSpec,
)


@dataclass(frozen=True, slots=True)
class ControlBracket:
    """One control call, bracketed on the wall clock.

    Attributes:
        called_at: Before the call — no frame completed before this is inside.
        settled_at: When the window is known to have opened or closed here. A
            local call stamps the bound itself, so it settles a bounded skew
            later; a remote one is a network round trip and a notification
            away, and is waited for.
        handle: The recording this process's gate held once the call settled.
    """

    called_at: float
    settled_at: float
    handle: str | None


class RecordingController(ABC):
    """Opens and closes one context's recordings, however they are triggered."""

    def __init__(self, spec: ContextSpec, robot: object) -> None:
        self.spec = spec
        self.robot = robot

    @abstractmethod
    def open(self, capture_start_s: float) -> ControlBracket:
        """Open a recording window and return once it is known to be open."""

    @abstractmethod
    def close(self, capture_stop_s: float) -> ControlBracket:
        """Close the current window and return once it is known to be closed."""


class LocalRecordingController(RecordingController):
    """The SDK's own calls, made by the process the test runs in."""

    def open(self, capture_start_s: float) -> ControlBracket:
        """Start the recording; the local gate is open by the time this returns."""
        called_at = time.time()
        with Timer(
            MAX_TIME_TO_START_S,
            label="nc.start_recording",
            always_log=True,
            assert_deadline=self.spec.assert_deadline,
        ):
            nc.start_recording(
                robot_name=self.spec.robot_name, timestamp=capture_start_s
            )
        return ControlBracket(
            called_at=called_at,
            settled_at=time.time(),
            handle=self.robot.get_current_recording_id(),
        )

    def close(self, capture_stop_s: float) -> ControlBracket:
        """Stop the recording; the window's bound is stamped inside the call.

        The daemon closes the window at the publish stamp of the envelope this
        call sends, which goes out before the flush ``wait`` waits on — so the
        bound has passed by :data:`STOP_PUBLISH_SKEW_S` after entry, whatever
        the call itself then spends.
        """
        called_at = time.time()
        handle = self.robot.get_current_recording_id()
        with Timer(
            self.spec.case.stop_recording_sla_s,
            label="nc.stop_recording",
            always_log=True,
            assert_deadline=self.spec.assert_deadline,
        ):
            nc.stop_recording(
                robot_name=self.spec.robot_name,
                wait=self.spec.case.wait,
                timestamp=capture_stop_s,
            )
        return ControlBracket(
            called_at=called_at,
            settled_at=called_at + STOP_PUBLISH_SKEW_S,
            handle=handle,
        )


class RemoteRecordingController(RecordingController):
    """The backend's own endpoints, called the way the web frontend calls them.

    No process here opens the window: the backend mints the recording and every
    connected process learns about it over the notification stream. Both
    brackets are therefore waited out, the start against the time the cloud
    holds for the recording (see :data:`REMOTE_START_ANNOUNCEMENT_SLA_S`).
    """

    def __init__(self, spec: ContextSpec, robot: object) -> None:
        super().__init__(spec, robot)
        self._dataset_id = nc.get_dataset(spec.dataset_name).id
        self._org_id = get_current_org()
        self._cloud_recording_id: str | None = None

    def _post(self, path: str, payload: dict) -> dict:
        """POST to this org's API, as an authenticated web client would.

        Deliberately without the transient-status retry the SDK uses for reads:
        every accepted ``/recording/start`` mints a *fresh* recording, so a
        retry after the write committed would leave two windows behind.
        """
        session = thread_local_session()
        response = session.post(
            f"{API_URL}/org/{self._org_id}{path}",
            json=payload,
            headers=get_auth().get_headers(),
            timeout=REMOTE_CONTROL_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()

    def open(self, capture_start_s: float) -> ControlBracket:
        """Ask the backend to start a recording, then wait to be told it did."""
        called_at = time.time()
        with Timer(
            MAX_TIME_TO_START_S,
            label="remote.start_recording",
            always_log=True,
            assert_deadline=self.spec.assert_deadline,
        ):
            pending = self._post(
                "/recording/start",
                {
                    "robot_id": str(self.robot.id),
                    "instance": int(self.robot.instance),
                    "dataset_id": self._dataset_id,
                    "start_time": capture_start_s,
                },
            )
        self._cloud_recording_id = str(pending["id"])
        announced_start_s = float(pending["start_time"])
        settled_at = self._await_gate(
            open_gate=True,
            deadline=time.time() + REMOTE_CONTROL_REQUEST_TIMEOUT_S,
            label="start",
            overdue=(
                f"the start of recording {self._cloud_recording_id} never reached "
                f"this process: nothing arrived in "
                f"{REMOTE_CONTROL_REQUEST_TIMEOUT_S}s"
            ),
        )
        window_start_lag_s = settled_at - announced_start_s
        assert window_start_lag_s <= REMOTE_START_ANNOUNCEMENT_SLA_S, (
            f"recording {self._cloud_recording_id} reached this process "
            f"{window_start_lag_s:.3f}s after the start time the cloud holds "
            f"for it, over the {REMOTE_START_ANNOUNCEMENT_SLA_S}s allowed"
        )
        return ControlBracket(
            called_at=called_at,
            settled_at=settled_at,
            handle=self.robot.get_current_recording_id(),
        )

    def close(self, capture_stop_s: float) -> ControlBracket:
        """Ask the backend to stop the recording, then wait to be told it did.

        No call here stamps the bound, so the gate transition is the only thing
        that settles a remote stop and is always waited for.
        """
        assert self._cloud_recording_id is not None, "close() before open()"
        called_at = time.time()
        handle = self.robot.get_current_recording_id()
        with Timer(
            MAX_TIME_TO_START_S,
            label="remote.stop_recording",
            always_log=True,
            assert_deadline=self.spec.assert_deadline,
        ):
            self._post(
                "/recording/stop",
                {
                    "recording_id": self._cloud_recording_id,
                    "end_time": capture_stop_s,
                },
            )
        settled_at = self._await_gate(
            open_gate=False,
            deadline=time.time() + REMOTE_STOP_PROPAGATION_SLA_S,
            label="stop",
            overdue=(
                f"the stop of recording {self._cloud_recording_id} did not reach "
                f"this process within {REMOTE_STOP_PROPAGATION_SLA_S}s"
            ),
        )
        self._cloud_recording_id = None
        return ControlBracket(called_at=called_at, settled_at=settled_at, handle=handle)

    def _await_gate(
        self, *, open_gate: bool, deadline: float, label: str, overdue: str
    ) -> float:
        """Wait for this process's gate to reach the expected state by *deadline*.

        The gate flips when the SDK's notification consumer applies the event,
        so this returns when the announcement became visible to a client. How
        late is late is the caller's judgement.
        """
        with Timer(
            max(deadline - time.time(), 0.0),
            label=f"remote.{label}_gate_wait",
            always_log=True,
            assert_deadline=self.spec.assert_deadline,
        ):
            while time.time() < deadline:
                try:
                    if (self.robot.get_current_recording_id() is not None) == open_gate:
                        return time.time()
                except RecordingStateUnavailableError:
                    # A busy daemon answers nothing rather than "not
                    # recording"; the gate has not moved, so poll again.
                    pass
                time.sleep(REMOTE_GATE_POLL_INTERVAL_S)
            raise AssertionError(overdue)


def make_recording_controller(
    spec: ContextSpec, *, robot: object
) -> RecordingController:
    """Build the controller a case asked for."""
    if spec.case.recording_control == CONTROL_REMOTE:
        return RemoteRecordingController(spec, robot)
    return LocalRecordingController(spec, robot)
