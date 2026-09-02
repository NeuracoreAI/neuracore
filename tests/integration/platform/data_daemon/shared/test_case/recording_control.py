"""Who opens and closes a recording window, behind one interface.

A case's producer decides what is logged; this decides who brackets it. Both
report on the same wall clock, so the boundary classifier reads one vocabulary
whichever combination ran — including across processes, where a controller may
make its calls from a peer.
"""

from __future__ import annotations

import queue
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_session import thread_local_session
from neuracore.data_daemon.bridge import RecordingStateUnavailableError
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.child_process import (
    ChildProcess,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CONTROL_REMOTE,
    CONTROL_SPLIT_PROCESS,
    MAX_TIME_TO_START_S,
    REMOTE_CONTROL_REQUEST_TIMEOUT_S,
    REMOTE_GATE_POLL_INTERVAL_S,
    REMOTE_START_ANNOUNCEMENT_SLA_S,
    REMOTE_STOP_PROPAGATION_SLA_S,
    SPLIT_CONTROL_ACK_POLL_S,
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


def await_gate(
    robot: object,
    *,
    open_gate: bool,
    deadline: float,
    label: str,
    overdue: str,
    assert_deadline: bool,
) -> float:
    """Wait for *robot*'s gate in this process to reach *open_gate* by *deadline*.

    The gate flips when the SDK's notification consumer applies the event, so
    this returns when the announcement became visible to a client. The deadline
    only bounds the blocking; how late is late is the caller's judgement, since
    only it knows what the wait should be measured from.
    """
    with Timer(
        max(deadline - time.time(), 0.0),
        label=label,
        always_log=True,
        assert_deadline=assert_deadline,
    ):
        while time.time() < deadline:
            try:
                handle = robot.get_current_recording_id()  # type: ignore[attr-defined]
            except RecordingStateUnavailableError:
                # A busy daemon answers nothing rather than "not recording";
                # the gate has not moved, so poll again.
                time.sleep(REMOTE_GATE_POLL_INTERVAL_S)
                continue
            if (handle is not None) == open_gate:
                return time.time()
            time.sleep(REMOTE_GATE_POLL_INTERVAL_S)
        raise AssertionError(overdue)


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

    def cancel(self, capture_stop_s: float) -> ControlBracket:
        """Discard the current window and return once it is known to be gone.

        Not abstract: a case that never cancels should not have to carry an
        implementation for it.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot cancel a recording")

    def shutdown(self) -> None:
        """Release whatever this controller holds past the last recording."""


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

    def cancel(self, capture_stop_s: float) -> ControlBracket:
        """Discard the recording with the SDK's own call, from this process."""
        called_at = time.time()
        handle = self.robot.get_current_recording_id()
        with Timer(
            self.spec.case.stop_recording_sla_s,
            label="nc.cancel_recording",
            always_log=True,
            assert_deadline=self.spec.assert_deadline,
        ):
            nc.cancel_recording(
                robot_name=self.spec.robot_name, timestamp=capture_stop_s
            )
        return ControlBracket(
            called_at=called_at, settled_at=time.time(), handle=handle
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
        settled_at = await_gate(
            self.robot,
            open_gate=True,
            deadline=time.time() + REMOTE_CONTROL_REQUEST_TIMEOUT_S,
            label="remote.start_gate_wait",
            assert_deadline=self.spec.assert_deadline,
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
        settled_at = await_gate(
            self.robot,
            open_gate=False,
            deadline=time.time() + REMOTE_STOP_PROPAGATION_SLA_S,
            label="remote.stop_gate_wait",
            assert_deadline=self.spec.assert_deadline,
            overdue=(
                f"the stop of recording {self._cloud_recording_id} did not reach "
                f"this process within {REMOTE_STOP_PROPAGATION_SLA_S}s"
            ),
        )
        self._cloud_recording_id = None
        return ControlBracket(called_at=called_at, settled_at=settled_at, handle=handle)


_PEER_AWAIT_OPEN = "await-open"
_PEER_STOP = "stop"
_PEER_CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class ControlProcessSpec:
    """What the stopping peer needs, in picklable form — a controller holds a
    live robot handle and a spec full of them, neither picklable."""

    robot_name: str
    wait: bool
    stop_sla_s: float
    assert_deadline: bool


@dataclass(frozen=True, slots=True)
class _OpenAck:
    """When the peer learned a window it did not open had opened."""

    gate_opened_at: float


@dataclass(frozen=True, slots=True)
class _StopAck:
    """When the peer's stop call returned, so the wait for that stop to come
    back round to this process is measured from the right instant."""

    returned_at: float


@dataclass(frozen=True, slots=True)
class _CancelAck:
    """When the peer's cancel call returned. Separate from :class:`_StopAck` so
    an answer out of step is caught by kind, not read as the other call."""

    returned_at: float


def _peer_control_process(
    spec: ControlProcessSpec,
    ready_event: Any,
    commands: Any,
    acks: Any,
    result_queue: Any,
) -> None:
    """End recordings this process never started, in its own OS process.

    Logs nothing and starts nothing. It connects — which subscribes it to the
    notification stream — waits to be told a window opened, and ends it with
    the SDK's own call: a stop that keeps the data, or a cancel that discards
    it. Every order is answered, so a caller blocked on one learns of a failure
    from the reaped traceback rather than a timeout.
    """
    robot = None
    try:
        ensure_login()
        robot = nc.connect_robot(spec.robot_name, overwrite=False)
        ready_event.set()
        while True:
            command = commands.get()
            if command is None:
                break
            order, timestamp = command
            if order == _PEER_AWAIT_OPEN:
                acks.put(_await_peer_open(spec, robot, announced_start_s=timestamp))
            elif order == _PEER_CANCEL:
                acks.put(_cancel_from_peer(spec, robot, capture_stop_s=timestamp))
            else:
                acks.put(_stop_from_peer(spec, robot, capture_stop_s=timestamp))
        result_queue.put(
            {
                "ok": True,
                "timer_stats": {label: dict(v) for label, v in Timer._stats.items()},
            }
        )
    except BaseException:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "traceback": traceback.format_exc(),
                "timer_stats": {label: dict(v) for label, v in Timer._stats.items()},
            }
        )
    finally:
        if robot is not None:
            robot.close()


def _await_peer_open(
    spec: ControlProcessSpec, robot: object, *, announced_start_s: float
) -> _OpenAck:
    """Wait until the announcement of a window opened elsewhere lands here.

    ``nc.stop_recording`` refuses — silently — in a process whose gate never
    opened, so this wait is what lets the peer close anything at all. It also
    brings the peer's daemon channel up, the notification handler publishing
    the start.
    """
    return _OpenAck(
        gate_opened_at=await_gate(
            robot,
            open_gate=True,
            deadline=time.time() + REMOTE_CONTROL_REQUEST_TIMEOUT_S,
            label="peer.start_gate_wait",
            assert_deadline=spec.assert_deadline,
            overdue=(
                f"a recording started at {announced_start_s} never reached the "
                f"stopping peer: nothing arrived in "
                f"{REMOTE_CONTROL_REQUEST_TIMEOUT_S}s"
            ),
        )
    )


def _cancel_from_peer(
    spec: ControlProcessSpec, robot: object, *, capture_stop_s: float
) -> _CancelAck:
    """Make the SDK's own cancel call for a window this process never opened.

    The discarding counterpart of :func:`_stop_from_peer`: it has to reach every
    trace of a recording this process never wrote a byte of, and drop them.
    """
    with Timer(
        spec.stop_sla_s,
        label="peer.cancel_recording",
        always_log=True,
        assert_deadline=spec.assert_deadline,
    ):
        nc.cancel_recording(robot_name=spec.robot_name, timestamp=capture_stop_s)
    assert robot.get_current_recording_id() is None, (  # type: ignore[attr-defined]
        "the cancelling peer's nc.cancel_recording left its gate open, so the "
        "call was refused before it reached the daemon at all"
    )
    return _CancelAck(returned_at=time.time())


def _stop_from_peer(
    spec: ControlProcessSpec, robot: object, *, capture_stop_s: float
) -> _StopAck:
    """Make the SDK's own stop call for a window this process never opened."""
    with Timer(
        spec.stop_sla_s,
        label="peer.stop_recording",
        always_log=True,
        assert_deadline=spec.assert_deadline,
    ):
        nc.stop_recording(
            robot_name=spec.robot_name, wait=spec.wait, timestamp=capture_stop_s
        )
    assert robot.get_current_recording_id() is None, (  # type: ignore[attr-defined]
        "the stopping peer's nc.stop_recording left its gate open, so the call "
        "was refused before it reached the daemon at all"
    )
    return _StopAck(returned_at=time.time())


class SplitProcessRecordingController(RecordingController):
    """This process starts every window; a peer process makes the stop call.

    An application opening a recording and something else — an operator tool, a
    supervisor, the next node along — closing it. Nothing routes the stop back
    by hand: the peer learns the window exists because the backend announced it,
    and this process learns it closed the same way, so both directions of the
    notification stream have to work for one recording to end.
    """

    def __init__(self, spec: ContextSpec, robot: object) -> None:
        super().__init__(spec, robot)
        self._peer = ChildProcess(f"peer-control-{spec.context_index}")
        self._commands = self._peer.queue()
        self._acks = self._peer.queue()
        self._peer_retired = False
        self._window_start_s = 0.0
        self._peer.start(
            _peer_control_process,
            (
                ControlProcessSpec(
                    robot_name=spec.robot_name,
                    wait=spec.case.wait,
                    stop_sla_s=spec.case.stop_recording_sla_s,
                    assert_deadline=spec.assert_deadline,
                ),
                self._peer.ready_event,
                self._commands,
                self._acks,
                self._peer.result_queue,
            ),
        )
        if not self._peer.await_ready():
            failure = self._peer.collect().failure or (
                f"still connecting after {MAX_TIME_TO_START_S}s"
            )
            raise RuntimeError(f"the stopping peer never connected: {failure}")

    def open(self, capture_start_s: float) -> ControlBracket:
        """Start the recording here, and set the peer waiting to hear of it.

        The peer's wait is not collected until :meth:`close`: this process's
        gate is open the moment the call returns, so blocking on a second
        process learning that would only postpone frames already inside.
        """
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
        self._window_start_s = capture_start_s
        self._commands.put((_PEER_AWAIT_OPEN, capture_start_s))
        return ControlBracket(
            called_at=called_at,
            settled_at=time.time(),
            handle=self.robot.get_current_recording_id(),
        )

    def close(self, capture_stop_s: float) -> ControlBracket:
        """Have the peer stop the recording, then wait to be told it did.

        This process makes no stop call to stamp the bound, so the gate
        transition is the only thing that settles the close and is always
        waited for.
        """
        self._await_peer_heard_of_the_window()
        called_at = time.time()
        handle = self.robot.get_current_recording_id()
        self._commands.put((_PEER_STOP, capture_stop_s))
        stopped: _StopAck = self._await_ack(
            _PEER_STOP,
            _StopAck,
            deadline=time.time()
            + self.spec.case.stop_recording_sla_s
            + REMOTE_CONTROL_REQUEST_TIMEOUT_S,
        )
        settled_at = await_gate(
            self.robot,
            open_gate=False,
            deadline=stopped.returned_at + REMOTE_STOP_PROPAGATION_SLA_S,
            label="split.stop_gate_wait",
            assert_deadline=self.spec.assert_deadline,
            overdue=(
                "the peer's stop never came back round to this process: its own "
                f"gate was still open {REMOTE_STOP_PROPAGATION_SLA_S}s after the "
                "peer's call returned, so its streams were never drained. A "
                "peer stop whose daemon publish failed reads exactly like this "
                "— Robot._drain_streams_and_notify_daemon logs it and returns, "
                "so check the peer's stderr for a failed publish"
            ),
        )
        return ControlBracket(called_at=called_at, settled_at=settled_at, handle=handle)

    def cancel(self, capture_stop_s: float) -> ControlBracket:
        """Have the peer cancel the recording, then wait to be told it did.

        The same split as :meth:`close`, discarding instead of keeping: the
        peer must still have heard the window open before it can name anything
        to cancel.
        """
        self._await_peer_heard_of_the_window()
        called_at = time.time()
        handle = self.robot.get_current_recording_id()
        self._commands.put((_PEER_CANCEL, capture_stop_s))
        cancelled: _CancelAck = self._await_ack(
            _PEER_CANCEL,
            _CancelAck,
            deadline=time.time()
            + self.spec.case.stop_recording_sla_s
            + REMOTE_CONTROL_REQUEST_TIMEOUT_S,
        )
        settled_at = await_gate(
            self.robot,
            open_gate=False,
            deadline=cancelled.returned_at + REMOTE_STOP_PROPAGATION_SLA_S,
            label="split.cancel_gate_wait",
            assert_deadline=self.spec.assert_deadline,
            overdue=(
                "the peer's cancel never came back round to this process: its "
                f"own gate was still open {REMOTE_STOP_PROPAGATION_SLA_S}s "
                "after the peer's call returned"
            ),
        )
        return ControlBracket(called_at=called_at, settled_at=settled_at, handle=handle)

    def _await_peer_heard_of_the_window(self) -> None:
        """Block until the peer has been told the window this process opened.

        Nothing the peer can do names a recording until this lands, so both
        ways of ending one wait on it, and both assert the same SLA.
        """
        opened: _OpenAck = self._await_ack(
            _PEER_AWAIT_OPEN,
            _OpenAck,
            deadline=time.time() + REMOTE_CONTROL_REQUEST_TIMEOUT_S,
        )
        peer_start_lag_s = opened.gate_opened_at - self._window_start_s
        assert peer_start_lag_s <= REMOTE_START_ANNOUNCEMENT_SLA_S, (
            f"the window opened at {self._window_start_s} reached the stopping "
            f"peer {peer_start_lag_s:.3f}s later, over the "
            f"{REMOTE_START_ANNOUNCEMENT_SLA_S}s allowed"
        )

    def shutdown(self) -> None:
        """Retire the peer and surface anything it raised."""
        if self._peer_retired:
            return
        self._peer_retired = True
        self._commands.put(None)
        failure = self._peer.collect().failure
        if failure:
            raise RuntimeError(f"the stopping peer failed:\n{failure}")

    def _await_ack(self, order: str, kind: type, *, deadline: float) -> Any:
        """Take the peer's answer to *order*, or say why none came.

        Polls rather than blocking outright so a peer that died mid-order is
        reaped for its own traceback. Both answers arrive on one channel, so
        the kind is checked rather than assumed.
        """
        while time.time() < deadline:
            try:
                ack = self._acks.get(timeout=SPLIT_CONTROL_ACK_POLL_S)
            except queue.Empty:
                if not self._peer.is_alive:
                    break
                continue
            assert isinstance(ack, kind), (
                f"the stopping peer answered {order!r} with {ack!r}, so its "
                "orders and answers have drifted out of step"
            )
            return ack
        failure = self._peer.collect().failure or "it is still running"
        self._peer_retired = True
        raise AssertionError(f"the stopping peer never answered {order!r}:\n{failure}")


def make_recording_controller(
    spec: ContextSpec, *, robot: object
) -> RecordingController:
    """Build the controller a case asked for."""
    if spec.case.recording_control == CONTROL_REMOTE:
        return RemoteRecordingController(spec, robot)
    if spec.case.recording_control == CONTROL_SPLIT_PROCESS:
        return SplitProcessRecordingController(spec, robot)
    return LocalRecordingController(spec, robot)
