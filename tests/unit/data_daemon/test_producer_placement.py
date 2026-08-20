"""Offline checks on the multi-process producer's placement and dispatch.

All of it is decided before a process is spawned or a robot connected.
"""

from __future__ import annotations

import pytest

from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    PerThread,
    SeparateProcessPerCamera,
    Synchronous,
    case_id,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CAMERA_0,
    CAMERA_1,
    DEPTH_CAMERA_0,
    STREAM_DEPTH,
    STREAM_JOINT_POSITIONS,
    STREAM_JOINT_TORQUES,
    STREAM_JOINT_VELOCITIES,
    STREAM_RGB,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.producers import (
    LifetimeProducerSession,
    MultiProcessProducerSession,
    make_producer_session,
    partition_plans,
)
from tests.integration.platform.data_daemon.shared.test_case.streams import (
    build_stream_plans,
    late_starting_trace_keys,
)

# cspell:ignore jointtorques


def _plans(**overrides):
    kwargs = {
        "joint_names": ["waist", "elbow"],
        "camera_name_list": [CAMERA_0],
        "depth_camera_name_list": [DEPTH_CAMERA_0],
        "depth_mode": "float32",
        "joint_fps": 10,
        "video_fps": 10,
    }
    return build_stream_plans(**{**kwargs, **overrides})


def _session(case, **spec_kwargs):
    spec = build_context_specs(case, dataset_name="unit-test-dataset")[0]
    return make_producer_session(
        spec, robot=object(), marker_name="marker", **spec_kwargs
    )


def test_partition_moves_only_the_named_streams() -> None:
    local, children = partition_plans(_plans(), ((STREAM_RGB,),))

    assert [plan.name for plan in children[0]] == [STREAM_RGB]
    assert [plan.name for plan in local] == [
        STREAM_DEPTH,
        STREAM_JOINT_POSITIONS,
        STREAM_JOINT_VELOCITIES,
        STREAM_JOINT_TORQUES,
    ]


def test_partition_gives_each_entry_its_own_group() -> None:
    local, children = partition_plans(
        _plans(), ((STREAM_RGB,), (STREAM_DEPTH, STREAM_JOINT_TORQUES))
    )

    assert [plan.name for plan in children[0]] == [STREAM_RGB]
    assert [plan.name for plan in children[1]] == [STREAM_DEPTH, STREAM_JOINT_TORQUES]
    assert [plan.name for plan in local] == [
        STREAM_JOINT_POSITIONS,
        STREAM_JOINT_VELOCITIES,
    ]


def test_partition_gives_each_camera_its_own_group_when_named_by_channel() -> None:
    """One camera per process, which naming the kind cannot express."""
    plans = _plans(camera_name_list=[CAMERA_0, CAMERA_1])
    local, children = partition_plans(plans, ((CAMERA_0,), (CAMERA_1,)))

    assert [plan.channel_names for plan in children[0]] == [(CAMERA_0,)]
    assert [plan.channel_names for plan in children[1]] == [(CAMERA_1,)]
    assert [plan.name for plan in local] == [
        STREAM_DEPTH,
        STREAM_JOINT_POSITIONS,
        STREAM_JOINT_VELOCITIES,
        STREAM_JOINT_TORQUES,
    ]


def test_partition_can_move_every_stream_out() -> None:
    """The owner is then only opening and closing the window, which is legal."""
    every = (
        STREAM_RGB,
        STREAM_DEPTH,
        STREAM_JOINT_POSITIONS,
        STREAM_JOINT_VELOCITIES,
        STREAM_JOINT_TORQUES,
    )
    local, children = partition_plans(_plans(), (every,))

    assert local == []
    assert len(children[0]) == len(_plans())


def test_partition_keeps_everything_local_with_no_entries() -> None:
    local, children = partition_plans(_plans(), ())

    assert children == []
    assert len(local) == len(_plans())


def test_placement_is_refused_without_a_multi_process_variant() -> None:
    with pytest.raises(ValueError, match="only applies to"):
        Synchronous(joint_count=2, producer_process_streams=((STREAM_RGB,),))
    with pytest.raises(ValueError, match="only applies to"):
        PerThread(
            joint_count=2, video_count=1, producer_process_streams=((STREAM_RGB,),)
        )


def test_multi_process_needs_a_stream_to_move() -> None:
    """A per-camera case with no cameras leaves nothing for a child to run."""
    with pytest.raises(ValueError, match="at least one stream to move"):
        SeparateProcessPerCamera(joint_count=2, video_count=0)


def test_multi_process_needs_a_single_context() -> None:
    """Pool workers are daemonic, so a context in one cannot spawn a child."""
    with pytest.raises(ValueError, match="parallel_contexts=1"):
        SeparateProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            parallel_contexts=2,
            recording_count=2,
        )


def test_placement_cannot_name_a_stream_the_case_does_not_produce() -> None:
    with pytest.raises(ValueError, match="does not produce"):
        SeparateProcessPerCamera(
            joint_count=0,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_JOINT_POSITIONS,),),
        )


def test_placement_cannot_run_one_stream_in_two_processes() -> None:
    with pytest.raises(ValueError, match="more than one"):
        SeparateProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_RGB,), (STREAM_RGB,)),
        )


def test_placement_cannot_name_a_kind_and_one_of_its_cameras() -> None:
    """``rgb`` already claims ``camera_0``, so naming both places it twice."""
    with pytest.raises(ValueError, match="more than one"):
        SeparateProcessPerCamera(
            joint_count=2,
            video_count=2,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_RGB,), (CAMERA_0,)),
        )


def test_placement_cannot_name_a_camera_the_case_does_not_have() -> None:
    with pytest.raises(ValueError, match="does not produce"):
        SeparateProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((CAMERA_1,),),
        )


def test_placement_entries_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        SeparateProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_RGB,), ()),
        )


def test_separate_process_per_camera_builds_a_multi_process_session() -> None:
    """The placement must survive `ContextCaseSpec` to reach the dispatch.

    `make_producer_session` reads it off the spec's case, so a spec that drops
    the field fails here and nowhere else.
    """
    session = _session(
        SeparateProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert isinstance(session, MultiProcessProducerSession)
    assert [plan.name for group in session._child_plan_groups for plan in group] == [
        STREAM_RGB
    ]
    assert session._child_trace_keys == {
        f"RGB_IMAGES/{CAMERA_0}",
        f"CUSTOM_1D/marker_{CAMERA_0}",
    }
    # A child's marker series is still expected on disk.
    assert f"marker_{CAMERA_0}" in session.marker_names


def test_a_camera_per_process_reaches_the_session_as_one_group_each() -> None:
    """The camera count sets the child count: two cameras, two groups."""
    session = _session(
        SeparateProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=2,
            image_width=8,
            image_height=8,
        )
    )

    assert [
        [plan.channel_names for plan in group] for group in session._child_plan_groups
    ] == [[(CAMERA_0,)], [(CAMERA_1,)]]
    assert session._child_trace_keys == {
        f"RGB_IMAGES/{CAMERA_0}",
        f"RGB_IMAGES/{CAMERA_1}",
        f"CUSTOM_1D/marker_{CAMERA_0}",
        f"CUSTOM_1D/marker_{CAMERA_1}",
    }


def test_one_cameras_rgb_and_depth_streams_share_its_process() -> None:
    """One RGBD device is one child, not two: both its outputs move together."""
    session = _session(
        SeparateProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=1,
            depth_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert [
        [plan.channel_names for plan in group] for group in session._child_plan_groups
    ] == [[(CAMERA_0,), (DEPTH_CAMERA_0,)]]
    assert session._child_trace_keys == {
        f"RGB_IMAGES/{CAMERA_0}",
        f"DEPTH_IMAGES/{DEPTH_CAMERA_0}",
        f"CUSTOM_1D/marker_{CAMERA_0}",
        f"CUSTOM_1D/marker_{DEPTH_CAMERA_0}",
    }


def test_an_unpaired_camera_index_is_a_device_with_one_output() -> None:
    """Unequal counts still place every stream: device 1 is RGB-only."""
    session = _session(
        SeparateProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=2,
            depth_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert [
        [plan.channel_names for plan in group] for group in session._child_plan_groups
    ] == [[(CAMERA_0,), (DEPTH_CAMERA_0,)], [(CAMERA_1,)]]


def test_a_depth_only_case_has_a_stream_to_move() -> None:
    """No RGB is not no cameras: the depth-only device is the child."""
    session = _session(
        SeparateProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            depth_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert [
        [plan.channel_names for plan in group] for group in session._child_plan_groups
    ] == [[(DEPTH_CAMERA_0,)]]


def test_per_thread_still_builds_a_single_process_session() -> None:
    session = _session(
        PerThread(
            duration_sec=1, joint_count=2, video_count=1, image_width=8, image_height=8
        )
    )

    assert isinstance(session, LifetimeProducerSession)
    assert not session.needs_stop_gate_bracket


def test_only_the_multi_process_session_asks_for_the_gate_bracket() -> None:
    """Measuring it costs a polling thread inside every stop_recording call."""
    assert MultiProcessProducerSession.needs_stop_gate_bracket
    assert not LifetimeProducerSession.needs_stop_gate_bracket


def test_late_starting_keys_cover_the_moved_streams_and_their_markers() -> None:
    """A moved stream's marker starts late for exactly the same reason it does."""
    case = SeparateProcessPerCamera(
        joint_count=2, video_count=1, image_width=8, image_height=8
    )

    assert late_starting_trace_keys(case) == {
        f"RGB_IMAGES/{CAMERA_0}",
        f"CUSTOM_1D/marker_{CAMERA_0}",
    }


def test_late_starting_keys_follow_a_camera_named_by_channel() -> None:
    """A camera moved on its own starts late; the one left behind does not."""
    case = SeparateProcessPerCamera(
        joint_count=2,
        video_count=2,
        image_width=8,
        image_height=8,
        producer_process_streams=((CAMERA_1,),),
    )

    assert late_starting_trace_keys(case) == {
        f"RGB_IMAGES/{CAMERA_1}",
        f"CUSTOM_1D/marker_{CAMERA_1}",
    }


def test_nothing_starts_late_without_a_moved_stream() -> None:
    for case in (
        Synchronous(joint_count=2),
        PerThread(joint_count=2, video_count=1, image_width=8, image_height=8),
    ):
        assert late_starting_trace_keys(case) == frozenset()


def test_case_id_names_an_overridden_placement_but_not_the_default_one() -> None:
    shape = {"joint_count": 2, "video_count": 1, "image_width": 8, "image_height": 8}

    assert "proc" not in case_id(SeparateProcessPerCamera(**shape))
    assert "2proc-rgb-jointtorques" in case_id(
        SeparateProcessPerCamera(
            **shape,
            producer_process_streams=((STREAM_RGB,), (STREAM_JOINT_TORQUES,)),
        )
    )
