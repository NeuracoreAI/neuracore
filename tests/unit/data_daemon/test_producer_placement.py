"""Offline checks on the multi-process producer's placement and dispatch.

All of it is decided before a process is spawned or a robot connected.
"""

from __future__ import annotations

import threading

import pytest

from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    PerThread,
    ProcessPerCamera,
    ProcessPerLimbPerCamera,
    SeparateProcessStartStop,
    Synchronous,
    case_id,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CAMERA_0,
    CAMERA_1,
    DEPTH_CAMERA_0,
    DETAIL_FLAT,
    PACING_DEADLINE,
    STREAM_DEPTH,
    STREAM_JOINT_POSITIONS,
    STREAM_JOINT_TORQUES,
    STREAM_JOINT_VELOCITIES,
    STREAM_RGB,
    joint_group_name,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.producers import (
    LifetimeProducerSession,
    MultiProcessProducerSession,
    ProducerRequest,
    make_producer_session,
    partition_plans,
    run_threaded_logging,
)
from tests.integration.platform.data_daemon.shared.test_case.streams import (
    build_stream_plans,
    case_stream_plans,
    late_starting_trace_keys,
)

# cspell:ignore jointtorques jointgroups


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


def _camera_children(session) -> list[list[tuple[str, ...]]]:
    """The child groups holding cameras: every session also has a joints child."""
    return [
        [plan.channel_names for plan in group]
        for group in session._child_plan_groups
        if all(plan.is_video for plan in group)
    ]


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
    """A workload with neither joints nor cameras leaves no child to run."""
    with pytest.raises(ValueError, match="at least one stream to move"):
        ProcessPerCamera(joint_count=0, video_count=0)


def test_multi_process_needs_a_single_context() -> None:
    """Pool workers are daemonic, so a context in one cannot spawn a child."""
    with pytest.raises(ValueError, match="parallel_contexts=1"):
        ProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            parallel_contexts=2,
            recording_count=2,
        )


def test_splitting_the_joints_is_refused_without_a_multi_process_variant() -> None:
    with pytest.raises(ValueError, match="only applies to"):
        PerThread(joint_count=2, joint_process_groups=2)


def test_a_joint_group_cannot_be_left_without_joints() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        ProcessPerLimbPerCamera(joint_count=1)
    with pytest.raises(ValueError, match="at least 1"):
        ProcessPerLimbPerCamera(joint_count=2, joint_process_groups=0)


def test_placement_cannot_name_a_stream_the_case_does_not_produce() -> None:
    with pytest.raises(ValueError, match="does not produce"):
        ProcessPerCamera(
            joint_count=0,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_JOINT_POSITIONS,),),
        )


def test_placement_cannot_run_one_stream_in_two_processes() -> None:
    with pytest.raises(ValueError, match="more than one"):
        ProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_RGB,), (STREAM_RGB,)),
        )


def test_placement_cannot_name_a_kind_and_one_of_its_cameras() -> None:
    """``rgb`` already claims ``camera_0``, so naming both places it twice."""
    with pytest.raises(ValueError, match="more than one"):
        ProcessPerCamera(
            joint_count=2,
            video_count=2,
            image_width=8,
            image_height=8,
            producer_process_streams=((STREAM_RGB,), (CAMERA_0,)),
        )


def test_placement_cannot_name_a_kind_and_a_group_holding_it() -> None:
    """``joints_0`` already claims its own positions, so naming both places that
    stream twice — exactly as ``rgb`` and ``camera_0`` do."""
    with pytest.raises(ValueError, match="more than one"):
        ProcessPerLimbPerCamera(
            joint_count=4,
            producer_process_streams=(
                (joint_group_name(0),),
                (STREAM_JOINT_POSITIONS,),
            ),
        )


def test_placement_cannot_name_a_camera_the_case_does_not_have() -> None:
    with pytest.raises(ValueError, match="does not produce"):
        ProcessPerCamera(
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
            producer_process_streams=((CAMERA_1,),),
        )


def test_placement_entries_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ProcessPerCamera(
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
        ProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert isinstance(session, MultiProcessProducerSession)
    assert [[plan.name for plan in group] for group in session._child_plan_groups] == [
        [STREAM_JOINT_POSITIONS, STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES],
        [STREAM_RGB],
    ]
    assert {f"RGB_IMAGES/{CAMERA_0}", f"CUSTOM_1D/marker_{CAMERA_0}"} <= (
        session._child_trace_keys
    )
    # A child's marker series is still expected on disk.
    assert f"marker_{CAMERA_0}" in session.marker_names


def test_a_camera_per_process_reaches_the_session_as_one_group_each() -> None:
    """The camera count sets the child count: two cameras, two groups."""
    session = _session(
        ProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=2,
            image_width=8,
            image_height=8,
        )
    )

    assert _camera_children(session) == [[(CAMERA_0,)], [(CAMERA_1,)]]
    assert {
        f"RGB_IMAGES/{CAMERA_0}",
        f"RGB_IMAGES/{CAMERA_1}",
        f"CUSTOM_1D/marker_{CAMERA_0}",
        f"CUSTOM_1D/marker_{CAMERA_1}",
    } <= session._child_trace_keys


def test_one_cameras_rgb_and_depth_streams_share_its_process() -> None:
    """One RGBD device is one child, not two: both its outputs move together."""
    session = _session(
        ProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=1,
            depth_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert _camera_children(session) == [[(CAMERA_0,), (DEPTH_CAMERA_0,)]]
    assert {
        f"RGB_IMAGES/{CAMERA_0}",
        f"DEPTH_IMAGES/{DEPTH_CAMERA_0}",
        f"CUSTOM_1D/marker_{CAMERA_0}",
        f"CUSTOM_1D/marker_{DEPTH_CAMERA_0}",
    } <= session._child_trace_keys


def test_an_unpaired_camera_index_is_a_device_with_one_output() -> None:
    """Unequal counts still place every stream: device 1 is RGB-only."""
    session = _session(
        ProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            video_count=2,
            depth_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert _camera_children(session) == [
        [(CAMERA_0,), (DEPTH_CAMERA_0,)],
        [(CAMERA_1,)],
    ]


def test_a_depth_only_case_has_a_stream_to_move() -> None:
    """No RGB is not no cameras: the depth-only device is the child."""
    session = _session(
        ProcessPerCamera(
            duration_sec=1,
            joint_count=2,
            depth_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert _camera_children(session) == [[(DEPTH_CAMERA_0,)]]


def test_one_limb_leaves_the_owner_only_the_window() -> None:
    """Three processes: the controller, the joints, and the camera."""
    session = _session(
        ProcessPerLimbPerCamera(
            duration_sec=1,
            joint_count=2,
            joint_process_groups=1,
            video_count=1,
            image_width=8,
            image_height=8,
        )
    )

    assert [[plan.name for plan in group] for group in session._child_plan_groups] == [
        [STREAM_JOINT_POSITIONS, STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES],
        [STREAM_RGB],
    ]
    assert session._local.plans == ()


def test_one_limb_without_a_camera_still_moves_the_joints() -> None:
    """A joints-only workload is still two processes, not one."""
    session = _session(
        ProcessPerLimbPerCamera(duration_sec=1, joint_count=2, joint_process_groups=1)
    )

    assert [[plan.name for plan in group] for group in session._child_plan_groups] == [
        [STREAM_JOINT_POSITIONS, STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES]
    ]
    assert session._local.plans == ()


def test_split_joints_give_each_group_a_child_holding_every_kind() -> None:
    """A group is a limb, not a data type: each child logs all three kinds for
    its own joints, and the remainder goes to the first group."""
    session = _session(ProcessPerLimbPerCamera(duration_sec=1, joint_count=3))

    assert [[plan.name for plan in group] for group in session._child_plan_groups] == [
        [STREAM_JOINT_POSITIONS, STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES],
        [STREAM_JOINT_POSITIONS, STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES],
    ]
    assert [
        [len(plan.channel_names) for plan in group]
        for group in session._child_plan_groups
    ] == [[2, 2, 2], [1, 1, 1]]
    assert session._local.plans == ()


def test_split_joint_groups_write_disjoint_traces() -> None:
    """Two children feeding one data type must still share no trace: a report
    is folded in per child, so an overlap would silently drop one child's."""
    case = ProcessPerLimbPerCamera(joint_count=4)
    _, children = partition_plans(
        case_stream_plans(case), case.producer_process_streams
    )

    first, second = (
        {key for plan in group for key in plan.trace_keys} for group in children
    )
    assert not first & second
    assert len(first | second) == len(first) + len(second)


def test_a_joint_stream_is_addressed_by_its_kind_and_its_group() -> None:
    """The rule cameras already follow: a stream's kind, or its device."""
    joints = [plan for plan in _plans() if not plan.is_video]

    assert [plan.placement_tokens for plan in joints] == [
        frozenset({STREAM_JOINT_POSITIONS, joint_group_name(0)}),
        frozenset({STREAM_JOINT_VELOCITIES, joint_group_name(0)}),
        frozenset({STREAM_JOINT_TORQUES, joint_group_name(0)}),
    ]


def test_unsplit_joints_keep_their_kind_named_markers() -> None:
    """Unsplit joints are group 0, but none of their traces may move for it."""
    joints = [plan for plan in _plans() if not plan.is_video]

    assert [plan.marker_name for plan in joints] == [
        f"marker_{STREAM_JOINT_POSITIONS}",
        f"marker_{STREAM_JOINT_VELOCITIES}",
        f"marker_{STREAM_JOINT_TORQUES}",
    ]


def test_splitting_the_joints_names_every_stream_by_group_as_well() -> None:
    joints = [plan for plan in _plans(joint_process_groups=2) if not plan.is_video]

    assert {token for plan in joints for token in plan.placement_tokens} == {
        STREAM_JOINT_POSITIONS,
        STREAM_JOINT_VELOCITIES,
        STREAM_JOINT_TORQUES,
        joint_group_name(0),
        joint_group_name(1),
    }


def test_split_joints_can_still_be_placed_by_kind() -> None:
    """One kind of every limb in one process. Splitting does not take the kind
    names away, any more than a per-camera case loses ``rgb``."""
    _, children = partition_plans(
        _plans(joint_process_groups=2), ((STREAM_JOINT_POSITIONS,),)
    )

    assert {plan.name for plan in children[0]} == {STREAM_JOINT_POSITIONS}
    assert [plan.group_name for plan in children[0]] == [
        joint_group_name(0),
        joint_group_name(1),
    ]


@pytest.mark.parametrize(
    "variant", [SeparateProcessStartStop, ProcessPerCamera, ProcessPerLimbPerCamera]
)
def test_every_trace_starts_late_whatever_the_owner_splits(variant) -> None:
    """The owner only holds the window in all of them, however many children
    the split makes, so every trace and every marker starts late."""
    case = variant(joint_count=2, video_count=1, image_width=8, image_height=8)

    assert late_starting_trace_keys(case) == frozenset(
        key for plan in case_stream_plans(case) for key in plan.trace_keys
    )


def test_an_owner_with_no_streams_of_its_own_runs_and_reports_nothing() -> None:
    """The engine is asked for zero threads, which it must not treat as an error."""
    request = ProducerRequest(
        robot=object(),
        robot_name="robot",
        context_index=0,
        recording_index=0,
        seed_ordinal=0,
        plans=(),
        image_width=None,
        image_height=None,
        video_detail=DETAIL_FLAT,
        timestamp_start_s=0.0,
        random_phase=False,
        duration_sec=1,
        pacing=PACING_DEADLINE,
        stop_event=threading.Event(),
    )

    assert run_threaded_logging(request) == {}


def test_per_thread_still_builds_a_single_process_session() -> None:
    session = _session(
        PerThread(
            duration_sec=1, joint_count=2, video_count=1, image_width=8, image_height=8
        )
    )

    assert isinstance(session, LifetimeProducerSession)


def test_late_starting_keys_follow_a_camera_named_by_channel() -> None:
    """A camera moved on its own starts late; the one left behind does not."""
    case = ProcessPerCamera(
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

    assert "proc" not in case_id(ProcessPerCamera(**shape))
    assert "2proc-rgb-jointtorques" in case_id(
        ProcessPerCamera(
            **shape,
            producer_process_streams=((STREAM_RGB,), (STREAM_JOINT_TORQUES,)),
        )
    )


def test_case_id_names_a_limb_count_that_is_not_its_class_default() -> None:
    """The class name implies two limbs and says nothing about any other count,
    so every other count is named — two cases sharing an id would collide as
    pytest parameters."""
    default = ProcessPerLimbPerCamera(joint_count=6)
    one = ProcessPerLimbPerCamera(joint_count=6, joint_process_groups=1)
    three = ProcessPerLimbPerCamera(joint_count=6, joint_process_groups=3)

    assert "jointgroups" not in case_id(default)
    assert "1jointgroups" in case_id(one)
    assert "3jointgroups" in case_id(three)
    assert len({case_id(default), case_id(one), case_id(three)}) == 3
