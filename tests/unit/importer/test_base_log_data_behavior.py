"""Unit tests for NeuracoreDatasetImporter data logging and config ordering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType
from neuracore_types.importer.config import (
    ActionSpaceConfig,
    ActionTypeConfig,
    EndEffectorPoseInputTypeConfig,
    JointPositionInputTypeConfig,
)

from neuracore.importer.core.base import NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import DataValidationError, ImporterError


class _ConcreteImporter(NeuracoreDatasetImporter):
    """Minimal concrete importer for calling _log_data without full initialization."""

    def build_work_items(self):
        return []

    def import_item(self, item):
        return None

    def _record_step(self, step, timestamp):
        return None


def _make_importer_for_log_data() -> NeuracoreDatasetImporter:
    """Build an importer instance with mocks for validation and nc logging."""
    importer = object.__new__(_ConcreteImporter)
    importer.suppress_warnings = False
    importer.logger = MagicMock()
    importer.robot_utils = None
    importer.curr_end_effector_poses = {}
    importer.curr_joint_positions = {}
    importer.prev_ik_solution = [0.1, 0.2]
    importer.joint_info = {}
    importer._validate_input_data = MagicMock()
    importer._validate_joint_data = MagicMock()
    importer._log_transformed_data = MagicMock()
    return importer


def _make_format(
    *,
    joint_position_input_type=JointPositionInputTypeConfig.CUSTOM,
    ee_pose_input_type=EndEffectorPoseInputTypeConfig.CUSTOM,
    action_type=ActionTypeConfig.ABSOLUTE,
    action_space=ActionSpaceConfig.JOINT,
):
    """Return a simple namespace mimicking DataFormat fields."""
    return SimpleNamespace(
        joint_position_input_type=joint_position_input_type,
        ee_pose_input_type=ee_pose_input_type,
        action_type=action_type,
        action_space=action_space,
    )


def _make_item(name="joint1", source_name="ee"):
    """Build a MagicMock MappingItem."""
    item = MagicMock()
    item.name = name
    item.source_name = source_name
    item.transforms = MagicMock(side_effect=lambda x: x)
    return item


@pytest.mark.parametrize(
    "data_type,source_data",
    [
        (DataType.RGB_IMAGES, "rgb"),
        (DataType.DEPTH_IMAGES, "depth"),
        (DataType.POINT_CLOUDS, "pcd"),
        (DataType.LANGUAGE, "pick up block"),
        (DataType.JOINT_POSITIONS, 0.25),
    ],
)
def test_log_data_logs_basic_datatypes(data_type, source_data):
    """_log_data applies transforms and logs for common data types."""
    importer = _make_importer_for_log_data()
    item = _make_item(name="joint1", source_name="ee")
    format_cfg = _make_format()

    importer._log_data(data_type, source_data, item, format_cfg, timestamp=12.0)

    item.transforms.assert_called_once_with(source_data)
    importer._log_transformed_data.assert_called_once()


def test_log_data_ik_requested_converts_end_effector_to_joints():
    """When IK is requested, EE pose is converted to joint positions."""
    importer = _make_importer_for_log_data()
    importer.robot_utils = MagicMock()
    importer.robot_utils.end_effector_to_joint_positions.return_value = {
        "joint1": 0.7,
        "joint2": -0.4,
    }
    importer.curr_end_effector_poses["ee"] = [1, 2, 3, 0, 0, 0, 1]
    item = _make_item(name="ee", source_name="ee")
    format_cfg = _make_format(
        joint_position_input_type=JointPositionInputTypeConfig.END_EFFECTOR
    )

    importer._log_data(DataType.JOINT_POSITIONS, [1, 2, 3], item, format_cfg, 1.5)

    # IK path does not call transforms() and logs per-joint positions.
    item.transforms.assert_not_called()
    importer.robot_utils.end_effector_to_joint_positions.assert_called_once()
    assert importer._validate_joint_data.call_count == 2
    assert importer._log_transformed_data.call_count == 2
    assert importer.prev_ik_solution == [0.7, -0.4]


def test_log_data_fk_requested_converts_joints_to_end_effector():
    """When FK is requested, joint state is converted to an end-effector pose."""
    importer = _make_importer_for_log_data()
    importer.robot_utils = MagicMock()
    importer.robot_utils.joint_positions_to_end_effector_pose.return_value = [0] * 16
    importer.curr_joint_positions = {"joint1": 0.1, "joint2": 0.2}
    item = _make_item(name="gripper")
    format_cfg = _make_format(
        ee_pose_input_type=EndEffectorPoseInputTypeConfig.JOINT_POSITIONS
    )

    importer._log_data(DataType.END_EFFECTOR_POSES, [0.1, 0.2], item, format_cfg, 2.0)

    item.transforms.assert_not_called()
    importer.robot_utils.joint_positions_to_end_effector_pose.assert_called_once_with(
        importer.curr_joint_positions, "gripper"
    )
    importer._log_transformed_data.assert_called_once()


def test_log_data_absolute_action_space_joint_validates_joint_value():
    """Absolute joint-space targets validate and log the transformed scalar."""
    importer = _make_importer_for_log_data()
    item = _make_item(name="joint1")
    format_cfg = _make_format(
        action_type=ActionTypeConfig.ABSOLUTE, action_space=ActionSpaceConfig.JOINT
    )

    importer._log_data(DataType.JOINT_TARGET_POSITIONS, 0.4, item, format_cfg, 3.0)

    importer._validate_joint_data.assert_called_once_with(
        DataType.JOINT_TARGET_POSITIONS, 0.4, "joint1"
    )
    importer._log_transformed_data.assert_called_once()


def test_log_data_absolute_action_space_end_effector_uses_ik():
    """Absolute EE-space targets run through IK and log each joint."""
    importer = _make_importer_for_log_data()
    importer.robot_utils = MagicMock()
    importer.curr_joint_positions = {"joint1": 0.2, "joint2": -0.1}
    importer.robot_utils.end_effector_to_joint_positions.return_value = {
        "joint1": 0.9,
        "joint2": -0.3,
    }
    item = _make_item(name="ee")
    format_cfg = _make_format(
        action_type=ActionTypeConfig.ABSOLUTE,
        action_space=ActionSpaceConfig.END_EFFECTOR,
    )

    importer._log_data(DataType.JOINT_TARGET_POSITIONS, [0] * 7, item, format_cfg, 4.0)

    importer.robot_utils.end_effector_to_joint_positions.assert_called_once()
    assert importer._validate_joint_data.call_count == 2
    assert importer._log_transformed_data.call_count == 2


def test_log_data_relative_action_space_joint_adds_current_position():
    """Relative joint targets add current joint position before validate/log."""
    importer = _make_importer_for_log_data()
    importer.curr_joint_positions = {"joint1": 1.25}
    item = _make_item(name="joint1")
    format_cfg = _make_format(
        action_type=ActionTypeConfig.RELATIVE,
        action_space=ActionSpaceConfig.JOINT,
    )

    importer._log_data(DataType.JOINT_TARGET_POSITIONS, 0.5, item, format_cfg, 5.0)

    importer._validate_joint_data.assert_called_once_with(
        DataType.JOINT_TARGET_POSITIONS, 1.75, "joint1"
    )
    importer._log_transformed_data.assert_called_once_with(
        DataType.JOINT_TARGET_POSITIONS,
        1.75,
        "joint1",
        5.0,
        extrinsics=None,
        intrinsics=None,
    )


def test_log_data_relative_action_space_end_effector_requires_existing_pose():
    """Relative EE-space targets require the current EE pose for the mapping name."""
    importer = _make_importer_for_log_data()
    item = _make_item(name="missing_ee")
    format_cfg = _make_format(
        action_type=ActionTypeConfig.RELATIVE,
        action_space=ActionSpaceConfig.END_EFFECTOR,
    )

    with pytest.raises(DataValidationError, match="not found"):
        importer._log_data(
            DataType.JOINT_TARGET_POSITIONS, [0] * 7, item, format_cfg, 6.0
        )


def test_log_data_ik_requested_raises_without_robot_utils():
    """IK path raises ImporterError when RobotUtils is not initialized."""
    importer = _make_importer_for_log_data()
    importer.curr_end_effector_poses["ee"] = [0] * 7
    item = _make_item(name="ee", source_name="ee")
    format_cfg = _make_format(
        joint_position_input_type=JointPositionInputTypeConfig.END_EFFECTOR
    )

    with pytest.raises(ImporterError, match="Robot utilities are not initialized"):
        importer._log_data(DataType.JOINT_POSITIONS, [0] * 7, item, format_cfg, 7.0)


def test_determine_skip_joint_target_positions_true_when_shift_matches():
    """Skip joint targets when targets match next-step positions (one joint)."""
    importer = object.__new__(_ConcreteImporter)
    importer.ordered_import_configs = [(DataType.JOINT_TARGET_POSITIONS, object())]
    importer.pre_check_joint_positions = {"joint1": [1.0, 2.0, 3.0]}
    importer.pre_check_joint_target_positions = {"joint1": [2.0, 3.0, 999.0]}

    assert importer._determine_skip_joint_target_positions() is True


def test_determine_skip_joint_target_positions_false_when_values_differ():
    """Do not skip when shifted positions and targets disagree."""
    importer = object.__new__(_ConcreteImporter)
    importer.ordered_import_configs = [(DataType.JOINT_TARGET_POSITIONS, object())]
    importer.pre_check_joint_positions = {"joint1": [1.0, 2.0, 3.0]}
    importer.pre_check_joint_target_positions = {"joint1": [0.0, 9.0, 2.0]}

    assert importer._determine_skip_joint_target_positions() is False


def test_determine_skip_joint_target_positions_true_for_multiple_joints():
    """Skip only when the shift relation holds for every joint."""
    importer = object.__new__(_ConcreteImporter)
    importer.ordered_import_configs = [(DataType.JOINT_TARGET_POSITIONS, object())]
    importer.pre_check_joint_positions = {
        "joint1": [1.0, 2.0, 3.0],
        "joint2": [10.0, 11.0, 12.0],
    }
    importer.pre_check_joint_target_positions = {
        "joint1": [2.0, 3.0, 999.0],
        "joint2": [11.0, 12.0, 999.0],
    }

    assert importer._determine_skip_joint_target_positions() is True


def test_determine_skip_joint_target_positions_false_if_one_joint_differs():
    """A mismatch on any joint prevents skipping joint target imports."""
    importer = object.__new__(_ConcreteImporter)
    importer.ordered_import_configs = [(DataType.JOINT_TARGET_POSITIONS, object())]
    importer.pre_check_joint_positions = {
        "joint1": [1.0, 2.0, 3.0],
        "joint2": [10.0, 11.0, 12.0],
    }
    importer.pre_check_joint_target_positions = {
        "joint1": [2.0, 3.0, 999.0],
        "joint2": [11.0, 99.0, 999.0],
    }

    assert importer._determine_skip_joint_target_positions() is False


def test_get_ordered_import_configs_default_order_preserved():
    """Without FK/IK, config order follows dict iteration order."""
    importer = object.__new__(_ConcreteImporter)
    importer.dataset_config = SimpleNamespace(
        data_import_config={
            DataType.RGB_IMAGES: SimpleNamespace(format=_make_format()),
            DataType.JOINT_POSITIONS: SimpleNamespace(format=_make_format()),
        }
    )

    ordered = importer._get_ordered_import_configs()
    assert [item[0] for item in ordered] == [
        DataType.RGB_IMAGES,
        DataType.JOINT_POSITIONS,
    ]


def test_get_ordered_import_configs_fk_prioritizes_joint_positions_first():
    """FK from joints puts joint positions before end-effector poses."""
    importer = object.__new__(_ConcreteImporter)
    importer.dataset_config = SimpleNamespace(
        data_import_config={
            DataType.RGB_IMAGES: SimpleNamespace(format=_make_format()),
            DataType.END_EFFECTOR_POSES: SimpleNamespace(
                format=_make_format(
                    ee_pose_input_type=EndEffectorPoseInputTypeConfig.JOINT_POSITIONS
                )
            ),
            DataType.JOINT_POSITIONS: SimpleNamespace(format=_make_format()),
        }
    )

    ordered = importer._get_ordered_import_configs()
    assert [item[0] for item in ordered][:2] == [
        DataType.JOINT_POSITIONS,
        DataType.END_EFFECTOR_POSES,
    ]


def test_get_ordered_import_configs_ik_prioritizes_end_effector_first():
    """IK to joints puts end-effector poses before joint positions."""
    importer = object.__new__(_ConcreteImporter)
    importer.dataset_config = SimpleNamespace(
        data_import_config={
            DataType.RGB_IMAGES: SimpleNamespace(format=_make_format()),
            DataType.END_EFFECTOR_POSES: SimpleNamespace(format=_make_format()),
            DataType.JOINT_POSITIONS: SimpleNamespace(
                format=_make_format(
                    joint_position_input_type=JointPositionInputTypeConfig.END_EFFECTOR
                )
            ),
        }
    )

    ordered = importer._get_ordered_import_configs()
    assert [item[0] for item in ordered][:2] == [
        DataType.END_EFFECTOR_POSES,
        DataType.JOINT_POSITIONS,
    ]


def test_get_ordered_import_configs_raises_when_fk_and_ik_both_requested():
    """Conflicting FK and IK requests raise ImportError."""
    importer = object.__new__(_ConcreteImporter)
    importer.dataset_config = SimpleNamespace(
        data_import_config={
            DataType.END_EFFECTOR_POSES: SimpleNamespace(
                format=_make_format(
                    ee_pose_input_type=EndEffectorPoseInputTypeConfig.JOINT_POSITIONS
                )
            ),
            DataType.JOINT_POSITIONS: SimpleNamespace(
                format=_make_format(
                    joint_position_input_type=JointPositionInputTypeConfig.END_EFFECTOR
                )
            ),
        }
    )

    with pytest.raises(ImportError, match="Cannot request both FK and IK"):
        importer._get_ordered_import_configs()
