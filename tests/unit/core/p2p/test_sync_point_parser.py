"""Unit tests for sync_point_parser module."""

import numpy as np
import pytest
from neuracore_types import (
    Custom1DData,
    DataType,
    DepthCameraData,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    RGBCameraData,
    RobotStreamTrack,
    SynchronizedPoint,
)

from neuracore.core.streaming.p2p.consumer.sync_point_parser import (
    merge_sync_points,
    parse_sync_point,
)


def _create_track(data_type: DataType, label: str = "test_label") -> RobotStreamTrack:
    """Helper to create a RobotStreamTrack for testing."""
    return RobotStreamTrack(
        robot_id="robot_1",
        robot_instance=0,
        stream_id="stream_1",
        data_type=data_type,
        label=label,
        mid="mid_1",
    )


class TestParseSyncPointJointData:
    """Tests for parsing joint data types."""

    @pytest.mark.parametrize(
        "data_type",
        [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.JOINT_TARGET_POSITIONS,
        ],
    )
    def test_parse_joint_data_types(self, data_type: DataType):
        """Test parsing all joint data types."""
        joint_data = JointData(timestamp=1.5, value=0.75)
        message_data = joint_data.model_dump_json()
        track = _create_track(data_type, label="arm_joint")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 1.5
        assert data_type in result.data
        assert "arm_joint" in result.data[data_type]
        parsed_joint = result.data[data_type]["arm_joint"]
        assert isinstance(parsed_joint, JointData)
        assert parsed_joint.value == 0.75


class TestParseSyncPointLanguageData:
    """Tests for parsing language data."""

    def test_parse_language_data(self):
        """Test parsing language data."""
        language_data = LanguageData(timestamp=2.0, text="pick up the red cube")
        message_data = language_data.model_dump_json()
        track = _create_track(DataType.LANGUAGE, label="instruction")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 2.0
        assert DataType.LANGUAGE in result.data
        assert "instruction" in result.data[DataType.LANGUAGE]
        parsed_lang = result.data[DataType.LANGUAGE]["instruction"]
        assert isinstance(parsed_lang, LanguageData)
        assert parsed_lang.text == "pick up the red cube"


class TestParseSyncPointCameraData:
    """Tests for parsing camera data (RGB and Depth)."""

    def test_parse_rgb_camera_data(self):
        """Test parsing RGB camera data with image decoding."""
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_image[5, 5] = [255, 0, 0]
        rgb_data = RGBCameraData(timestamp=3.0, frame_idx=42, frame=test_image)
        message_data = rgb_data.model_dump_json()
        track = _create_track(DataType.RGB_IMAGES, label="front_camera")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 3.0
        assert DataType.RGB_IMAGES in result.data
        assert "front_camera" in result.data[DataType.RGB_IMAGES]
        parsed_rgb = result.data[DataType.RGB_IMAGES]["front_camera"]
        assert isinstance(parsed_rgb, RGBCameraData)
        assert parsed_rgb.frame_idx == 42
        assert isinstance(parsed_rgb.frame, np.ndarray)
        assert parsed_rgb.frame.shape == (10, 10, 3)
        np.testing.assert_array_equal(parsed_rgb.frame, test_image)

    def test_parse_depth_camera_data(self):
        """Test parsing depth camera data with image decoding."""
        # Depth is encoded as an RGB PNG and decoded back to a 2D depth map.
        test_depth = np.arange(64, dtype=np.float32).reshape(8, 8) / 10.0
        depth_data = DepthCameraData(timestamp=3.5, frame_idx=10, frame=test_depth)
        message_data = depth_data.model_dump_json()
        track = _create_track(DataType.DEPTH_IMAGES, label="depth_sensor")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 3.5
        assert DataType.DEPTH_IMAGES in result.data
        assert "depth_sensor" in result.data[DataType.DEPTH_IMAGES]
        parsed_depth = result.data[DataType.DEPTH_IMAGES]["depth_sensor"]
        assert isinstance(parsed_depth, DepthCameraData)
        assert parsed_depth.frame_idx == 10
        frame = parsed_depth.frame
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (8, 8)
        assert np.allclose(frame, test_depth, atol=1e-5)


class TestParseSyncPointEndEffectorPose:
    """Tests for parsing end effector pose data."""

    def test_parse_end_effector_pose_data(self):
        """Test parsing end effector pose data."""
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        ee_data = EndEffectorPoseData(timestamp=4.0, pose=pose)
        message_data = ee_data.model_dump_json()
        track = _create_track(DataType.END_EFFECTOR_POSES, label="left_gripper")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 4.0
        assert DataType.END_EFFECTOR_POSES in result.data
        assert "left_gripper" in result.data[DataType.END_EFFECTOR_POSES]
        parsed_ee = result.data[DataType.END_EFFECTOR_POSES]["left_gripper"]
        assert isinstance(parsed_ee, EndEffectorPoseData)
        np.testing.assert_array_almost_equal(parsed_ee.pose, pose)


class TestParseSyncPointGripper:
    """Tests for parsing parallel gripper data."""

    def test_parse_parallel_gripper_open_amount(self):
        """Test parsing parallel gripper open amount data."""
        gripper_data = ParallelGripperOpenAmountData(timestamp=5.0, open_amount=0.8)
        message_data = gripper_data.model_dump_json()
        track = _create_track(DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS, label="gripper_1")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 5.0
        assert DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in result.data
        assert "gripper_1" in result.data[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        parsed_gripper = result.data[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS][
            "gripper_1"
        ]
        assert isinstance(parsed_gripper, ParallelGripperOpenAmountData)
        assert parsed_gripper.open_amount == 0.8


class TestParseSyncPointPointCloud:
    """Tests for parsing point cloud data."""

    def test_parse_point_cloud_data(self):
        """Test parsing point cloud data."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        pc_data = PointCloudData(timestamp=6.0, points=points)
        message_data = pc_data.model_dump_json()
        track = _create_track(DataType.POINT_CLOUDS, label="lidar")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 6.0
        assert DataType.POINT_CLOUDS in result.data
        assert "lidar" in result.data[DataType.POINT_CLOUDS]
        parsed_pc = result.data[DataType.POINT_CLOUDS]["lidar"]
        assert isinstance(parsed_pc, PointCloudData)
        assert isinstance(parsed_pc.points, np.ndarray)
        np.testing.assert_allclose(parsed_pc.points, points)


class TestParseSyncPointCustom1D:
    """Tests for parsing custom 1D data."""

    def test_parse_custom_1d_data(self):
        """Test parsing custom 1D data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        custom_data = Custom1DData(timestamp=7.0, data=data)
        message_data = custom_data.model_dump_json()
        track = _create_track(DataType.CUSTOM_1D, label="sensor_readings")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 7.0
        assert DataType.CUSTOM_1D in result.data
        assert "sensor_readings" in result.data[DataType.CUSTOM_1D]
        parsed_custom = result.data[DataType.CUSTOM_1D]["sensor_readings"]
        assert isinstance(parsed_custom, Custom1DData)
        assert isinstance(parsed_custom.data, np.ndarray)
        np.testing.assert_allclose(parsed_custom.data, data)


class TestParseSyncPointPose:
    """Tests for parsing pose data."""

    def test_parse_pose_data(self):
        """Test parsing pose data."""
        pose = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        pose_data = PoseData(timestamp=8.0, pose=pose)
        message_data = pose_data.model_dump_json()
        track = _create_track(DataType.POSES, label="base_pose")

        result = parse_sync_point(message_data, track)

        assert isinstance(result, SynchronizedPoint)
        assert result.timestamp == 8.0
        assert DataType.POSES in result.data
        assert "base_pose" in result.data[DataType.POSES]
        parsed_pose = result.data[DataType.POSES]["base_pose"]
        assert isinstance(parsed_pose, PoseData)
        assert isinstance(parsed_pose.pose, np.ndarray)
        np.testing.assert_array_almost_equal(parsed_pose.pose, pose)


class TestParseSyncPointErrors:
    """Tests for error handling in parse_sync_point."""

    def test_invalid_json_raises_value_error(self):
        """Test that invalid JSON raises ValueError."""
        track = _create_track(DataType.JOINT_POSITIONS)

        with pytest.raises(ValueError, match="Invalid or unsupported data"):
            parse_sync_point("not valid json", track)

    def test_invalid_data_for_type_raises_value_error(self):
        """Test that data mismatched with expected type raises ValueError."""
        language_data = LanguageData(timestamp=1.0, text="hello")
        message_data = language_data.model_dump_json()
        track = _create_track(DataType.JOINT_POSITIONS)

        with pytest.raises(ValueError, match="Invalid or unsupported data"):
            parse_sync_point(message_data, track)

    def test_unsupported_track_data_type_raises_value_error(self):
        """Test that an unsupported track data_type raises ValueError."""
        # Use model_construct to bypass pydantic validation so we can reach the
        # parser's "unsupported track data_type" branch.
        track = RobotStreamTrack.model_construct(
            data_type="UNSUPPORTED_DATA_TYPE",
            label="any",
        )

        with pytest.raises(ValueError, match="Unsupported track data_type"):
            parse_sync_point("{}", track)  # payload irrelevant for unsupported types


class TestMergeSyncPoints:
    """Tests for merge_sync_points function."""

    def test_merge_empty_returns_empty_sync_point(self):
        """Test merging no sync points returns empty SynchronizedPoint."""
        result = merge_sync_points()

        assert isinstance(result, SynchronizedPoint)
        assert result.data == {}

    def test_merge_single_sync_point(self):
        """Test merging a single sync point returns equivalent point."""
        joint_data = JointData(timestamp=1.0, value=0.5)
        sp = SynchronizedPoint(
            timestamp=1.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data}},
        )

        result = merge_sync_points(sp)

        assert result.timestamp == 1.0
        assert DataType.JOINT_POSITIONS in result.data
        assert "arm" in result.data[DataType.JOINT_POSITIONS]

    def test_merge_multiple_different_data_types(self):
        """Test merging sync points with different data types."""
        joint_data = JointData(timestamp=1.0, value=0.5)
        language_data = LanguageData(timestamp=2.0, text="hello")

        sp1 = SynchronizedPoint(
            timestamp=1.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data}},
        )
        sp2 = SynchronizedPoint(
            timestamp=2.0,
            data={DataType.LANGUAGE: {"instruction": language_data}},
        )

        result = merge_sync_points(sp1, sp2)

        assert result.timestamp == 2.0
        assert DataType.JOINT_POSITIONS in result.data
        assert DataType.LANGUAGE in result.data
        assert "arm" in result.data[DataType.JOINT_POSITIONS]
        assert "instruction" in result.data[DataType.LANGUAGE]

    def test_merge_later_timestamp_overrides(self):
        """Test that later timestamps override earlier data for same key."""
        joint_data_early = JointData(timestamp=1.0, value=0.5)
        joint_data_late = JointData(timestamp=3.0, value=0.9)

        sp1 = SynchronizedPoint(
            timestamp=1.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data_early}},
        )
        sp2 = SynchronizedPoint(
            timestamp=3.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data_late}},
        )

        result = merge_sync_points(sp1, sp2)

        assert result.timestamp == 3.0
        arm = result.data[DataType.JOINT_POSITIONS]["arm"]
        assert isinstance(arm, JointData)
        assert arm.value == 0.9

    def test_merge_preserves_different_labels(self):
        """Test that merging preserves data with different labels."""
        joint_data_1 = JointData(timestamp=1.0, value=0.5)
        joint_data_2 = JointData(timestamp=2.0, value=0.7)

        sp1 = SynchronizedPoint(
            timestamp=1.0,
            data={DataType.JOINT_POSITIONS: {"arm_left": joint_data_1}},
        )
        sp2 = SynchronizedPoint(
            timestamp=2.0,
            data={DataType.JOINT_POSITIONS: {"arm_right": joint_data_2}},
        )

        result = merge_sync_points(sp1, sp2)

        assert result.timestamp == 2.0
        assert "arm_left" in result.data[DataType.JOINT_POSITIONS]
        assert "arm_right" in result.data[DataType.JOINT_POSITIONS]
        arm_left = result.data[DataType.JOINT_POSITIONS]["arm_left"]
        arm_right = result.data[DataType.JOINT_POSITIONS]["arm_right"]
        assert isinstance(arm_left, JointData)
        assert isinstance(arm_right, JointData)
        assert arm_left.value == 0.5
        assert arm_right.value == 0.7

    def test_merge_out_of_order_timestamps(self):
        """Test merging sync points provided out of timestamp order."""
        joint_data_early = JointData(timestamp=1.0, value=0.5)
        joint_data_late = JointData(timestamp=3.0, value=0.9)

        sp_late = SynchronizedPoint(
            timestamp=3.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data_late}},
        )
        sp_early = SynchronizedPoint(
            timestamp=1.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data_early}},
        )

        result = merge_sync_points(sp_late, sp_early)

        assert result.timestamp == 3.0
        arm = result.data[DataType.JOINT_POSITIONS]["arm"]
        assert isinstance(arm, JointData)
        assert arm.value == 0.9

    def test_merge_three_sync_points(self):
        """Test merging three sync points."""
        joint_data = JointData(timestamp=1.0, value=0.5)
        language_data = LanguageData(timestamp=2.0, text="hello")
        gripper_data = ParallelGripperOpenAmountData(timestamp=3.0, open_amount=0.8)

        sp1 = SynchronizedPoint(
            timestamp=1.0,
            data={DataType.JOINT_POSITIONS: {"arm": joint_data}},
        )
        sp2 = SynchronizedPoint(
            timestamp=2.0,
            data={DataType.LANGUAGE: {"instruction": language_data}},
        )
        sp3 = SynchronizedPoint(
            timestamp=3.0,
            data={DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {"gripper": gripper_data}},
        )

        result = merge_sync_points(sp1, sp2, sp3)

        assert result.timestamp == 3.0
        assert len(result.data) == 3
        assert DataType.JOINT_POSITIONS in result.data
        assert DataType.LANGUAGE in result.data
        assert DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in result.data
