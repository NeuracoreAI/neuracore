"""Unit tests for validation methods in NeuracoreDatasetImporter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from neuracore_types import DataType
from neuracore_types.importer.config import (
    AngleConfig,
    EulerOrderConfig,
    ImageConventionConfig,
    IntrinsicsConfig,
    LanguageConfig,
    OrientationConfig,
    PoseConfig,
    QuaternionOrderConfig,
    RotationConfig,
)
from neuracore_types.importer.data_config import DataFormat
from neuracore_types.importer.transform import ExtrinsicsToMatrix, IntrinsicsToMatrix
from neuracore_types.nc_data.nc_data import MappingItem

from neuracore.core.robot import JointInfo, JointLimits
from neuracore.importer.core.base import NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import (
    DataValidationError,
    DataValidationWarning,
)


class ConcreteTestImporter(NeuracoreDatasetImporter):
    """Concrete implementation of NeuracoreDatasetImporter for testing."""

    def build_work_items(self):
        """Dummy implementation."""
        return []

    def import_item(self, item):
        """Dummy implementation."""
        pass

    def _record_step(self, step, timestamp):
        """Dummy implementation."""
        pass


@pytest.fixture
def mock_dataset_config():
    """Create a mock DatasetImportConfig."""
    config = MagicMock()
    config.robot.name = "test_robot"
    config.frequency = 30.0
    return config


@pytest.fixture
def mock_joint_info():
    """Create mock joint info for testing."""
    return {
        "joint1": JointInfo(
            type="revolute",
            limits=JointLimits(lower=-1.0, upper=1.0, velocity=2.0, effort=10.0),
        ),
        "joint2": JointInfo(
            type="revolute",
            limits=JointLimits(lower=-2.0, upper=2.0, velocity=3.0, effort=20.0),
        ),
    }


@pytest.fixture
def importer(mock_dataset_config, mock_joint_info, tmp_path):
    """Create a test importer instance."""
    return ConcreteTestImporter(
        dataset_dir=tmp_path,
        dataset_config=mock_dataset_config,
        output_dataset_name="test_dataset",
        joint_info=mock_joint_info,
    )


@pytest.fixture
def mock_mapping_item():
    """Create a mock MappingItem."""
    item = MagicMock(spec=MappingItem)
    item.name = "test_item"
    item.transforms = MagicMock(return_value="transformed_data")
    return item


class TestValidateInputData:
    """Tests for _validate_input_data method."""

    def test_validate_rgb_images_channels_last(self, importer):
        """Test RGB image validation with channels last convention."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        data = np.zeros((100, 100, 3), dtype=np.uint8)

        importer._validate_input_data(DataType.RGB_IMAGES, data, format)

    def test_validate_rgb_images_channels_first(self, importer):
        """Test RGB image validation with channels first convention."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_FIRST,
            normalized_pixel_values=False,
        )
        data = np.zeros((3, 100, 100), dtype=np.uint8)

        importer._validate_input_data(DataType.RGB_IMAGES, data, format)

    def test_validate_rgb_images_wrong_dimensions(self, importer):
        """Test RGB image validation with wrong dimensions."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        data = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.RGB_IMAGES, data, format)

    def test_validate_rgb_images_wrong_channels_last(self, importer):
        """Test RGB image validation with wrong channel count for channels last."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        data = np.zeros((100, 100, 4), dtype=np.uint8)  # 4 channels instead of 3

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.RGB_IMAGES, data, format)

    def test_validate_rgb_images_wrong_channels_first(self, importer):
        """Test RGB image validation with wrong channel count for channels first."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_FIRST,
            normalized_pixel_values=False,
        )
        data = np.zeros((4, 100, 100), dtype=np.uint8)  # 4 channels instead of 3

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.RGB_IMAGES, data, format)

    def test_validate_depth_images_valid(self, importer):
        """Test depth image validation with valid 2D data."""
        format = DataFormat()
        data = np.zeros((100, 100), dtype=np.float32)

        importer._validate_input_data(DataType.DEPTH_IMAGES, data, format)

    def test_validate_depth_images_wrong_dimensions(self, importer):
        """Test depth image validation with wrong dimensions."""
        format = DataFormat()
        data = np.zeros((100, 100, 3), dtype=np.float32)  # 3D instead of 2D

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.DEPTH_IMAGES, data, format)

    def test_validate_point_clouds_valid(self, importer):
        """Test point cloud validation with valid data."""
        format = DataFormat()
        data = np.zeros((100, 3), dtype=np.float32)

        importer._validate_input_data(DataType.POINT_CLOUDS, data, format)

    def test_validate_point_clouds_wrong_dimensions(self, importer):
        """Test point cloud validation with wrong dimensions."""
        format = DataFormat()
        data = np.zeros((100,), dtype=np.float32)  # 1D instead of 2D

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.POINT_CLOUDS, data, format)

    def test_validate_point_clouds_wrong_columns(self, importer):
        """Test point cloud validation with wrong number of columns."""
        format = DataFormat()
        data = np.zeros((100, 4), dtype=np.float32)  # 4 columns instead of 3

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.POINT_CLOUDS, data, format)

    def test_validate_language_string(self, importer):
        """Test language validation with string type."""
        format = DataFormat(language_type=LanguageConfig.STRING)
        data = "test string"

        importer._validate_input_data(DataType.LANGUAGE, data, format)

    def test_validate_language_wrong_type(self, importer):
        """Test language validation with wrong type."""
        format = DataFormat(language_type=LanguageConfig.STRING)
        data = 123  # Not a string

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.LANGUAGE, data, format)

    def test_validate_poses_matrix(self, importer):
        """Test pose validation with matrix type."""
        format = DataFormat(pose_type=PoseConfig.MATRIX)
        data = np.zeros(16, dtype=np.float32)

        importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_poses_quaternion(self, importer):
        """Test pose validation with quaternion orientation."""
        format = DataFormat(
            pose_type=PoseConfig.POSITION_ORIENTATION,
            orientation=OrientationConfig(type=RotationConfig.QUATERNION),
        )
        data = np.zeros(7, dtype=np.float32)

        importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_poses_euler(self, importer):
        """Test pose validation with euler orientation."""
        format = DataFormat(
            pose_type=PoseConfig.POSITION_ORIENTATION,
            orientation=OrientationConfig(type=RotationConfig.EULER),
        )
        data = np.zeros(6, dtype=np.float32)

        importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_poses_axis_angle(self, importer):
        """Test pose validation with axis-angle orientation."""
        format = DataFormat(
            pose_type=PoseConfig.POSITION_ORIENTATION,
            orientation=OrientationConfig(type=RotationConfig.AXIS_ANGLE),
        )
        data = np.zeros(6, dtype=np.float32)

        importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_poses_matrix_orientation(self, importer):
        """Test pose validation with matrix orientation."""
        format = DataFormat(
            pose_type=PoseConfig.POSITION_ORIENTATION,
            orientation=OrientationConfig(type=RotationConfig.MATRIX),
        )
        data = np.zeros(9, dtype=np.float32)

        importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_poses_wrong_dimensions(self, importer):
        """Test pose validation with wrong dimensions."""
        format = DataFormat(pose_type=PoseConfig.MATRIX)
        data = np.zeros((16, 1), dtype=np.float32)  # 2D instead of 1D

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_poses_wrong_matrix_size(self, importer):
        """Test pose validation with wrong matrix size."""
        format = DataFormat(pose_type=PoseConfig.MATRIX)
        data = np.zeros(12, dtype=np.float32)  # Wrong size for matrix

        with pytest.raises(DataValidationError):
            importer._validate_input_data(DataType.POSES, data, format)

    def test_validate_end_effector_poses(self, importer):
        """Test end effector pose validation."""
        format = DataFormat(pose_type=PoseConfig.MATRIX)
        data = np.zeros(16, dtype=np.float32)

        importer._validate_input_data(DataType.END_EFFECTOR_POSES, data, format)

    def test_validate_unknown_data_type(self, importer):
        """Test validation with unknown data type (should not raise)."""
        format = DataFormat()
        data = "some data"

        importer._validate_input_data(DataType.CUSTOM_1D, data, format)


class TestValidateJointData:
    """Tests for _validate_joint_data method."""

    def test_validate_joint_positions_valid(self, importer):
        """Test joint position validation with valid data."""
        data = 0.5

        importer._validate_joint_data(DataType.JOINT_POSITIONS, data, "joint1")

    def test_validate_joint_positions_below_limit(self, importer):
        """Test joint position validation with value below limit."""
        data = -1.5  # Below lower limit -1.0

        with pytest.raises(DataValidationWarning):
            importer._validate_joint_data(DataType.JOINT_POSITIONS, data, "joint1")

    def test_validate_joint_positions_above_limit(self, importer):
        """Test joint position validation with value above limit."""
        data = 1.5  # Above upper limit 1.0

        with pytest.raises(DataValidationWarning):
            importer._validate_joint_data(DataType.JOINT_POSITIONS, data, "joint1")

    def test_validate_joint_positions_within_upper_tolerance(self, importer):
        """Do not warn for tiny floating-point overshoot near upper limit."""
        data = 1.0000005  # Within epsilon above upper limit 1.0

        importer._validate_joint_data(DataType.JOINT_POSITIONS, data, "joint1")

    def test_validate_joint_positions_within_lower_tolerance(self, importer):
        """Do not warn for tiny floating-point undershoot near lower limit."""
        data = -1.0000005  # Within epsilon below lower limit -1.0

        importer._validate_joint_data(DataType.JOINT_POSITIONS, data, "joint1")

    def test_validate_joint_positions_joint_not_found(self, importer):
        """Test joint position validation with joint not in joint_info."""
        data = 0.5

        with pytest.raises(DataValidationError):
            importer._validate_joint_data(
                DataType.JOINT_POSITIONS, data, "unknown_joint"
            )

    def test_validate_joint_velocities_valid(self, importer):
        """Test joint velocity validation with valid data."""
        data = 1.0

        importer._validate_joint_data(DataType.JOINT_VELOCITIES, data, "joint1")

    def test_validate_joint_velocities_exceeds_limit(self, importer):
        """Test joint velocity validation with value exceeding limit."""
        data = 2.5  # Exceeds limit 2.0

        with pytest.raises(DataValidationWarning):
            importer._validate_joint_data(DataType.JOINT_VELOCITIES, data, "joint1")

    def test_validate_joint_velocities_negative_exceeds_limit(self, importer):
        """Test joint velocity validation with negative value exceeding limit."""
        data = -2.5  # Exceeds limit 2.0 (absolute value)

        with pytest.raises(DataValidationWarning):
            importer._validate_joint_data(DataType.JOINT_VELOCITIES, data, "joint1")

    def test_validate_joint_velocities_joint_not_found(self, importer):
        """Test joint velocity validation with joint not in joint_info."""
        data = 1.0

        with pytest.raises(DataValidationError):
            importer._validate_joint_data(
                DataType.JOINT_VELOCITIES, data, "unknown_joint"
            )

    def test_validate_joint_torques_valid(self, importer):
        """Test joint torque validation with valid data."""
        data = 5.0

        importer._validate_joint_data(DataType.JOINT_TORQUES, data, "joint1")

    def test_validate_joint_torques_exceeds_limit(self, importer):
        """Test joint torque validation with value exceeding limit."""
        data = 15.0  # Exceeds limit 10.0

        with pytest.raises(DataValidationWarning):
            importer._validate_joint_data(DataType.JOINT_TORQUES, data, "joint1")

    def test_validate_joint_torques_negative_exceeds_limit(self, importer):
        """Test joint torque validation with negative value exceeding limit."""
        data = -15.0  # Exceeds limit 10.0 (absolute value)

        with pytest.raises(DataValidationWarning):
            importer._validate_joint_data(DataType.JOINT_TORQUES, data, "joint1")

    def test_validate_joint_torques_joint_not_found(self, importer):
        """Test joint torque validation with joint not in joint_info."""
        data = 5.0

        with pytest.raises(DataValidationError):
            importer._validate_joint_data(DataType.JOINT_TORQUES, data, "unknown_joint")

    def test_validate_joint_target_positions(self, importer):
        """Test joint target position validation (uses same validator as positions)."""
        data = 0.5

        importer._validate_joint_data(DataType.JOINT_TARGET_POSITIONS, data, "joint1")


class TestLogData:
    """Tests for _log_data method."""

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_success(self, mock_nc, importer, mock_mapping_item):
        """Test successful data logging."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        source_data = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 1234567890.0

        importer._log_data(
            DataType.RGB_IMAGES, source_data, mock_mapping_item, format, timestamp
        )

        mock_mapping_item.transforms.assert_called_once_with(source_data)
        mock_nc.log_rgb.assert_called_once()

    def test_log_data_validation_error(self, importer, mock_mapping_item):
        """Test data logging with validation error."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        source_data = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        timestamp = 1234567890.0

        with pytest.raises(DataValidationError):
            with patch.object(importer.logger, "error") as mock_error:
                importer._log_data(
                    DataType.RGB_IMAGES,
                    source_data,
                    mock_mapping_item,
                    format,
                    timestamp,
                )
                assert mock_error.called

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_joint_validation_warning(
        self, mock_nc, importer, mock_mapping_item
    ):
        """Test data logging with joint validation warning after transform."""
        format = DataFormat()
        source_data = 0.5
        timestamp = 1234567890.0
        mock_mapping_item.transforms.return_value = 2.0  # Above limit 1.0
        mock_mapping_item.name = "joint1"  # Valid joint name

        with patch.object(importer.logger, "warning") as mock_warning:
            importer._log_data(
                DataType.JOINT_POSITIONS,
                source_data,
                mock_mapping_item,
                format,
                timestamp,
            )
            assert mock_warning.called
            mock_nc.log_joint_position.assert_called_once()

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_transform_exception(self, mock_nc, importer, mock_mapping_item):
        """Test data logging when transform raises an exception."""
        format = DataFormat()
        source_data = 0.5
        timestamp = 1234567890.0
        mock_mapping_item.transforms.side_effect = ValueError("Transform error")

        with pytest.raises(ValueError):
            with patch.object(importer.logger, "error") as mock_error:
                importer._log_data(
                    DataType.JOINT_POSITIONS,
                    source_data,
                    mock_mapping_item,
                    format,
                    timestamp,
                )
                assert mock_error.called

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_logging_exception(self, mock_nc, importer, mock_mapping_item):
        """Test data logging when _log_transformed_data raises an exception."""
        format = DataFormat()
        source_data = np.zeros((100, 100), dtype=np.float32)
        timestamp = 1234567890.0
        mock_nc.log_depth.side_effect = RuntimeError("Logging error")

        with pytest.raises(RuntimeError):
            with patch.object(importer.logger, "error") as mock_error:
                importer._log_data(
                    DataType.DEPTH_IMAGES,
                    source_data,
                    mock_mapping_item,
                    format,
                    timestamp,
                )
                assert mock_error.called

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_joint_data_validates_transformed(
        self, mock_nc, importer, mock_mapping_item
    ):
        """Test that joint data validates the transformed data, not source data."""
        format = DataFormat()
        source_data = 0.5
        timestamp = 1234567890.0
        mock_mapping_item.transforms.return_value = 0.3
        mock_mapping_item.name = "joint1"

        importer._log_data(
            DataType.JOINT_POSITIONS, source_data, mock_mapping_item, format, timestamp
        )

        mock_mapping_item.transforms.assert_called_once_with(source_data)
        mock_nc.log_joint_position.assert_called_once()

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_rgb_with_extrinsics_and_intrinsics(
        self, mock_nc, importer, mock_mapping_item
    ):
        """Test if extrinsics and intrinsics are passed to nc.log_rgb."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        source_data = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 1234567890.0
        extrinsics = np.eye(4, dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float32)

        importer._log_data(
            DataType.RGB_IMAGES,
            source_data,
            mock_mapping_item,
            format,
            timestamp,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

        mock_nc.log_rgb.assert_called_once()
        call_kwargs = mock_nc.log_rgb.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs["extrinsics"], extrinsics)
        np.testing.assert_array_equal(call_kwargs["intrinsics"], intrinsics)

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_depth_with_extrinsics_and_intrinsics(
        self, mock_nc, importer, mock_mapping_item
    ):
        """Test if extrinsics and intrinsics are passed to nc.log_depth."""
        format = DataFormat()
        source_data = np.zeros((100, 100), dtype=np.float32)
        timestamp = 1234567890.0
        extrinsics = np.eye(4, dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float32)

        importer._log_data(
            DataType.DEPTH_IMAGES,
            source_data,
            mock_mapping_item,
            format,
            timestamp,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

        mock_nc.log_depth.assert_called_once()
        call_kwargs = mock_nc.log_depth.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs["extrinsics"], extrinsics)
        np.testing.assert_array_equal(call_kwargs["intrinsics"], intrinsics)

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_point_cloud_with_extrinsics_and_intrinsics(
        self, mock_nc, importer, mock_mapping_item
    ):
        """Test if extrinsics and intrinsics are passed to nc.log_point_cloud."""
        format = DataFormat()
        source_data = np.zeros((100, 3), dtype=np.float32)
        timestamp = 1234567890.0
        extrinsics = np.eye(4, dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float32)

        importer._log_data(
            DataType.POINT_CLOUDS,
            source_data,
            mock_mapping_item,
            format,
            timestamp,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

        mock_nc.log_point_cloud.assert_called_once()
        call_kwargs = mock_nc.log_point_cloud.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs["extrinsics"], extrinsics)
        np.testing.assert_array_equal(call_kwargs["intrinsics"], intrinsics)

    @patch("neuracore.importer.core.base.nc")
    def test_log_data_without_camera_params_passes_none(
        self, mock_nc, importer, mock_mapping_item
    ):
        """Test that omitting extrinsics/intrinsics passes None to the logging call."""
        format = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            normalized_pixel_values=False,
        )
        source_data = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 1234567890.0

        importer._log_data(
            DataType.RGB_IMAGES, source_data, mock_mapping_item, format, timestamp
        )

        mock_nc.log_rgb.assert_called_once()
        call_kwargs = mock_nc.log_rgb.call_args.kwargs
        assert call_kwargs["extrinsics"] is None
        assert call_kwargs["intrinsics"] is None


class TestConvertExtrinsics:
    """Tests for ExtrinsicsToMatrix transform."""

    def test_convert_matrix_format(self):
        """Test converting a flat 16-element array to 4x4 matrix."""
        raw = np.eye(4).flatten()

        result = ExtrinsicsToMatrix(extrinsics_format=PoseConfig.MATRIX)(raw)

        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_convert_position_quaternion_xyzw(self):
        """Test converting position + quaternion (XYZW) to 4x4 matrix."""
        # Identity rotation quaternion (XYZW) = [0, 0, 0, 1]
        raw = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

        result = ExtrinsicsToMatrix(
            extrinsics_format=PoseConfig.POSITION_ORIENTATION,
            extrinsics_orientation=OrientationConfig(
                type=RotationConfig.QUATERNION,
                quaternion_order=QuaternionOrderConfig.XYZW,
            ),
        )(raw)

        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(result[3, :], [0, 0, 0, 1])

    def test_convert_position_quaternion_wxyz(self):
        """Test converting position + quaternion (WXYZ) to 4x4 matrix."""
        # Identity rotation quaternion (WXYZ) = [1, 0, 0, 0]
        raw = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        result = ExtrinsicsToMatrix(
            extrinsics_format=PoseConfig.POSITION_ORIENTATION,
            extrinsics_orientation=OrientationConfig(
                type=RotationConfig.QUATERNION,
                quaternion_order=QuaternionOrderConfig.WXYZ,
            ),
        )(raw)

        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[:3, :3], np.eye(3))

    def test_convert_position_quaternion_xyzw_90deg_x(self):
        """Test non-trivial 90-degree X rotation with XYZW quaternion."""
        # 90 deg around X: quat XYZW = [sin(45), 0, 0, cos(45)]
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        raw = np.array([1.0, 2.0, 3.0, s, 0.0, 0.0, c])

        result = ExtrinsicsToMatrix(
            extrinsics_format=PoseConfig.POSITION_ORIENTATION,
            extrinsics_orientation=OrientationConfig(
                type=RotationConfig.QUATERNION,
                quaternion_order=QuaternionOrderConfig.XYZW,
            ),
        )(raw)

        # 90 deg rotation around X: Y→Z, Z→-Y
        expected_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result[:3, :3], expected_rot)
        np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])

    def test_convert_position_quaternion_wxyz_90deg_x(self):
        """Test non-trivial 90-degree X rotation with WXYZ quaternion."""
        # Same rotation but in WXYZ order: [cos(45), sin(45), 0, 0]
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        raw = np.array([1.0, 2.0, 3.0, c, s, 0.0, 0.0])

        result = ExtrinsicsToMatrix(
            extrinsics_format=PoseConfig.POSITION_ORIENTATION,
            extrinsics_orientation=OrientationConfig(
                type=RotationConfig.QUATERNION,
                quaternion_order=QuaternionOrderConfig.WXYZ,
            ),
        )(raw)

        # Should produce the same rotation matrix as the XYZW test
        expected_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result[:3, :3], expected_rot)
        np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])

    def test_convert_position_euler_radians(self):
        """Test converting position + euler angles (radians) to 4x4 matrix."""
        raw = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

        result = ExtrinsicsToMatrix(
            extrinsics_format=PoseConfig.POSITION_ORIENTATION,
            extrinsics_orientation=OrientationConfig(
                type=RotationConfig.EULER,
                euler_order=EulerOrderConfig.XYZ,
                angle_units=AngleConfig.RADIANS,
            ),
        )(raw)

        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[:3, :3], np.eye(3))

    def test_convert_position_euler_degrees(self):
        """Test converting position + euler angles (degrees) to 4x4 matrix."""
        raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 90.0])

        result = ExtrinsicsToMatrix(
            extrinsics_format=PoseConfig.POSITION_ORIENTATION,
            extrinsics_orientation=OrientationConfig(
                type=RotationConfig.EULER,
                euler_order=EulerOrderConfig.XYZ,
                angle_units=AngleConfig.DEGREES,
            ),
        )(raw)

        assert result.shape == (4, 4)
        # 90 degree rotation around Z axis
        expected_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result[:3, :3], expected_rot)


class TestConvertIntrinsics:
    """Tests for IntrinsicsToMatrix transform."""

    def test_convert_matrix_format(self):
        """Test converting a flat 9-element array to 3x3 matrix."""
        raw = np.eye(3).flatten()

        result = IntrinsicsToMatrix(intrinsics_format=IntrinsicsConfig.MATRIX)(raw)

        assert result.shape == (3, 3)
        np.testing.assert_array_almost_equal(result, np.eye(3))

    def test_convert_flat_format(self):
        """Test converting [fx, fy, cx, cy] to 3x3 intrinsics matrix."""
        raw = np.array([500.0, 500.0, 320.0, 240.0])

        result = IntrinsicsToMatrix(intrinsics_format=IntrinsicsConfig.FLAT)(raw)

        expected = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        assert result.shape == (3, 3)
        np.testing.assert_array_almost_equal(result, expected)
