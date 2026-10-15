import io
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from neuracore_types import DataType
from neuracore_types.importer.config import (
    ImageChannelOrderConfig,
    ImageConventionConfig,
)
from neuracore_types.importer.data_config import DataFormat
from neuracore_types.importer.transform import (
    CastToNumpyDtype,
    Clip,
    DataTransformSequence,
    Scale,
)
from neuracore_types.nc_data.camera_data import (
    RGBCameraDataImportConfig,
    RGBCameraDataMappingItem,
)
from PIL import Image

from neuracore.importer.core import base
from neuracore.importer.core.base import NeuracoreDatasetImporter
from neuracore.importer.core.png_frame import PNGFrame, keeps_uint8_rgb_unchanged
from neuracore.importer.mcap.utils import read_image_data


def _png(array: np.ndarray, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", **save_kwargs)
    return buffer.getvalue()


def _interlaced(png: bytes) -> bytes:
    """Mark the IHDR as Adam7; PIL cannot write interlaced PNGs."""
    return png[:28] + b"\x01" + png[29:]


def _rgb() -> np.ndarray:
    return np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)


def test_from_bytes_accepts_rgb8_png_and_decodes_losslessly():
    frame = PNGFrame.from_bytes(_png(_rgb()))

    assert frame is not None
    assert (frame.width, frame.height) == (5, 4)
    np.testing.assert_array_equal(frame.decode(), _rgb())


@pytest.mark.parametrize(
    "payload",
    [
        _png(np.zeros((4, 5, 4), dtype=np.uint8)),  # RGBA
        _png(np.zeros((4, 5), dtype=np.uint8)),  # greyscale
        _png(np.zeros((4, 5), dtype=np.uint16)),  # 16-bit
        _interlaced(_png(_rgb())),
        b"not a png",
        _png(_rgb())[:-1],  # truncated
    ],
)
def test_from_bytes_rejects_other_images(payload):
    assert PNGFrame.from_bytes(payload) is None


def test_default_rgb_transforms_keep_uint8_unchanged():
    config = RGBCameraDataImportConfig(
        mapping=[RGBCameraDataMappingItem(name="cam")],
        format=DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_LAST,
            order_of_channels=ImageChannelOrderConfig.RGB,
        ),
    )

    assert keeps_uint8_rgb_unchanged(config.mapping[0].transforms)


@pytest.mark.parametrize(
    "transforms",
    [
        [Clip(min=0.0, max=200.0)],
        [CastToNumpyDtype(dtype=np.float32)],
        [Scale(factor=2.0)],
    ],
)
def test_other_transforms_need_a_decode(transforms):
    assert not keeps_uint8_rgb_unchanged(DataTransformSequence(transforms=transforms))


def test_read_image_data_keeps_rgb_png_encoded():
    payload = _png(_rgb())
    message = SimpleNamespace(data=payload, format="png")

    result = read_image_data(
        DataType.RGB_IMAGES, message, message, logger=logging.getLogger()
    )

    assert result == PNGFrame.from_bytes(payload)


def test_read_image_data_decodes_rgba_png():
    message = SimpleNamespace(
        data=_png(np.zeros((4, 5, 4), dtype=np.uint8)), format="png"
    )

    result = read_image_data(
        DataType.RGB_IMAGES, message, message, logger=logging.getLogger()
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 5, 3)


class _Importer(NeuracoreDatasetImporter):
    def build_work_items(self):
        return []

    def import_item(self, item):
        return None

    def _record_step(self, step, timestamp):
        return None

    def _resolve_source_path(self, source, source_name):
        return source


def _importer() -> _Importer:
    importer = object.__new__(_Importer)
    importer.dry_run = False
    importer._worker_id = 0
    importer._instance_base = 0
    importer._last_logged_timestamps = {}
    importer.dataset_config = SimpleNamespace(robot=SimpleNamespace(name="robot"))
    importer._validate_input_data = MagicMock()
    importer._log_transformed_data = MagicMock()
    return importer


def _item(transforms: list) -> SimpleNamespace:
    return SimpleNamespace(
        name="cam", transforms=DataTransformSequence(transforms=transforms)
    )


def test_log_data_sends_png_without_decoding(monkeypatch):
    importer = _importer()
    log_rgb_png = MagicMock()
    monkeypatch.setattr(base.nc, "log_rgb_png", log_rgb_png)
    frame = PNGFrame.from_bytes(_png(_rgb()))

    importer._log_data(
        DataType.RGB_IMAGES,
        frame,
        _item([Clip(min=0.0, max=255.0), CastToNumpyDtype(dtype=np.uint8)]),
        format=SimpleNamespace(),
        timestamp=1.0,
    )

    assert log_rgb_png.call_args.kwargs["png"] == frame.data
    importer._validate_input_data.assert_not_called()
    importer._log_transformed_data.assert_not_called()


def test_log_data_decodes_png_when_transforms_change_pixels(monkeypatch):
    importer = _importer()
    log_rgb_png = MagicMock()
    monkeypatch.setattr(base.nc, "log_rgb_png", log_rgb_png)
    frame = PNGFrame.from_bytes(_png(_rgb()))

    importer._log_data(
        DataType.RGB_IMAGES,
        frame,
        _item([Scale(factor=0.5)]),
        format=SimpleNamespace(
            joint_position_input_type=None,
            ee_pose_input_type=None,
            action_type=None,
            action_space=None,
        ),
        timestamp=1.0,
    )

    log_rgb_png.assert_not_called()
    logged = importer._log_transformed_data.call_args.args[1]
    np.testing.assert_array_equal(logged, _rgb() * 0.5)
