import shutil
import subprocess
import tempfile
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from neuracore_types import BatchedDepthData, BatchedRGBData, DataType
from omegaconf import OmegaConf

import neuracore.ml
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.preprocessing.methods.h264_match import H264Match
from neuracore.ml.utils.preprocessing import validate_preprocessing_configuration

HEIGHT, WIDTH = 48, 64


def _scene(height: int = HEIGHT, width: int = WIDTH, shift: int = 0) -> np.ndarray:
    """A smooth, camera-like frame. Noise would be destroyed by any codec."""
    ys, xs = np.mgrid[0:height, 0:width]
    red = (128 + 100 * np.sin((xs + shift) / 9.0)).clip(0, 255)
    green = (128 + 100 * np.sin((ys + shift) / 7.0)).clip(0, 255)
    blue = np.full_like(red, 60 + shift)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def _rgb(*frames: np.ndarray) -> BatchedRGBData:
    """Pack HWC frames into a (1, T, C, H, W) float32 batch in 0-255."""
    stacked = np.stack(frames)  # (T, H, W, C)
    frame = torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0).float()
    time_steps = frame.shape[1]
    return BatchedRGBData(
        frame=frame,
        extrinsics=torch.zeros((1, time_steps, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, time_steps, 3, 3), dtype=torch.float32),
    )


def test_preserves_shape_dtype_and_value_range():
    data = _rgb(_scene())
    original = data.frame.clone()

    out = H264Match()(data)

    assert out.frame.shape == original.shape
    assert out.frame.dtype == torch.float32
    assert out.frame.min() >= 0.0
    assert out.frame.max() <= 255.0


def test_degrades_the_frame():
    data = _rgb(_scene())
    original = data.frame.clone()

    out = H264Match()(data)

    assert not torch.equal(out.frame, original), "frame passed through untouched"
    # A smooth scene at CRF 23 should stay recognisable, not be destroyed.
    assert (out.frame - original).abs().mean() < 10.0


def test_is_deterministic_across_instances():
    first = H264Match()(_rgb(_scene()))
    second = H264Match()(_rgb(_scene()))

    assert torch.equal(first.frame, second.frame)


def test_encoder_state_persists_across_calls():
    """Later frames must be predicted from earlier ones, not encoded standalone."""
    method = H264Match()
    method(_rgb(_scene(shift=0)))
    continued = method(_rgb(_scene(shift=1))).frame

    # The same frame encoded by a fresh instance is an IDR rather than a P-frame, so
    # it cannot decode identically to one predicted from a reference.
    standalone = H264Match()(_rgb(_scene(shift=1))).frame

    assert not torch.equal(continued, standalone)


def test_handles_multiple_time_steps():
    data = _rgb(_scene(shift=0), _scene(shift=1), _scene(shift=2))

    out = H264Match()(data)

    assert out.frame.shape == (1, 3, 3, HEIGHT, WIDTH)


def test_resolution_change_rebuilds_codecs():
    method = H264Match()
    method(_rgb(_scene()))

    out = method(_rgb(_scene(height=32, width=40)))

    assert out.frame.shape == (1, 1, 3, 32, 40)


@pytest.mark.parametrize(
    ("height", "width"), [(HEIGHT + 1, WIDTH), (HEIGHT, WIDTH + 1)]
)
def test_odd_dimensions_raise(height: int, width: int):
    data = _rgb(_scene(height=height, width=width))

    with pytest.raises(ValueError, match="even frame dimensions"):
        H264Match()(data)


def test_rejects_non_rgb_batched_data():
    depth = BatchedDepthData(
        frame=torch.ones((1, 1, 1, HEIGHT, WIDTH), dtype=torch.float32),
        extrinsics=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, 1, 3, 3), dtype=torch.float32),
    )

    with pytest.raises(TypeError, match="Unsupported batched data type"):
        H264Match()(depth)


def test_allowed_data_types_excludes_depth():
    """Depth keeps lossless storage whatever codec is selected."""
    assert H264Match.allowed_data_types() == frozenset({DataType.RGB_IMAGES})

    with pytest.raises(ValueError, match="not allowed for data type"):
        validate_preprocessing_configuration(
            PreprocessingConfiguration({DataType.DEPTH_IMAGES: [H264Match()]})
        )


def test_serialization_round_trip():
    """Guards the getattr-by-parameter-name contract in PreprocessingMethod.to_dict."""
    method = H264Match(crf="18", preset="fast")

    serialized = method.to_dict()
    assert serialized == {
        "_target_": "neuracore.ml.preprocessing.methods.h264_match.H264Match",
        "crf": "18",
        "preset": "fast",
    }

    restored = instantiate(serialized)
    assert isinstance(restored, H264Match)
    assert restored.crf == "18"
    assert restored.preset == "fast"


def test_shipped_training_config_does_not_reference_this_method():
    """Regression guard against double-degradation.

    The same PreprocessingConfiguration is applied by the training dataloader and at
    inference. Listing H264Match in the shipped config would put it in the training
    path, degrading frames that a lossy recording already degraded once. Injection
    belongs in PolicyInference, which only runs at inference.
    """
    config_path = Path(neuracore.ml.__file__).parent / "config" / "config.yaml"
    preprocessing = OmegaConf.load(config_path).preprocessing

    assert "H264Match" not in OmegaConf.to_yaml(preprocessing)


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires the ffmpeg binary")
def test_matches_the_daemon_encoders_colour_pipeline():
    """Compare against the daemon's literal lossy encode of the same clip.

    Quantization cannot match exactly -- the daemon's `-preset medium` uses lookahead
    and mbtree, which need frames this method will never see. What must match is the
    colour pipeline: getting the BT.601/BT.709 matrix or the limited/full range
    conversion wrong would shift every pixel and collapse PSNR to roughly 20dB.
    """
    height, width, frame_count = 48, 64, 6
    frames = [_scene(height, width, shift=i) for i in range(frame_count)]

    method = H264Match()
    ours = [
        method(_rgb(frame)).frame[0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
        for frame in frames
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "in.rgb"
        mp4_path = Path(temp_dir) / "out.mp4"
        raw_path.write_bytes(
            b"".join(frame.astype(np.uint8).tobytes() for frame in frames)
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                "30",
                "-i",
                str(raw_path),
                # Matches the lossy-only pass in
                # rust/data_daemon/src/encoding/video_encoder.rs
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "23",
                str(mp4_path),
            ],
            check=True,
        )
        with av.open(str(mp4_path)) as container:
            reference = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]

    assert len(reference) == frame_count
    for index, (mine, theirs) in enumerate(zip(ours, reference)):
        assert _psnr(mine, theirs) > 30.0, f"frame {index} diverges from the daemon"
