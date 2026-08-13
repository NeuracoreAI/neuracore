"""Re-encode inference frames to match lossy-recorded training data."""

from __future__ import annotations

import av
import numpy as np
import torch
from neuracore_types import BatchedNCData, BatchedRGBData, DataType

from ..base import PreprocessingMethod

# x264 buffers a frame internally unless zerolatency is tuned in, which would put a
# one-step lag in the control loop. zerolatency also implies bframes=0,
# rc-lookahead=0, sync-lookahead=0 and disables mbtree -- exactly the causal subset
# of `-preset medium` this method needs.
_CAUSAL_TUNE = "zerolatency"


class H264Match(PreprocessingMethod):
    """Put RGB frames through the codec the training data was recorded with.

    When a recording is made under ``nc.Codec.H264_MEDIUM`` the daemon writes a
    single ``libx264 -pix_fmt yuv420p -preset medium -crf 23`` video and training
    reads it, so the model learns on 4:2:0-subsampled, quantized pixels. Inference
    hands the model pristine frames from the sync point, so without this step the
    policy is served out-of-distribution input.

    An exact frame-by-frame reproduction is impossible, not merely expensive:
    ``-preset medium`` gives x264 ``rc-lookahead=40`` and ``mbtree=1``, making the
    quantizer for frame N a function of frames N+1..N+40 -- information that does not
    exist yet during a live rollout. This runs the same encoder made causal, which
    leaves:

    - exact: the RGB -> YUV 4:2:0 -> RGB colour matrix, range and chroma subsampling
    - close: quantization, at the same codec/pix_fmt/CRF/preset but without
      lookahead- and mbtree-driven bit allocation
    - preserved: inter-frame prediction, since the encoder is held across calls

    Depth is never a valid target: depth cameras keep their lossless storage whatever
    codec is selected, so their pixels are always bit-exact.

    Note:
        The encoder is stateful, so one instance must be used per camera. Frames are
        fed in ``(batch, time)`` order, which only forms a coherent video when a single
        camera's frames arrive in capture order -- the inference case, where the shape
        is always ``(1, 1, C, H, W)``.
    """

    def __init__(self, crf: str = "23", preset: str = "medium") -> None:
        """Initialize the encoder settings to match against.

        Args:
            crf: libx264 constant rate factor the training video was encoded at.
            preset: libx264 preset the training video was encoded at.
        """
        # Attribute names must match the parameter names: PreprocessingMethod.to_dict
        # recovers constructor arguments with getattr and silently writes None when a
        # name does not resolve.
        self.crf = crf
        self.preset = preset
        self._encoder: av.CodecContext | None = None
        self._decoder: av.CodecContext | None = None
        self._resolution: tuple[int, int] | None = None

    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        """Return data types supported by this method."""
        return frozenset({DataType.RGB_IMAGES})

    def _reset_codecs(self, height: int, width: int) -> None:
        """Build a fresh encoder/decoder pair for a given frame size.

        Args:
            height: Frame height in pixels.
            width: Frame width in pixels.

        Raises:
            ValueError: If either dimension is odd, which ``yuv420p`` cannot represent.
        """
        if height % 2 or width % 2:
            raise ValueError(
                f"h264_match needs even frame dimensions for yuv420p, got "
                f"{width}x{height}. The daemon's encoder has the same constraint, so "
                "training data recorded at this size could not have been lossy."
            )

        encoder = av.CodecContext.create("libx264", "w")
        encoder.width = width
        encoder.height = height
        encoder.pix_fmt = "yuv420p"
        # Single-threaded so a frame's output does not depend on how x264 happened to
        # slice the work, which keeps the degradation reproducible in tests.
        encoder.thread_count = 1
        encoder.options = {
            "crf": self.crf,
            "preset": self.preset,
            "tune": _CAUSAL_TUNE,
        }

        decoder = av.CodecContext.create("h264", "r")
        decoder.thread_count = 1

        self._encoder = encoder
        self._decoder = decoder
        self._resolution = (height, width)

    def _degrade(self, image: np.ndarray) -> np.ndarray:
        """Round-trip one HxWx3 uint8 RGB image through the codec.

        Args:
            image: Contiguous ``(H, W, 3)`` uint8 RGB array.

        Returns:
            The decoded ``(H, W, 3)`` uint8 RGB array.

        Raises:
            RuntimeError: If the codec did not yield exactly one frame for this input,
                which would mean it is buffering and returning a stale frame.
        """
        height, width = image.shape[:2]
        if self._resolution != (height, width):
            self._reset_codecs(height, width)
        assert self._encoder is not None and self._decoder is not None

        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        decoded = []
        for packet in self._encoder.encode(frame):
            decoded.extend(self._decoder.decode(packet))

        if len(decoded) != 1:
            raise RuntimeError(
                f"h264_match expected exactly one decoded frame per input frame, got "
                f"{len(decoded)}. The encoder is buffering, so frames would reach the "
                "policy out of step with the robot's state."
            )
        return decoded[0].to_ndarray(format="rgb24")

    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        """Degrade every frame in the batch to match the training encoding.

        Args:
            data: Batched RGB data whose ``frame`` is ``(B, T, C, H, W)`` float32 with
                values in 0-255.

        Returns:
            The same object, with ``frame`` replaced by the degraded frames.

        Raises:
            TypeError: If the data is not batched RGB data.
        """
        if not isinstance(data, BatchedRGBData):
            raise TypeError(
                f"Unsupported batched data type for h264_match: {type(data)!r}"
            )

        frame = data.frame
        device, dtype = frame.device, frame.dtype
        batch_size, time_steps = frame.shape[0], frame.shape[1]

        # to_ndarray hands back HWC, so go through HWC here too and permute once at
        # the end rather than round-tripping the layout per frame.
        as_uint8 = (
            frame.detach()
            .to("cpu", torch.float32)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
            .permute(0, 1, 3, 4, 2)
            .numpy()
        )

        degraded = np.empty_like(as_uint8)
        for batch_index in range(batch_size):
            for time_index in range(time_steps):
                degraded[batch_index, time_index] = self._degrade(
                    np.ascontiguousarray(as_uint8[batch_index, time_index])
                )

        data.frame = (
            torch.from_numpy(degraded)
            .permute(0, 1, 4, 2, 3)
            .to(device, dtype)
            .contiguous()
        )
        return data
