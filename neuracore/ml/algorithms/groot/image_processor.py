"""Image and language preprocessing for the GR00T N1.6 model.

Handles SigLip2 image normalization (model constants, not dataset statistics)
and construction of VLM input tokens from Neuracore batch data. Also provides
language token extraction from pre-tokenized batch data.

The image preprocessing uses fixed constants from the SigLip2 vision encoder:
  - Resolution: 448 x 448
  - Normalization: mean=0.5, std=0.5 (maps [0,1] -> [-1,1])

These are model-level constants determined during VLM pretraining and are
the same for all robots and datasets.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
import torch.nn.functional as F
from neuracore_types import BatchedLanguageData, BatchedRGBData

logger = logging.getLogger(__name__)

# Target resolution for the SigLip2 vision encoder
SIGLIP2_IMAGE_SIZE = 448

# ImageNet-standard normalization for SigLip2:
# pixel_value = (pixel_value - mean) / std
# Maps [0, 1] range to [-1, 1] range
SIGLIP2_IMAGE_MEAN = [0.5, 0.5, 0.5]
SIGLIP2_IMAGE_STD = [0.5, 0.5, 0.5]

# Number of visual tokens per image after SigLip2 + downsampling
# SigLip2: (448/14)^2 = 1024 patches, with downsample_ratio=0.5: 1024 * 0.25 = 256
NUM_IMAGE_TOKENS_PER_IMAGE = 256


class GrootImageProcessor:
    """Preprocessor for GR00T N1.6 visual and language inputs.

    Handles two responsibilities:
      1. Image preprocessing: resize RGB frames to 448x448 and normalize
         with SigLip2 model constants (not dataset statistics).
      2. VLM input construction: build input_ids with image token placeholders
         and extract pre-tokenized language data from the batch.

    The processor is stateless and uses fixed model constants. The
    ``from_pretrained`` factory loads the image_token_index from the
    model configuration.

    Args:
        image_token_index: Token ID used as placeholder for image features
            in the VLM input_ids sequence. Determined by the Eagle model
            config (default: 151669 for Qwen-based tokenizer).
    """

    def __init__(self, image_token_index: int = 151669):
        """Initialize the image processor with model constants."""
        self.image_token_index = image_token_index

        # Register normalization constants as tensors (will be moved to device)
        self._image_mean = torch.tensor(SIGLIP2_IMAGE_MEAN).view(1, 3, 1, 1)
        self._image_std = torch.tensor(SIGLIP2_IMAGE_STD).view(1, 3, 1, 1)

    @classmethod
    def from_pretrained(cls, model_path: str) -> GrootImageProcessor:
        """Create a processor from a GR00T N1.6 checkpoint.

        Reads the Eagle model config to determine the image_token_index.

        Args:
            model_path: Local path or HuggingFace model ID.

        Returns:
            GrootImageProcessor configured for the checkpoint.
        """
        from .utils import load_config_json

        config = load_config_json(model_path)

        # The image_token_index is stored in the Eagle backbone config
        # Default to the Qwen-based tokenizer's image token
        image_token_index = config.get("image_token_index", 151669)

        return cls(image_token_index=image_token_index)

    def preprocess_images(
        self,
        rgb_data_list: list[BatchedRGBData],
    ) -> torch.Tensor:
        """Preprocess RGB images for the SigLip2 vision encoder.

        Takes the last frame from each camera, resizes to 448x448, and
        normalizes with SigLip2 model constants. Multiple cameras are
        concatenated along the batch dimension.

        Args:
            rgb_data_list: List of BatchedRGBData, one per camera.
                Each has .frame of shape (B, T, 3, H, W).

        Returns:
            Preprocessed pixel values (B * num_cameras, 3, 448, 448).
        """
        all_frames = []
        for rgb_data in rgb_data_list:
            batched_rgb = cast(BatchedRGBData, rgb_data)
            # Take last frame from the temporal sequence
            last_frame = batched_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)

            # Resize to SigLip2 resolution using bilinear interpolation
            resized = F.interpolate(
                last_frame,
                size=(SIGLIP2_IMAGE_SIZE, SIGLIP2_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            # Normalize: (pixel - 0.5) / 0.5 -> maps [0,1] to [-1,1]
            mean = self._image_mean.to(device=resized.device, dtype=resized.dtype)
            std = self._image_std.to(device=resized.device, dtype=resized.dtype)
            normalized = (resized - mean) / std

            all_frames.append(normalized)

        # Stack cameras: (B * num_cameras, 3, 448, 448)
        return torch.cat(all_frames, dim=0)

    def construct_vlm_inputs(
        self,
        pixel_values: torch.Tensor,
        language_data: list[BatchedLanguageData] | None,
        batch_size: int,
        num_cameras: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct VLM input_ids and attention_mask.

        Builds the input sequence for the Eagle VLM by placing image token
        placeholders for each camera view, followed by the language tokens.
        The resulting input_ids tell the VLM where to inject visual features.

        Input format per sample:
            [<img_tok> x 256] x num_cameras + [language tokens]

        Args:
            pixel_values: Preprocessed images
                (B * num_cameras, 3, 448, 448).
            language_data: Pre-tokenized language data, or None for no text.
            batch_size: Number of samples in the batch.
            num_cameras: Number of camera views per sample.
            device: Target device for output tensors.

        Returns:
            Tuple of:
              - pixel_values: Images reshaped for the VLM
                  (B * num_cameras, 3, 448, 448).
              - input_ids: Combined token sequence of shape (B, L) where
                  L = 256 * num_cameras + L_text. The first 256 * num_cameras
                  positions are filled with image_token_index placeholders
                  (one block of 256 per camera); the remaining positions are
                  the pre-tokenized language prompt. Eagle3 VLM uses these
                  placeholder positions to inject the actual visual features
                  from pixel_values during its forward pass.
              - attention_mask: Attention mask (B, L).
        """
        # Number of image tokens for all cameras
        total_image_tokens = NUM_IMAGE_TOKENS_PER_IMAGE * num_cameras

        # Extract language tokens if available
        if language_data is not None and len(language_data) > 0:
            lang_data = cast(BatchedLanguageData, language_data[-1])
            # Take last timestep
            lang_tokens = lang_data.input_ids[:, -1, :]  # (B, L_text)
            lang_mask = lang_data.attention_mask[:, -1, :]  # (B, L_text)
            text_len = lang_tokens.shape[1]
        else:
            # No language input - create empty placeholder
            text_len = 1
            lang_tokens = torch.zeros(
                batch_size, text_len, dtype=torch.long, device=device
            )
            lang_mask = torch.zeros(
                batch_size, text_len, dtype=torch.long, device=device
            )

        # Build input_ids: [image_tokens_cam1, image_tokens_cam2, ..., text]
        total_len = total_image_tokens + text_len

        input_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(
            batch_size, total_len, dtype=torch.long, device=device
        )

        # Fill image token placeholders
        input_ids[:, :total_image_tokens] = self.image_token_index
        attention_mask[:, :total_image_tokens] = 1

        # Fill language tokens
        input_ids[:, total_image_tokens:] = lang_tokens
        attention_mask[:, total_image_tokens:] = lang_mask

        return pixel_values, input_ids, attention_mask

    def tokenize_language(
        self,
        language_data: list[BatchedLanguageData] | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract pre-tokenized language data from the batch.

        Neuracore's data pipeline provides pre-tokenized language. This
        method simply extracts and returns the tokens and masks from the
        last timestep.

        Args:
            language_data: Pre-tokenized language data, or None.
            batch_size: Number of samples in the batch.
            device: Target device for output tensors.

        Returns:
            Tuple of (input_ids, attention_mask), each (B, L).
        """
        if language_data is None or len(language_data) == 0:
            # Return empty tokens if no language input is provided
            return (
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            )

        lang_data = cast(BatchedLanguageData, language_data[-1])
        # Take last timestep from the temporal sequence
        input_ids = lang_data.input_ids[:, -1, :]  # (B, L)
        attention_mask = lang_data.attention_mask[:, -1, :]  # (B, L)
        return input_ids, attention_mask
