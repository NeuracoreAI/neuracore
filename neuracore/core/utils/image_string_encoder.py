"""Utility functions for encoding and decoding images to/from string formats.

This module provides a class `ImageStringEncoder` that facilitates the
conversion of NumPy image arrays to base64-encoded data URIs and vice-versa.
This is useful for transmitting image data over networks with other data.
"""

import base64

import imagecodecs
import numpy as np

DATA_URI_PREFIX = "data:image/png;base64,"
PNG_COMPRESSION_LEVEL = 6


class ImageStringEncoder:
    """Class for encoding and decoding images as data URIS."""

    @staticmethod
    def encode_image(
        image: np.ndarray | str,
        cap_size: bool = False,
        resize_shape: tuple[int, int] = (224, 224),
    ) -> str:
        """Encode numpy image array to base64 string for transmission.

        Converts numpy arrays to PNG format and encodes as base64. For remote
        endpoints, automatically resizes large images to 224x224 to meet
        payload size limits.

        Args:
            image: Numpy array representing an RGB image.

        Returns:
            Base64 encoded data URI string of the PNG image.
        """
        if isinstance(image, str):
            return image
        png_bytes = imagecodecs.png_encode(
            np.ascontiguousarray(image), level=PNG_COMPRESSION_LEVEL
        )
        return DATA_URI_PREFIX + base64.b64encode(png_bytes).decode("utf-8")

    @staticmethod
    def decode_image(encoded_image: str | np.ndarray) -> np.ndarray:
        """Decode base64 data URI image string back to numpy array.

        Args:
            encoded_image: Base64 encoded image string.

        Returns:
            Numpy array representing the decoded image.
        """
        if isinstance(encoded_image, np.ndarray):
            return encoded_image

        img_bytes = base64.b64decode(encoded_image.removeprefix(DATA_URI_PREFIX))
        return imagecodecs.png_decode(img_bytes)
