"""Preprocessing method implementations."""

from ..base import PreprocessingMethod
from .color_jitter import ColorJitter
from .gaussian_noise import GaussianNoise
from .resize_pad import ResizePad

# Preprocessing configs support only the methods defined here.
PREPROCESSING_METHODS: dict[str, type[PreprocessingMethod]] = {
    method_class.name: method_class
    for method_class in (ResizePad, ColorJitter, GaussianNoise)
}

__all__ = ["ColorJitter", "GaussianNoise", "PREPROCESSING_METHODS", "ResizePad"]
