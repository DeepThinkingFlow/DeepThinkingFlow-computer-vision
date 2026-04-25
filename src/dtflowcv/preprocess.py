from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def load_rgb_resized(path: str | Path, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(size)
        return np.asarray(rgb, dtype=np.uint8)


def normalize_hwc_u8_to_chw_f32(
    image: np.ndarray,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have HWC shape with 3 RGB channels")
    arr = image.astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess_image(path: str | Path, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    return normalize_hwc_u8_to_chw_f32(load_rgb_resized(path, size))
