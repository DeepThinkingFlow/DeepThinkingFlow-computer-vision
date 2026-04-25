from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

# Try to import native kernel at module load
_native_available = False
_native_module = None
try:
    import dtflowcv_native as _native_module  # type: ignore

    _native_available = True
except ImportError:
    pass


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

    # Use native kernel if available
    if _native_available and _native_module is not None and image.flags["C_CONTIGUOUS"]:
        return np.asarray(
            _native_module.normalize_hwc_u8_to_chw_f32(
                image, list(mean), list(std)
            )
        )

    # Optimized numpy fallback: in-place operations, precomputed constants
    arr = image.astype(np.float32)
    inv_255_std = np.array([1.0 / (255.0 * s) for s in std], dtype=np.float32)
    bias = np.array([-m / s for m, s in zip(mean, std, strict=False)], dtype=np.float32)
    # Fused multiply-add: out = pixel * (1/(255*std)) + (-mean/std)
    np.multiply(arr, inv_255_std, out=arr)
    np.add(arr, bias, out=arr)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess_image(path: str | Path, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    return normalize_hwc_u8_to_chw_f32(load_rgb_resized(path, size))


def preprocess_batch(
    paths: list[str | Path],
    size: tuple[int, int] = (640, 640),
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> np.ndarray:
    """Preprocess multiple images into a (N, 3, H, W) float32 batch."""
    n = len(paths)
    if n == 0:
        return np.empty((0, 3, size[1], size[0]), dtype=np.float32)
    batch = np.empty((n, 3, size[1], size[0]), dtype=np.float32)
    for i, path in enumerate(paths):
        batch[i] = normalize_hwc_u8_to_chw_f32(load_rgb_resized(path, size), mean, std)
    return batch
