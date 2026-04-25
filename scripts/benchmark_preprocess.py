from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import dtflowcv_native
except ImportError:
    dtflowcv_native = None


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit]


def cv2_decode_resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR))


def numpy_pipeline(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2_decode_resize_rgb(path, size)
    arr = image.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def native_pipeline(path: Path, size: tuple[int, int]) -> np.ndarray:
    if dtflowcv_native is None:
        raise RuntimeError("dtflowcv_native is not importable")
    image = cv2_decode_resize_rgb(path, size)
    return np.asarray(dtflowcv_native.normalize_hwc_u8_to_chw_f32(image, MEAN.tolist(), STD.tolist()))


def benchmark(paths: list[Path], fn, size: tuple[int, int], repeats: int) -> dict[str, float | int]:
    samples: list[float] = []
    for _ in range(repeats):
        for path in paths:
            start = time.perf_counter()
            out = fn(path, size)
            if out.shape != (3, size[1], size[0]):
                raise RuntimeError(f"unexpected output shape {out.shape}")
            samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "fps": len(ordered) / (sum(ordered) / 1000.0),
    }


def percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    weight = idx - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/preprocess_benchmark.json"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    paths = iter_images(args.images, args.limit)
    if not paths:
        raise ValueError(f"no images found under {args.images}")
    size = (args.width, args.height)
    payload = {
        "image_root": str(args.images),
        "image_count": len(paths),
        "repeats": args.repeats,
        "resize": {"width": args.width, "height": args.height},
        "native_importable": dtflowcv_native is not None,
        "numpy_cv2_pipeline": benchmark(paths, numpy_pipeline, size, args.repeats),
    }
    if dtflowcv_native is not None:
        payload["native_cv2_pipeline"] = benchmark(paths, native_pipeline, size, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
