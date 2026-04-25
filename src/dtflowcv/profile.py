from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from dtflowcv.preprocess import preprocess_image
from dtflowcv.yolo import iter_images


def profile_preprocess(
    images_dir: str | Path,
    iterations: int = 3,
    size: tuple[int, int] = (640, 640),
    image_manifest: str | Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    image_paths = _selected_images(images_dir, image_manifest, max_images)
    if not image_paths:
        raise ValueError(f"no images found under {images_dir}")

    timings_ms: list[float] = []

    # Warmup iteration (not timed) to warm caches and JIT
    for image_path in image_paths:
        tensor = preprocess_image(image_path, size)
        if tensor.shape != (3, size[1], size[0]):
            raise RuntimeError(f"unexpected tensor shape {tensor.shape}")

    for _ in range(iterations):
        for image_path in image_paths:
            start = time.perf_counter()
            tensor = preprocess_image(image_path, size)
            if tensor.shape != (3, size[1], size[0]):
                raise RuntimeError(f"unexpected tensor shape {tensor.shape}")
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(timings_ms)
    total_seconds = sum(timings_ms) / 1000.0
    return {
        "status": "ok",
        "image_count": len(image_paths),
        "iterations": iterations,
        "sample_count": len(timings_ms),
        "resize": {"width": size[0], "height": size[1]},
        "latency_ms": {
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "mean": statistics.fmean(ordered),
        },
        "fps": len(timings_ms) / total_seconds if total_seconds > 0 else 0.0,
    }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    weight = idx - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _selected_images(
    images_dir: str | Path,
    image_manifest: str | Path | None,
    max_images: int | None,
) -> list[Path]:
    if image_manifest is None:
        paths = iter_images(images_dir)
    else:
        paths = [Path(line.strip()) for line in Path(image_manifest).read_text(encoding="utf-8").splitlines()]
        paths = [path for path in paths if str(path).strip()]
    if max_images is not None:
        return paths[:max_images]
    return paths
