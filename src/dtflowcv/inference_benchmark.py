from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

from dtflowcv.deps import blocked_payload, missing_optional_blockers
from dtflowcv.yolo import iter_images


def benchmark_inference(
    images: str | Path,
    model_path: str | Path,
    *,
    problem_path: str | Path | None = None,
    device: str | int | None = "cpu",
    warmup: int = 5,
    runs: int = 30,
    batch: int = 1,
    image_size: int = 640,
    max_images: int | None = None,
) -> dict[str, Any]:
    blockers = missing_optional_blockers(["ultralytics", "torch"])
    if blockers:
        return blocked_payload(blockers)

    image_paths = iter_images(images)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        return blocked_payload([f"no_images_found:{Path(images)}"])
    if batch != 1:
        return blocked_payload(["unsupported_batch_size: only batch=1 is currently measured with per-image timing"])

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    samples = [str(path) for path in image_paths]

    for idx in range(max(warmup, 0)):
        model.predict(samples[idx % len(samples)], imgsz=image_size, device=device, verbose=False)

    preprocess_ms: list[float] = []
    model_ms: list[float] = []
    postprocess_ms: list[float] = []
    end_to_end_ms: list[float] = []

    for idx in range(max(runs, 1)):
        image = samples[idx % len(samples)]
        start = time.perf_counter()
        result = model.predict(image, imgsz=image_size, device=device, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        speed = getattr(result, "speed", {}) or {}
        preprocess_ms.append(float(speed.get("preprocess", 0.0)))
        model_ms.append(float(speed.get("inference", elapsed_ms)))
        postprocess_ms.append(float(speed.get("postprocess", 0.0)))
        end_to_end_ms.append(elapsed_ms)

    total_seconds = sum(end_to_end_ms) / 1000.0
    return {
        "status": "ok",
        "images": str(images),
        "problem": str(problem_path) if problem_path is not None else None,
        "model": str(model_path),
        "device": device,
        "image_size": image_size,
        "batch": batch,
        "warmup": warmup,
        "runs": runs,
        "preprocess_latency_ms": _latency(preprocess_ms),
        "model_latency_ms": _latency(model_ms),
        "postprocess_latency_ms": _latency(postprocess_ms),
        "end_to_end_latency_ms": _latency(end_to_end_ms),
        "fps": len(end_to_end_ms) / total_seconds if total_seconds > 0 else 0.0,
        "dependency_versions": _dependency_versions(["ultralytics", "torch", "opencv-python", "numpy"]),
        "claim_boundary": (
            "This measures Ultralytics runtime inference on this host; it is not a model-quality benchmark."
        ),
    }


def _latency(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "mean": sum(values) / max(len(values), 1),
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


def _dependency_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions
