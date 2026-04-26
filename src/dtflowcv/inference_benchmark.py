from __future__ import annotations

import hashlib
import importlib.metadata
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

from dtflowcv.config import load_yaml
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

    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    samples = [str(path) for path in image_paths]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx in range(max(warmup, 0)):
        model.predict(samples[idx % len(samples)], imgsz=image_size, device=device, verbose=False)

    preprocess_ms: list[float] = []
    model_ms: list[float] = []
    postprocess_ms: list[float] = []
    end_to_end_ms: list[float] = []

    for idx in range(max(runs, 1)):
        image = samples[idx % len(samples)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(image, imgsz=image_size, device=device, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        speed = getattr(result, "speed", {}) or {}
        preprocess_ms.append(float(speed.get("preprocess", 0.0)))
        model_ms.append(float(speed.get("inference", elapsed_ms)))
        postprocess_ms.append(float(speed.get("postprocess", 0.0)))
        end_to_end_ms.append(elapsed_ms)

    total_seconds = sum(end_to_end_ms) / 1000.0
    model_file = Path(model_path)
    return {
        "status": "ok",
        "benchmark_id": _benchmark_id(images, model_path, problem_path),
        "git_commit": _git_commit(),
        "images": str(images),
        "problem": str(problem_path) if problem_path is not None else None,
        "model": str(model_path),
        "model_sha256": _file_sha256(model_file) if model_file.exists() else None,
        "dataset_sha256": _dataset_sha256(image_paths),
        "class_schema_sha256": _class_schema_sha256(problem_path),
        "image_count": len(image_paths),
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
        "memory": {
            "max_rss_mb": _max_rss_mb(),
            "cuda_max_allocated_mb": _cuda_max_allocated_mb(torch),
        },
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


def _benchmark_id(
    images: str | Path,
    model_path: str | Path,
    problem_path: str | Path | None,
) -> str:
    payload = {
        "images": str(images),
        "model": str(model_path),
        "problem": str(problem_path) if problem_path is not None else None,
        "git_commit": _git_commit(),
        "timestamp_floor": int(time.time() // 3600),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(image_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for image_path in sorted(image_paths, key=lambda path: str(path)):
        digest.update(str(image_path).encode("utf-8"))
        digest.update(_file_sha256(image_path).encode("utf-8"))
    return digest.hexdigest()


def _class_schema_sha256(problem_path: str | Path | None) -> str | None:
    if problem_path is None:
        return None
    path = Path(problem_path)
    if not path.exists():
        return None
    problem = load_yaml(path)
    classes = problem.get("classes", [])
    payload = json.dumps(classes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(float(usage.ru_maxrss) / 1024.0, 3)


def _cuda_max_allocated_mb(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return round(float(torch_module.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 3)
