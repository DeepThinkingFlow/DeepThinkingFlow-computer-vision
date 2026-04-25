from __future__ import annotations

from pathlib import Path
from typing import Any

from dtflowcv.config import load_yaml
from dtflowcv.evaluate import evaluate_yolo_predictions
from dtflowcv.profile import profile_preprocess
from dtflowcv.specs import class_names, validate_problem_spec


def benchmark_yolo_pipeline(
    images: str | Path,
    labels: str | Path,
    predictions: str | Path,
    problem_path: str | Path,
    iou_threshold: float = 0.5,
    profile_iterations: int = 3,
    profile_size: tuple[int, int] = (640, 640),
    image_manifest: str | Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    problem = load_yaml(problem_path)
    spec_errors = validate_problem_spec(problem)
    if spec_errors:
        return {
            "status": "blocked",
            "problem": str(problem_path),
            "build_blockers": [f"invalid_problem_spec:{error}" for error in spec_errors],
        }

    names = class_names(problem)
    detection = evaluate_yolo_predictions(
        images,
        labels,
        predictions,
        len(names),
        iou_threshold=iou_threshold,
        image_manifest=image_manifest,
        max_images=max_images,
    )
    preprocess = profile_preprocess(
        images,
        iterations=profile_iterations,
        size=profile_size,
        image_manifest=image_manifest,
        max_images=max_images,
    )
    acceptance = _acceptance(problem, detection, preprocess)
    return {
        "status": "passed" if acceptance["passed"] else "failed",
        "problem": str(problem_path),
        "images": str(images),
        "labels": str(labels),
        "predictions": str(predictions),
        "image_manifest": str(image_manifest) if image_manifest is not None else None,
        "max_images": max_images,
        "classes": names,
        "detection": detection,
        "preprocess": preprocess,
        "acceptance": acceptance,
        "claim_boundary": (
            "This is a prediction-file benchmark plus preprocessing micro-profile. It does not measure model "
            "forward pass, NMS, device transfer, or end-to-end inference latency. Use benchmark-inference for "
            "runtime latency/FPS on a model checkpoint."
        ),
    }


def _acceptance(problem: dict[str, Any], detection: dict[str, Any], preprocess: dict[str, Any]) -> dict[str, Any]:
    thresholds = problem.get("phase_1", {}).get("metrics", {}).get("acceptance", {})
    checks = {
        "map50_min": {
            "threshold": float(thresholds.get("map50_min", 0.0)),
            "actual": float(detection["map"]),
            "passed": float(detection["map"]) >= float(thresholds.get("map50_min", 0.0)),
        },
        "latency_p99_ms_max": {
            "threshold": float(thresholds.get("latency_p99_ms_max", float("inf"))),
            "actual": float(preprocess["latency_ms"]["p99"]),
            "passed": float(preprocess["latency_ms"]["p99"])
            <= float(thresholds.get("latency_p99_ms_max", float("inf"))),
        },
        "fps_min": {
            "threshold": float(thresholds.get("fps_min", 0.0)),
            "actual": float(preprocess["fps"]),
            "passed": float(preprocess["fps"]) >= float(thresholds.get("fps_min", 0.0)),
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
