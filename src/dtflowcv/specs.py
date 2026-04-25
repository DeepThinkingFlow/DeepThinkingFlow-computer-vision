from __future__ import annotations

from typing import Any


REQUIRED_PHASE1_INPUT = [
    "modality",
    "color_space",
    "dtype",
    "average_resolution",
    "noise_profile",
    "lighting",
    "source",
]

REQUIRED_PHASE1_OUTPUT = [
    "type",
    "training_format",
    "prediction_format",
    "coordinate_system",
]

REQUIRED_METRICS = ["primary", "secondary", "acceptance"]


def validate_problem_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    project = spec.get("project", {})
    if project.get("task_type") != "detection":
        errors.append("project.task_type must be detection for this pipeline")

    phase_1 = spec.get("phase_1", {})
    input_spec = phase_1.get("input", {})
    output_spec = phase_1.get("output", {})
    metrics = phase_1.get("metrics", {})

    errors.extend(_missing("phase_1.input", input_spec, REQUIRED_PHASE1_INPUT))
    errors.extend(_missing("phase_1.output", output_spec, REQUIRED_PHASE1_OUTPUT))
    errors.extend(_missing("phase_1.metrics", metrics, REQUIRED_METRICS))

    if metrics.get("primary") != "map50":
        errors.append("phase_1.metrics.primary must be map50")

    acceptance = metrics.get("acceptance", {})
    for key in ("map50_min", "latency_p99_ms_max", "fps_min"):
        if key not in acceptance:
            errors.append(f"phase_1.metrics.acceptance.{key} is required")

    classes = spec.get("classes", [])
    if not isinstance(classes, list) or not classes:
        errors.append("classes must be a non-empty list")
    if len(classes) != len(set(classes)):
        errors.append("classes must not contain duplicates")

    dataset = spec.get("dataset", {})
    if not dataset.get("public_benchmark", {}).get("name"):
        errors.append("dataset.public_benchmark.name is required")
    split = dataset.get("split", {})
    split_sum = sum(float(split.get(name, 0.0)) for name in ("train", "val", "test"))
    if abs(split_sum - 1.0) > 1e-6:
        errors.append("dataset.split train/val/test ratios must sum to 1.0")

    return errors


def class_names(spec: dict[str, Any]) -> list[str]:
    names = spec.get("classes", [])
    if not isinstance(names, list):
        raise ValueError("classes must be a list")
    return [str(name) for name in names]


def split_ratios(spec: dict[str, Any]) -> dict[str, float]:
    split = spec.get("dataset", {}).get("split", {})
    return {
        "train": float(split.get("train", 0.70)),
        "val": float(split.get("val", 0.15)),
        "test": float(split.get("test", 0.15)),
    }


def _missing(prefix: str, payload: Any, required: list[str]) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{prefix} must be a mapping"]
    return [f"{prefix}.{key} is required" for key in required if _empty(payload.get(key))]


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False
