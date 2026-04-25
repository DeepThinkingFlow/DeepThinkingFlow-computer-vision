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
        elif not _is_non_negative_number(acceptance[key]):
            errors.append(f"phase_1.metrics.acceptance.{key} must be a non-negative number")

    classes = spec.get("classes", [])
    if not isinstance(classes, list) or not classes:
        errors.append("classes must be a non-empty list")
    elif any(not isinstance(name, str) or not name.strip() for name in classes):
        errors.append("classes must contain non-empty strings")
    string_classes = isinstance(classes, list) and all(isinstance(name, str) for name in classes)
    if string_classes and len(classes) != len(set(classes)):
        errors.append("classes must not contain duplicates")

    dataset = spec.get("dataset", {})
    if not dataset.get("public_benchmark", {}).get("name"):
        errors.append("dataset.public_benchmark.name is required")
    split = dataset.get("split", {})
    split_values: dict[str, float] = {}
    for name in ("train", "val", "test"):
        try:
            split_values[name] = float(split.get(name, 0.0))
        except (TypeError, ValueError):
            errors.append(f"dataset.split.{name} must be numeric")
            split_values[name] = 0.0
        if split_values[name] < 0.0:
            errors.append(f"dataset.split.{name} must be non-negative")
    split_sum = sum(split_values.values())
    if abs(split_sum - 1.0) > 1e-6:
        errors.append("dataset.split train/val/test ratios must sum to 1.0")
    min_area = dataset.get("annotation_schema", {}).get("min_box_area_ratio", 0.0)
    if not _is_ratio(min_area):
        errors.append("dataset.annotation_schema.min_box_area_ratio must be in [0, 1]")

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
    return bool(isinstance(value, (list, dict)) and not value)


def _is_non_negative_number(value: Any) -> bool:
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def _is_ratio(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= number <= 1.0
