from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from dtflowcv.config import load_yaml, write_json
from dtflowcv.specs import class_names, validate_problem_spec
from dtflowcv.yolo import iter_images


def predict_ultralytics_yolo(
    images: str | Path,
    problem_path: str | Path,
    model_path: str | Path = "yolov8n.pt",
    out_dir: str | Path = "artifacts/predictions",
    conf: float = 0.001,
    iou: float = 0.7,
    device: str | int | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    problem = load_yaml(problem_path)
    errors = validate_problem_spec(problem)
    if errors:
        return {
            "status": "blocked",
            "build_blockers": [f"invalid_problem_spec:{error}" for error in errors],
        }

    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "status": "blocked",
            "build_blockers": ["missing_python_module:ultralytics: install with python3 -m pip install -e '.[train]'"],
        }

    image_root = Path(images)
    image_paths = iter_images(image_root)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        return {"status": "blocked", "build_blockers": [f"no_images_found:{image_root}"]}

    model = YOLO(str(model_path))
    class_map = model_class_map(getattr(model, "names", {}), class_names(problem))
    output_root = Path(out_dir)
    written = 0
    skipped_non_target = 0
    prediction_count = 0

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        rows: list[str] = []
        result = model.predict(str(image_path), conf=conf, iou=iou, device=device, verbose=False)[0]
        for xyxy, model_class_id, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            strict=False,
        ):
            target_class_id = class_map.get(int(model_class_id))
            if target_class_id is None:
                skipped_non_target += 1
                continue
            line = yolo_prediction_line(target_class_id, xyxy, float(score), width, height)
            if line is None:
                skipped_non_target += 1
                continue
            rows.append(line)
            prediction_count += 1

        label_path = prediction_path_for_image(image_path, image_root, output_root)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
        written += 1

    summary = {
        "status": "ok",
        "images": str(image_root),
        "model": str(model_path),
        "out_dir": str(output_root),
        "image_count": len(image_paths),
        "prediction_file_count": written,
        "prediction_count": prediction_count,
        "skipped_non_target_prediction_count": skipped_non_target,
        "model_class_to_problem_class": {str(key): value for key, value in sorted(class_map.items())},
    }
    write_json(output_root / "predict_summary.json", summary)
    return summary


def model_class_map(
    model_names: dict[int, str] | dict[str, str] | list[str],
    target_names: list[str],
) -> dict[int, int]:
    target_by_name = {_normalize_name(name): idx for idx, name in enumerate(target_names)}
    mapped: dict[int, int] = {}
    items = enumerate(model_names) if isinstance(model_names, list) else model_names.items()
    for raw_id, raw_name in items:
        name = _normalize_name(str(raw_name))
        if name in target_by_name:
            mapped[int(raw_id)] = target_by_name[name]
    return mapped


def prediction_path_for_image(image_path: Path, image_root: Path, output_root: Path) -> Path:
    try:
        relative = image_path.relative_to(image_root)
    except ValueError:
        relative = Path(image_path.name)
    return output_root / relative.with_suffix(".txt")


def yolo_prediction_line(
    class_id: int,
    xyxy: Any,
    score: float,
    image_width: int,
    image_height: int,
) -> str | None:
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    box_w = max(x2 - x1, 0.0)
    box_h = max(y2 - y1, 0.0)
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    width = box_w / image_width
    height = box_h / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
