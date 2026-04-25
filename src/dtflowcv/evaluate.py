from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dtflowcv.metrics import (
    DetectionPrediction,
    DetectionTarget,
    box_iou_matrix_np,
    confusion_matrix,
    map_at_iou,
    precision_recall_curve,
)
from dtflowcv.yolo import iter_images, parse_yolo_label_file, related_label_path, yolo_to_xyxy


def evaluate_yolo_predictions(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    class_count: int,
    iou_threshold: float = 0.5,
    image_manifest: str | Path | None = None,
    max_images: int | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate YOLO predictions with per-class depth.

    Returns mAP, per-class AP/precision/recall/F1, confusion matrix, hardest classes.
    """
    targets: list[DetectionTarget] = []
    predictions: list[DetectionPrediction] = []
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    image_paths = _selected_images(images_root, image_manifest, max_images)

    # Cache image dimensions
    _dim_cache: dict[str, tuple[int, int]] = {}

    for image_path in image_paths:
        image_id = image_path.stem
        cache_key = str(image_path)
        dims = _dim_cache.get(cache_key)
        if dims is None:
            with Image.open(image_path) as image:
                dims = image.size
            _dim_cache[cache_key] = dims
        width, height = dims

        label_path = related_label_path(image_path, images_root, labels_root)
        for box in parse_yolo_label_file(label_path):
            targets.append(DetectionTarget(image_id, box.class_id, yolo_to_xyxy(box, width, height)))

        pred_path = related_label_path(image_path, images_root, preds_root)
        for box in parse_yolo_label_file(pred_path, with_confidence=True):
            predictions.append(
                DetectionPrediction(
                    image_id=image_id,
                    class_id=box.class_id,
                    box_xyxy=yolo_to_xyxy(box, width, height),
                    score=float(box.confidence if box.confidence is not None else 0.0),
                )
            )

    result = map_at_iou(targets, predictions, class_count, iou_threshold)
    result["metric_name"] = f"mAP@{iou_threshold:.2f}"
    result["image_count"] = len(image_paths)

    # Confusion matrix
    cm = confusion_matrix(targets, predictions, class_count, iou_threshold)
    result["confusion_matrix"] = cm

    # Per-class names mapping
    if class_names is not None:
        named_detail: dict[str, Any] = {}
        for cid_str, detail in result.get("class_detail", {}).items():
            cid = int(cid_str)
            name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            named_detail[name] = detail
        result["class_detail_named"] = named_detail

        # Named hardest classes
        hardest = result.get("hardest_classes", [])
        for entry in hardest:
            cid = entry.get("class_id", 0)
            entry["class_name"] = class_names[cid] if cid < len(class_names) else f"class_{cid}"

    return result


def evaluate_per_class_pr_curves(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    class_count: int,
    iou_threshold: float = 0.5,
    image_manifest: str | Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Compute precision-recall curves for every class."""
    targets: list[DetectionTarget] = []
    predictions: list[DetectionPrediction] = []
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    image_paths = _selected_images(images_root, image_manifest, max_images)

    for image_path in image_paths:
        image_id = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = related_label_path(image_path, images_root, labels_root)
        for box in parse_yolo_label_file(label_path):
            targets.append(DetectionTarget(image_id, box.class_id, yolo_to_xyxy(box, width, height)))

        pred_path = related_label_path(image_path, images_root, preds_root)
        for box in parse_yolo_label_file(pred_path, with_confidence=True):
            predictions.append(
                DetectionPrediction(
                    image_id=image_id,
                    class_id=box.class_id,
                    box_xyxy=yolo_to_xyxy(box, width, height),
                    score=float(box.confidence if box.confidence is not None else 0.0),
                )
            )

    curves = {}
    for cid in range(class_count):
        curve = precision_recall_curve(targets, predictions, cid, iou_threshold)
        curves[str(cid)] = curve

    return {"iou_threshold": iou_threshold, "curves": curves}


def _selected_images(images_root: Path, image_manifest: str | Path | None, max_images: int | None) -> list[Path]:
    if image_manifest is None:
        image_paths = iter_images(images_root)
    else:
        image_paths = [Path(line.strip()) for line in Path(image_manifest).read_text(encoding="utf-8").splitlines()]
        image_paths = [path for path in image_paths if str(path).strip()]
    if max_images is not None:
        return image_paths[:max_images]
    return image_paths
