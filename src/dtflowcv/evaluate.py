from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dtflowcv.metrics import DetectionPrediction, DetectionTarget, box_iou_matrix_np, map_at_iou
from dtflowcv.yolo import iter_images, parse_yolo_label_file, related_label_path, yolo_to_xyxy


def evaluate_yolo_predictions(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    class_count: int,
    iou_threshold: float = 0.5,
    image_manifest: str | Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    targets: list[DetectionTarget] = []
    predictions: list[DetectionPrediction] = []
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    image_paths = _selected_images(images_root, image_manifest, max_images)

    # Cache image dimensions to avoid reopening
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
    return result


def _selected_images(images_root: Path, image_manifest: str | Path | None, max_images: int | None) -> list[Path]:
    if image_manifest is None:
        image_paths = iter_images(images_root)
    else:
        image_paths = [Path(line.strip()) for line in Path(image_manifest).read_text(encoding="utf-8").splitlines()]
        image_paths = [path for path in image_paths if str(path).strip()]
    if max_images is not None:
        return image_paths[:max_images]
    return image_paths
