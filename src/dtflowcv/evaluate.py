from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from dtflowcv.metrics import DetectionPrediction, DetectionTarget, map_at_iou
from dtflowcv.yolo import iter_images, label_path_for_image, parse_yolo_label_file, yolo_to_xyxy


def evaluate_yolo_predictions(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    class_count: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    targets: list[DetectionTarget] = []
    predictions: list[DetectionPrediction] = []
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    for image_path in iter_images(images_dir):
        image_id = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = labels_root / f"{image_id}.txt"
        if not label_path.exists():
            label_path = label_path_for_image(image_path, labels_root.parent)
        for box in parse_yolo_label_file(label_path):
            targets.append(DetectionTarget(image_id, box.class_id, yolo_to_xyxy(box, width, height)))

        pred_path = preds_root / f"{image_id}.txt"
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
    return result
