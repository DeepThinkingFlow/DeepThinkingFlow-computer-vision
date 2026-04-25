from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dtflowcv.metrics import box_iou, box_iou_matrix_np
from dtflowcv.yolo import iter_images, parse_yolo_label_file, related_label_path, yolo_to_xyxy


def export_detection_errors(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    class_names: list[str],
    iou_threshold: float = 0.5,
    small_area_ratio: float = 0.0001,
    image_manifest: str | Path | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    image_paths = _manifest_images(image_manifest) if image_manifest else iter_images(images_root)
    for image_path in image_paths:
        image_id = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = related_label_path(image_path, images_root, labels_root)
        pred_path = related_label_path(image_path, images_root, preds_root)

        raw_targets = parse_yolo_label_file(label_path)
        raw_preds = parse_yolo_label_file(pred_path, with_confidence=True)

        targets = [
            {
                "class_id": box.class_id,
                "box_xyxy": yolo_to_xyxy(box, width, height),
                "area_ratio": box.area_ratio,
                "matched": False,
            }
            for box in raw_targets
        ]
        predictions = [
            {
                "class_id": box.class_id,
                "box_xyxy": yolo_to_xyxy(box, width, height),
                "score": float(box.confidence if box.confidence is not None else 0.0),
            }
            for box in raw_preds
        ]

        # Vectorized IoU if enough boxes
        if targets and predictions:
            t_boxes = np.array([t["box_xyxy"] for t in targets], dtype=np.float32)
            p_boxes = np.array([p["box_xyxy"] for p in predictions], dtype=np.float32)
            iou_matrix = box_iou_matrix_np(p_boxes, t_boxes)  # (P, T)

            # Sort predictions by score descending
            pred_order = sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True)

            for pred_idx in pred_order:
                ious = iou_matrix[pred_idx]
                best_index = int(np.argmax(ious))
                best_iou = float(ious[best_index])

                if best_iou >= iou_threshold:
                    target = targets[best_index]
                    prediction = predictions[pred_idx]
                    if target["class_id"] == prediction["class_id"] and not target["matched"]:
                        target["matched"] = True
                        continue
                    kind = "class_confusion" if target["class_id"] != prediction["class_id"] else "duplicate_detection"
                    errors.append(
                        {
                            "image_id": image_id,
                            "kind": kind,
                            "predicted_class": _class_name(class_names, prediction["class_id"]),
                            "target_class": _class_name(class_names, target["class_id"]),
                            "score": prediction["score"],
                            "iou": best_iou,
                        }
                    )
                else:
                    errors.append(
                        {
                            "image_id": image_id,
                            "kind": "false_positive",
                            "predicted_class": _class_name(class_names, predictions[pred_idx]["class_id"]),
                            "score": predictions[pred_idx]["score"],
                            "best_iou": best_iou,
                        }
                    )
        elif predictions:
            for prediction in predictions:
                errors.append(
                    {
                        "image_id": image_id,
                        "kind": "false_positive",
                        "predicted_class": _class_name(class_names, prediction["class_id"]),
                        "score": prediction["score"],
                        "best_iou": 0.0,
                    }
                )

        for target in targets:
            if not target["matched"]:
                subtype = "small_object" if target["area_ratio"] < small_area_ratio else "missed_object"
                errors.append(
                    {
                        "image_id": image_id,
                        "kind": "false_negative",
                        "subtype": subtype,
                        "target_class": _class_name(class_names, target["class_id"]),
                        "area_ratio": target["area_ratio"],
                    }
                )

    by_kind = Counter(str(error["kind"]) for error in errors)
    by_class = Counter(
        str(error.get("target_class") or error.get("predicted_class"))
        for error in errors
    )
    return {
        "error_count": len(errors),
        "by_kind": dict(sorted(by_kind.items())),
        "by_class": dict(sorted(by_class.items())),
        "examples": errors,
    }


def _manifest_images(path: str | Path | None) -> list[Path]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8") as fh:
        return [Path(line.strip()) for line in fh if line.strip()]


def _best_target(box: tuple[float, float, float, float], targets: list[dict[str, Any]]) -> tuple[int, float]:
    best_index = -1
    best_iou = 0.0
    for index, target in enumerate(targets):
        iou = box_iou(box, target["box_xyxy"])
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_index, best_iou


def _class_name(class_names: list[str], class_id: int) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"unknown_{class_id}"
