from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from dtflowcv.metrics import box_iou
from dtflowcv.yolo import iter_images, parse_yolo_label_file, yolo_to_xyxy


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
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    image_paths = _manifest_images(image_manifest) if image_manifest else iter_images(images_dir)
    for image_path in image_paths:
        image_id = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size

        targets = [
            {
                "class_id": box.class_id,
                "box_xyxy": yolo_to_xyxy(box, width, height),
                "area_ratio": box.area_ratio,
                "matched": False,
            }
            for box in parse_yolo_label_file(labels_root / f"{image_id}.txt")
        ]
        predictions = [
            {
                "class_id": box.class_id,
                "box_xyxy": yolo_to_xyxy(box, width, height),
                "score": float(box.confidence if box.confidence is not None else 0.0),
            }
            for box in parse_yolo_label_file(preds_root / f"{image_id}.txt", with_confidence=True)
        ]

        for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
            best_index, best_iou = _best_target(prediction["box_xyxy"], targets)
            if best_index >= 0 and best_iou >= iou_threshold:
                target = targets[best_index]
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
                        "predicted_class": _class_name(class_names, prediction["class_id"]),
                        "score": prediction["score"],
                        "best_iou": best_iou,
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
