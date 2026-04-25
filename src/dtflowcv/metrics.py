from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionTarget:
    image_id: str
    class_id: int
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionPrediction:
    image_id: str
    class_id: int
    box_xyxy: tuple[float, float, float, float]
    score: float


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def map_at_iou(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    class_count: int,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    per_class: dict[int, float | None] = {}
    false_positive_count = 0
    false_negative_count = 0

    targets_by_class: dict[int, list[DetectionTarget]] = defaultdict(list)
    predictions_by_class: dict[int, list[DetectionPrediction]] = defaultdict(list)
    for target in targets:
        targets_by_class[target.class_id].append(target)
    for prediction in predictions:
        predictions_by_class[prediction.class_id].append(prediction)

    for class_id in range(class_count):
        class_targets = targets_by_class[class_id]
        class_predictions = sorted(predictions_by_class[class_id], key=lambda item: item.score, reverse=True)
        if not class_targets:
            per_class[class_id] = None
            false_positive_count += len(class_predictions)
            continue

        matched: set[int] = set()
        tp: list[int] = []
        fp: list[int] = []
        for prediction in class_predictions:
            best_index = -1
            best_iou = 0.0
            for index, target in enumerate(class_targets):
                if index in matched or target.image_id != prediction.image_id:
                    continue
                iou = box_iou(target.box_xyxy, prediction.box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_iou >= iou_threshold and best_index >= 0:
                matched.add(best_index)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        false_positive_count += sum(fp)
        false_negative_count += max(len(class_targets) - len(matched), 0)
        per_class[class_id] = _average_precision(tp, fp, len(class_targets))

    valid_ap = [ap for ap in per_class.values() if ap is not None]
    map_value = float(sum(valid_ap) / len(valid_ap)) if valid_ap else 0.0
    return {
        "map": map_value,
        "iou_threshold": iou_threshold,
        "class_ap": {str(class_id): ap for class_id, ap in per_class.items()},
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "target_count": len(targets),
        "prediction_count": len(predictions),
    }


def _average_precision(tp: list[int], fp: list[int], target_count: int) -> float:
    if target_count == 0:
        return 0.0
    if not tp:
        return 0.0
    cum_tp: list[int] = []
    cum_fp: list[int] = []
    for idx in range(len(tp)):
        cum_tp.append(tp[idx] + (cum_tp[idx - 1] if idx else 0))
        cum_fp.append(fp[idx] + (cum_fp[idx - 1] if idx else 0))

    recalls = [value / target_count for value in cum_tp]
    precisions = [
        cum_tp[idx] / max(cum_tp[idx] + cum_fp[idx], 1)
        for idx in range(len(cum_tp))
    ]
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return float(ap)
