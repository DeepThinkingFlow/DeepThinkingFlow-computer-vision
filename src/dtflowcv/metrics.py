from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


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
    ix1 = ax1 if ax1 > bx1 else bx1
    iy1 = ay1 if ay1 > by1 else by1
    ix2 = ax2 if ax2 < bx2 else bx2
    iy2 = ay2 if ay2 < by2 else by2
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def box_iou_matrix_np(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Vectorized IoU: boxes_a (N,4), boxes_b (M,4) → (N,M) IoU matrix."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    a = boxes_a.astype(np.float32)
    b = boxes_b.astype(np.float32)

    # (N,1) vs (1,M) broadcasting
    ix1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    iy1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    ix2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    iy2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    inter = iw * ih

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, np.newaxis]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[np.newaxis, :]
    union = area_a + area_b - inter

    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def map_at_iou(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    class_count: int,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    per_class: dict[int, float | None] = {}
    false_positive_count = 0
    false_negative_count = 0

    # Group by class once
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

        # Index targets by image_id for O(1) lookup instead of linear scan
        targets_by_image: dict[str, list[tuple[int, DetectionTarget]]] = defaultdict(list)
        for idx, target in enumerate(class_targets):
            targets_by_image[target.image_id].append((idx, target))

        matched: set[int] = set()
        n_preds = len(class_predictions)
        tp = np.zeros(n_preds, dtype=np.int32)
        fp = np.zeros(n_preds, dtype=np.int32)

        for pred_idx, prediction in enumerate(class_predictions):
            best_index = -1
            best_iou = 0.0
            # Only check targets from same image
            for index, target in targets_by_image.get(prediction.image_id, []):
                if index in matched:
                    continue
                iou = box_iou(target.box_xyxy, prediction.box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_iou >= iou_threshold and best_index >= 0:
                matched.add(best_index)
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        false_positive_count += int(np.sum(fp))
        false_negative_count += max(len(class_targets) - len(matched), 0)
        per_class[class_id] = _average_precision_np(tp, fp, len(class_targets))

    valid_ap = [ap for ap in per_class.values() if ap is not None]
    map_value = float(np.mean(valid_ap)) if valid_ap else 0.0
    return {
        "map": map_value,
        "iou_threshold": iou_threshold,
        "class_ap": {str(class_id): ap for class_id, ap in per_class.items()},
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "target_count": len(targets),
        "prediction_count": len(predictions),
    }


def _average_precision_np(tp: np.ndarray, fp: np.ndarray, target_count: int) -> float:
    if target_count == 0 or tp.shape[0] == 0:
        return 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / target_count
    denom = cum_tp + cum_fp
    denom[denom == 0] = 1
    precisions = cum_tp / denom

    # Prepend/append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(mpre.shape[0] - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    change_mask = np.diff(mrec) != 0
    indices = np.where(change_mask)[0] + 1

    ap = float(np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices]))
    return ap
