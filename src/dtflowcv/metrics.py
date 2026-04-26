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


COCO_IOU_THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05))
COCO_AREA_RANGES = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, float("inf")),
}


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
    valid_targets = [target for target in targets if _is_valid_class_id(target.class_id, class_count)]
    invalid_targets = [target for target in targets if not _is_valid_class_id(target.class_id, class_count)]
    valid_predictions = [
        prediction for prediction in predictions if _is_valid_class_id(prediction.class_id, class_count)
    ]
    unknown_predictions = [
        prediction for prediction in predictions if not _is_valid_class_id(prediction.class_id, class_count)
    ]

    per_class: dict[int, float | None] = {}
    per_class_detail: dict[int, dict[str, object]] = {}
    false_positive_count = len(unknown_predictions)
    false_negative_count = 0

    # Group by class once
    targets_by_class: dict[int, list[DetectionTarget]] = defaultdict(list)
    predictions_by_class: dict[int, list[DetectionPrediction]] = defaultdict(list)
    for target in valid_targets:
        targets_by_class[target.class_id].append(target)
    for prediction in valid_predictions:
        predictions_by_class[prediction.class_id].append(prediction)

    for class_id in range(class_count):
        class_targets = targets_by_class[class_id]
        class_predictions = sorted(predictions_by_class[class_id], key=lambda item: item.score, reverse=True)
        if not class_targets:
            per_class[class_id] = None
            per_class_detail[class_id] = {
                "ap": None, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": len(class_predictions), "fn": 0,
                "target_count": 0, "prediction_count": len(class_predictions),
            }
            false_positive_count += len(class_predictions)
            continue

        # Index targets by image_id for O(1) lookup
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
            for index, target in targets_by_image.get(prediction.image_id, []):
                if index in matched:
                    continue
                iou_val = box_iou(target.box_xyxy, prediction.box_xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_index = index
            if best_iou >= iou_threshold and best_index >= 0:
                matched.add(best_index)
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        class_tp = int(np.sum(tp))
        class_fp = int(np.sum(fp))
        class_fn = max(len(class_targets) - len(matched), 0)

        false_positive_count += class_fp
        false_negative_count += class_fn

        ap = _average_precision_np(tp, fp, len(class_targets))
        per_class[class_id] = ap

        # Precision, recall, F1
        precision = class_tp / max(class_tp + class_fp, 1)
        recall = class_tp / max(class_tp + class_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class_detail[class_id] = {
            "ap": ap,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "target_count": len(class_targets),
            "prediction_count": len(class_predictions),
        }

    valid_ap = [ap for ap in per_class.values() if ap is not None]
    map_value = float(np.mean(valid_ap)) if valid_ap else 0.0

    # Rank classes by AP (hardest first)
    hardest = sorted(
        [(cid, detail) for cid, detail in per_class_detail.items() if detail["ap"] is not None],
        key=lambda x: x[1]["ap"],
    )

    return {
        "map": map_value,
        "iou_threshold": iou_threshold,
        "class_ap": {str(class_id): ap for class_id, ap in per_class.items()},
        "class_detail": {str(class_id): detail for class_id, detail in per_class_detail.items()},
        "hardest_classes": [{"class_id": cid, **d} for cid, d in hardest[:5]],
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "target_count": len(valid_targets),
        "prediction_count": len(predictions),
        "evaluated_prediction_count": len(valid_predictions),
        "invalid_target_class_id_count": len(invalid_targets),
        "unknown_prediction_fp": len(unknown_predictions),
    }


def coco_style_metrics(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    class_count: int,
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
) -> dict[str, object]:
    """Compute COCO-style AP/AR without requiring pycocotools.

    This is an internal deterministic evaluator over this repo's DetectionTarget/
    DetectionPrediction contract. It is not a pycocotools parity claim.
    """
    ap_by_iou: dict[str, float] = {}
    for threshold in iou_thresholds:
        metric = map_at_iou(targets, predictions, class_count, threshold)
        ap_by_iou[f"{threshold:.2f}"] = float(metric["map"])

    ap_values = list(ap_by_iou.values())
    area_ap: dict[str, float | None] = {}
    for name, area_range in COCO_AREA_RANGES.items():
        area_targets = [target for target in targets if _box_area_in_range(target.box_xyxy, area_range)]
        if not area_targets:
            area_ap[name] = None
            continue
        area_predictions = [
            prediction for prediction in predictions if _box_area_in_range(prediction.box_xyxy, area_range)
        ]
        area_scores = [
            float(map_at_iou(area_targets, area_predictions, class_count, threshold)["map"])
            for threshold in iou_thresholds
        ]
        area_ap[name] = float(np.mean(area_scores)) if area_scores else None

    return {
        "ap50_95": float(np.mean(ap_values)) if ap_values else 0.0,
        "ap50": ap_by_iou.get("0.50", 0.0),
        "ap75": ap_by_iou.get("0.75", 0.0),
        "ap_by_iou": ap_by_iou,
        "ap_small": area_ap["small"],
        "ap_medium": area_ap["medium"],
        "ap_large": area_ap["large"],
        "ar_1": average_recall(targets, predictions, class_count, iou_thresholds, max_detections=1),
        "ar_10": average_recall(targets, predictions, class_count, iou_thresholds, max_detections=10),
        "ar_100": average_recall(targets, predictions, class_count, iou_thresholds, max_detections=100),
        "iou_thresholds": list(iou_thresholds),
        "claim_boundary": "COCO-style internal metrics; pycocotools parity is not claimed unless separately verified.",
    }


def average_recall(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    class_count: int,
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
    max_detections: int = 100,
) -> float:
    recalls: list[float] = []
    valid_targets = [target for target in targets if _is_valid_class_id(target.class_id, class_count)]
    valid_predictions = [
        prediction for prediction in predictions if _is_valid_class_id(prediction.class_id, class_count)
    ]
    for threshold in iou_thresholds:
        matched_count = 0
        for class_id in range(class_count):
            class_targets = [target for target in valid_targets if target.class_id == class_id]
            if not class_targets:
                continue
            class_predictions = [pred for pred in valid_predictions if pred.class_id == class_id]
            matched_count += _matched_target_count(class_targets, class_predictions, threshold, max_detections)
        recalls.append(matched_count / max(len(valid_targets), 1))
    return float(np.mean(recalls)) if recalls else 0.0


def confusion_matrix(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    class_count: int,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Build NxN confusion matrix + background row/column.

    Matrix[i][j] = count of GT class i predicted as class j.
    Matrix[i][class_count] = GT class i missed (FN).
    Matrix[class_count][j] = FP predicted as class j with no GT match.
    Matrix[*][class_count + 1] and Matrix[class_count + 1][*] are out-of-schema class IDs.
    """
    # valid classes + background + unknown schema bucket
    background_idx = class_count
    unknown_idx = class_count + 1
    n = class_count + 2
    matrix = np.zeros((n, n), dtype=np.int32)

    # Group by image
    targets_by_image: dict[str, list[tuple[int, DetectionTarget]]] = defaultdict(list)
    preds_by_image: dict[str, list[DetectionPrediction]] = defaultdict(list)
    for idx, t in enumerate(targets):
        targets_by_image[t.image_id].append((idx, t))
    for p in predictions:
        preds_by_image[p.image_id].append(p)

    all_image_ids = set(targets_by_image.keys()) | set(preds_by_image.keys())

    for image_id in all_image_ids:
        img_targets = targets_by_image.get(image_id, [])
        img_preds = sorted(preds_by_image.get(image_id, []), key=lambda p: p.score, reverse=True)
        targets_by_idx = {idx: target for idx, target in img_targets}

        matched_gt: set[int] = set()

        for pred in img_preds:
            best_idx = -1
            best_iou = 0.0
            for idx, gt in img_targets:
                if idx in matched_gt:
                    continue
                iou_val = box_iou(gt.box_xyxy, pred.box_xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx >= 0:
                gt_class = _matrix_class_index(targets_by_idx[best_idx].class_id, class_count, unknown_idx)
                pred_class = _matrix_class_index(pred.class_id, class_count, unknown_idx)
                matched_gt.add(best_idx)
                # GT=gt_class, Pred=pred.class_id
                matrix[gt_class][pred_class] += 1
            else:
                # FP: no GT match → background row
                pred_class = _matrix_class_index(pred.class_id, class_count, unknown_idx)
                matrix[background_idx][pred_class] += 1

        # Unmatched GTs → FN (predicted as background)
        for idx, gt in img_targets:
            if idx not in matched_gt:
                gt_class = _matrix_class_index(gt.class_id, class_count, unknown_idx)
                matrix[gt_class][background_idx] += 1

    return {
        "matrix": matrix.tolist(),
        "size": n,
        "class_count": class_count,
        "background_index": background_idx,
        "unknown_index": unknown_idx,
        "note": (
            "matrix[i][j] = GT class i predicted as class j. "
            "Background index is missed/no-GT. Unknown index is out-of-schema class id."
        ),
    }


def precision_recall_curve(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    class_id: int,
    iou_threshold: float = 0.5,
    n_points: int = 101,
) -> dict[str, object]:
    """Compute precision-recall curve for a single class.

    Returns sampled at n_points recall levels.
    """
    class_targets = [t for t in targets if t.class_id == class_id]
    class_preds = sorted([p for p in predictions if p.class_id == class_id], key=lambda p: p.score, reverse=True)

    if not class_targets:
        return {"recall": [], "precision": [], "class_id": class_id, "target_count": 0}

    targets_by_image: dict[str, list[tuple[int, DetectionTarget]]] = defaultdict(list)
    for idx, t in enumerate(class_targets):
        targets_by_image[t.image_id].append((idx, t))

    matched: set[int] = set()
    tp_arr = np.zeros(len(class_preds), dtype=np.int32)
    fp_arr = np.zeros(len(class_preds), dtype=np.int32)

    for pi, pred in enumerate(class_preds):
        best_idx = -1
        best_iou = 0.0
        for idx, t in targets_by_image.get(pred.image_id, []):
            if idx in matched:
                continue
            iou_val = box_iou(t.box_xyxy, pred.box_xyxy)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add(best_idx)
            tp_arr[pi] = 1
        else:
            fp_arr[pi] = 1

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    recalls = cum_tp / len(class_targets)
    precisions = cum_tp / (cum_tp + cum_fp)

    # Interpolate at n_points
    recall_levels = np.linspace(0, 1, n_points)
    interp_precision = np.zeros(n_points)
    for i, r in enumerate(recall_levels):
        mask = recalls >= r
        if mask.any():
            interp_precision[i] = float(np.max(precisions[mask]))

    return {
        "recall": recall_levels.tolist(),
        "precision": interp_precision.tolist(),
        "class_id": class_id,
        "target_count": len(class_targets),
        "prediction_count": len(class_preds),
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


def _matched_target_count(
    targets: list[DetectionTarget],
    predictions: list[DetectionPrediction],
    iou_threshold: float,
    max_detections: int,
) -> int:
    targets_by_image: dict[str, list[tuple[int, DetectionTarget]]] = defaultdict(list)
    predictions_by_image: dict[str, list[DetectionPrediction]] = defaultdict(list)
    for idx, target in enumerate(targets):
        targets_by_image[target.image_id].append((idx, target))
    for prediction in predictions:
        predictions_by_image[prediction.image_id].append(prediction)

    matched: set[int] = set()
    for image_id, image_targets in targets_by_image.items():
        image_predictions = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda item: item.score,
            reverse=True,
        )[:max_detections]
        for prediction in image_predictions:
            best_index = -1
            best_iou = 0.0
            for index, target in image_targets:
                if index in matched:
                    continue
                iou_value = box_iou(target.box_xyxy, prediction.box_xyxy)
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_index = index
            if best_iou >= iou_threshold and best_index >= 0:
                matched.add(best_index)
    return len(matched)


def _box_area_in_range(
    box_xyxy: tuple[float, float, float, float],
    area_range: tuple[float, float],
) -> bool:
    x1, y1, x2, y2 = box_xyxy
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    low, high = area_range
    return low <= area < high


def _is_valid_class_id(class_id: int, class_count: int) -> bool:
    return 0 <= class_id < class_count


def _matrix_class_index(class_id: int, class_count: int, unknown_idx: int) -> int:
    if _is_valid_class_id(class_id, class_count):
        return class_id
    return unknown_idx
