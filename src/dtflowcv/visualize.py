from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dtflowcv.metrics import box_iou
from dtflowcv.yolo import parse_yolo_label_file, related_label_path, yolo_to_xyxy, iter_images


# ── Color Palette ────────────────────────────────────────────

# Distinct colors per class (BGR for OpenCV, RGB for PIL)
CLASS_COLORS_RGB = [
    (0, 114, 189),    # person — blue
    (217, 83, 25),    # bicycle — orange
    (237, 177, 32),   # car — gold
    (126, 47, 142),   # motorcycle — purple
    (119, 172, 48),   # bus — green
    (77, 190, 238),   # truck — cyan
    (162, 20, 47),    # traffic_light — crimson
    (255, 127, 14),   # stop_sign — bright orange
    (44, 160, 44),    # class 8 — green
    (214, 39, 40),    # class 9 — red
    (148, 103, 189),  # class 10 — purple
    (140, 86, 75),    # class 11 — brown
]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS_RGB[class_id % len(CLASS_COLORS_RGB)]


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


# ── Drawing Functions (OpenCV) ───────────────────────────────

def draw_detections(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray | None = None,
    class_names: list[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw detection bounding boxes on image (BGR). Returns annotated copy."""
    import cv2

    out = image.copy()
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i]]
        cid = int(class_ids[i])
        color = _bgr(_color_for_class(cid))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = class_names[cid] if class_names and cid < len(class_names) else f"cls_{cid}"
        if scores is not None:
            label = f"{label} {scores[i]:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return out


def draw_errors(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float = 0.5,
    class_names: list[str] | None = None,
) -> np.ndarray:
    """Draw error analysis overlay: TP=green, FP=red, FN=dashed yellow."""
    import cv2

    out = image.copy()
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Match predictions to ground truth
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        from dtflowcv.metrics import box_iou_matrix_np
        iou_matrix = box_iou_matrix_np(pred_boxes.astype(np.float32), gt_boxes.astype(np.float32))

        order = np.argsort(-pred_scores)
        for pi in order:
            if pi in matched_pred:
                continue
            ious = iou_matrix[pi]
            best_gi = int(np.argmax(ious))
            if ious[best_gi] >= iou_threshold and best_gi not in matched_gt:
                if int(pred_classes[pi]) == int(gt_classes[best_gi]):
                    matched_gt.add(best_gi)
                    matched_pred.add(int(pi))

    # Draw TP (green)
    for pi in matched_pred:
        x1, y1, x2, y2 = [int(v) for v in pred_boxes[pi]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = "TP"
        if class_names and int(pred_classes[pi]) < len(class_names):
            label = f"TP:{class_names[int(pred_classes[pi])]}"
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    # Draw FP (red)
    for pi in range(len(pred_boxes)):
        if pi in matched_pred:
            continue
        x1, y1, x2, y2 = [int(v) for v in pred_boxes[pi]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
        label = f"FP {pred_scores[pi]:.2f}"
        if class_names and int(pred_classes[pi]) < len(class_names):
            label = f"FP:{class_names[int(pred_classes[pi])]} {pred_scores[pi]:.2f}"
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

    # Draw FN (yellow dashed)
    for gi in range(len(gt_boxes)):
        if gi in matched_gt:
            continue
        x1, y1, x2, y2 = [int(v) for v in gt_boxes[gi]]
        # Dashed rectangle
        _draw_dashed_rect(out, (x1, y1), (x2, y2), (0, 200, 255), 2, dash_length=8)
        label = "FN"
        if class_names and int(gt_classes[gi]) < len(class_names):
            label = f"FN:{class_names[int(gt_classes[gi])]}"
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    return out


def draw_tracking(
    image: np.ndarray,
    tracks: list[Any],
    class_names: list[str] | None = None,
    draw_trajectory: bool = True,
    trail_length: int = 30,
) -> np.ndarray:
    """Draw tracking info: boxes, IDs, trajectories, motion state."""
    import cv2

    out = image.copy()
    for track in tracks:
        bbox = track.last_bbox
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = _bgr(_color_for_class(track.class_id))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label with ID and motion state
        label = f"#{track.track_id}"
        if class_names and track.class_id < len(class_names):
            label = f"#{track.track_id} {class_names[track.class_id]}"
        motion_tag = "MOV" if track.motion.value == "moving" else "STA"
        label = f"{label} [{motion_tag}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Trajectory trail
        if draw_trajectory and len(track.trajectory) > 1:
            pts = track.trajectory[-trail_length:]
            for j in range(1, len(pts)):
                p1 = (int(pts[j - 1][0]), int(pts[j - 1][1]))
                p2 = (int(pts[j][0]), int(pts[j][1]))
                alpha = j / len(pts)
                thick = max(1, int(2 * alpha))
                cv2.line(out, p1, p2, color, thick)

    return out


def draw_line_zones(
    image: np.ndarray,
    lines: list[Any] | None = None,
    zones: list[Any] | None = None,
) -> np.ndarray:
    """Draw virtual lines and zone polygons on image."""
    import cv2

    out = image.copy()

    if lines:
        for lc in lines:
            cv2.line(out, lc.line_start, lc.line_end, (0, 255, 255), 2)
            mid = (
                (lc.line_start[0] + lc.line_end[0]) // 2,
                (lc.line_start[1] + lc.line_end[1]) // 2,
            )
            label = f"A>{lc.count_ab} B>{lc.count_ba}"
            cv2.putText(out, label, (mid[0] - 40, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if zones:
        for zone in zones:
            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], (0, 100, 200))
            cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
            cv2.polylines(out, [pts], True, (0, 200, 255), 2)
            cx = int(np.mean([p[0] for p in zone.polygon]))
            cy = int(np.mean([p[1] for p in zone.polygon]))
            cv2.putText(out, f"In:{zone.current_count}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out


# ── Visual Error Report ──────────────────────────────────────

def save_annotated_errors(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    iou_threshold: float = 0.5,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Generate error overlay images for each image in dataset.

    Saves annotated images showing TP/FP/FN boxes to output_dir.
    Returns summary statistics.
    """
    import cv2

    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(images_root)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    saved = 0

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = related_label_path(image_path, images_root, labels_root)
        pred_path = related_label_path(image_path, images_root, preds_root)

        gt_raw = parse_yolo_label_file(label_path)
        pred_raw = parse_yolo_label_file(pred_path, with_confidence=True)

        gt_boxes = np.array([yolo_to_xyxy(b, w, h) for b in gt_raw], dtype=np.float32).reshape(-1, 4)
        gt_classes = np.array([b.class_id for b in gt_raw], dtype=np.int32)
        pred_boxes = np.array([yolo_to_xyxy(b, w, h) for b in pred_raw], dtype=np.float32).reshape(-1, 4)
        pred_classes = np.array([b.class_id for b in pred_raw], dtype=np.int32)
        pred_scores = np.array(
            [float(b.confidence if b.confidence is not None else 0.0) for b in pred_raw],
            dtype=np.float32,
        )

        annotated = draw_errors(img, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores,
                                iou_threshold=iou_threshold, class_names=class_names)

        # Count TP/FP/FN for this image
        n_matched = _count_matches(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_threshold)
        total_tp += n_matched
        total_fp += len(pred_boxes) - n_matched
        total_fn += len(gt_boxes) - n_matched

        out_path = out_root / image_path.name
        cv2.imwrite(str(out_path), annotated)
        saved += 1

    return {
        "output_dir": str(out_root),
        "images_annotated": saved,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def make_error_grid(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    output_path: str | Path,
    class_names: list[str],
    grid_size: tuple[int, int] = (4, 4),
    cell_size: tuple[int, int] = (320, 320),
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Generate a contact sheet grid of worst errors (highest confidence FPs).

    Returns summary with output path.
    """
    import cv2

    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    preds_root = Path(predictions_dir)

    # Collect all FPs with scores
    error_cells: list[tuple[float, Path, np.ndarray, int]] = []

    for image_path in iter_images(images_root):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = related_label_path(image_path, images_root, labels_root)
        pred_path = related_label_path(image_path, images_root, preds_root)

        gt_raw = parse_yolo_label_file(label_path)
        pred_raw = parse_yolo_label_file(pred_path, with_confidence=True)

        gt_boxes = np.array([yolo_to_xyxy(b, w, h) for b in gt_raw], dtype=np.float32).reshape(-1, 4)
        gt_classes = np.array([b.class_id for b in gt_raw], dtype=np.int32)
        pred_boxes = np.array([yolo_to_xyxy(b, w, h) for b in pred_raw], dtype=np.float32).reshape(-1, 4)
        pred_scores = np.array(
            [float(b.confidence if b.confidence is not None else 0.0) for b in pred_raw],
            dtype=np.float32,
        )
        pred_classes = np.array([b.class_id for b in pred_raw], dtype=np.int32)

        # Identify FPs
        matched_pred = _matched_pred_indices(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_threshold)
        for pi in range(len(pred_boxes)):
            if pi not in matched_pred:
                error_cells.append((float(pred_scores[pi]), image_path, pred_boxes[pi], int(pred_classes[pi])))

    # Sort by confidence descending (worst errors first)
    error_cells.sort(key=lambda x: -x[0])

    rows, cols = grid_size
    cw, ch = cell_size
    grid = np.full((rows * ch, cols * cw, 3), 240, dtype=np.uint8)

    for idx in range(min(rows * cols, len(error_cells))):
        score, img_path, bbox, cid = error_cells[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Crop around the FP box
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (cw, ch))
        label = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
        cv2.putText(crop, f"FP:{label} {score:.2f}", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        r, c = divmod(idx, cols)
        grid[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = crop

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)

    return {
        "output": str(out_path),
        "total_fps_found": len(error_cells),
        "grid_cells": min(rows * cols, len(error_cells)),
    }


# ── Helpers ──────────────────────────────────────────────────

def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
) -> None:
    """Draw a dashed rectangle."""
    import cv2

    edges = [
        (pt1, (pt2[0], pt1[1])),
        ((pt2[0], pt1[1]), pt2),
        (pt2, (pt1[0], pt2[1])),
        ((pt1[0], pt2[1]), pt1),
    ]
    for start, end in edges:
        _draw_dashed_line(img, start, end, color, thickness, dash_length)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
) -> None:
    import cv2

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = max(1, int(np.sqrt(dx * dx + dy * dy)))
    dashes = length // dash_length

    for i in range(0, dashes, 2):
        frac_start = i / dashes
        frac_end = min((i + 1) / dashes, 1.0)
        sx = int(pt1[0] + dx * frac_start)
        sy = int(pt1[1] + dy * frac_start)
        ex = int(pt1[0] + dx * frac_end)
        ey = int(pt1[1] + dy * frac_end)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def _count_matches(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float,
) -> int:
    return len(_matched_pred_indices(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_threshold))


def _matched_pred_indices(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float,
) -> set[int]:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return set()

    from dtflowcv.metrics import box_iou_matrix_np
    iou_matrix = box_iou_matrix_np(pred_boxes.astype(np.float32), gt_boxes.astype(np.float32))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    order = np.argsort(-pred_scores)

    for pi in order:
        ious = iou_matrix[pi]
        best_gi = int(np.argmax(ious))
        if ious[best_gi] >= iou_threshold and best_gi not in matched_gt:
            if int(pred_classes[pi]) == int(gt_classes[best_gi]):
                matched_gt.add(best_gi)
                matched_pred.add(int(pi))

    return matched_pred
