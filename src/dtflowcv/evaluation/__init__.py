"""Detection evaluation boundaries and metric adapters."""

from dtflowcv.evaluation.ap import average_recall, map_at_iou
from dtflowcv.evaluation.coco_ap import coco_style_metrics
from dtflowcv.evaluation.confusion import confusion_matrix
from dtflowcv.evaluation.iou import box_iou, box_iou_matrix_np
from dtflowcv.evaluation.report import evaluate_per_class_pr_curves, evaluate_yolo_predictions
from dtflowcv.evaluation.types import DetectionPrediction, DetectionTarget

__all__ = [
    "DetectionPrediction",
    "DetectionTarget",
    "average_recall",
    "box_iou",
    "box_iou_matrix_np",
    "coco_style_metrics",
    "confusion_matrix",
    "evaluate_per_class_pr_curves",
    "evaluate_yolo_predictions",
    "map_at_iou",
]
