from dtflowcv.metrics import (
    DetectionPrediction,
    DetectionTarget,
    box_iou,
    coco_style_metrics,
    confusion_matrix,
    map_at_iou,
)


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_map_perfect_prediction() -> None:
    targets = [DetectionTarget("a", 0, (0, 0, 10, 10))]
    predictions = [DetectionPrediction("a", 0, (0, 0, 10, 10), 0.99)]
    result = map_at_iou(targets, predictions, class_count=1)
    assert result["map"] == 1.0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_unknown_prediction_class_counts_as_false_positive() -> None:
    targets = [DetectionTarget("a", 0, (0, 0, 10, 10))]
    predictions = [DetectionPrediction("a", 999, (0, 0, 10, 10), 0.99)]
    result = map_at_iou(targets, predictions, class_count=1)
    assert result["unknown_prediction_fp"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_confusion_matrix_uses_unknown_bucket_for_out_of_schema_prediction() -> None:
    targets = [DetectionTarget("a", 0, (0, 0, 10, 10))]
    predictions = [DetectionPrediction("a", 999, (0, 0, 10, 10), 0.99)]
    result = confusion_matrix(targets, predictions, class_count=1)
    assert result["size"] == 3
    assert result["unknown_index"] == 2
    assert result["matrix"][0][2] == 1


def test_coco_style_metrics_are_perfect_for_perfect_predictions() -> None:
    targets = [DetectionTarget("a", 0, (0, 0, 100, 100))]
    predictions = [DetectionPrediction("a", 0, (0, 0, 100, 100), 0.99)]
    result = coco_style_metrics(targets, predictions, class_count=1)
    assert result["ap50_95"] == 1.0
    assert result["ap50"] == 1.0
    assert result["ap75"] == 1.0
    assert result["ar_1"] == 1.0
