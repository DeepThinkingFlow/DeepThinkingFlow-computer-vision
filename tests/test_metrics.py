from dtflowcv.metrics import DetectionPrediction, DetectionTarget, box_iou, map_at_iou


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
