from pathlib import Path

from dtflowcv.demo import create_demo_dataset
from dtflowcv.errors import export_detection_errors
from dtflowcv.evaluate import evaluate_yolo_predictions
from dtflowcv.profile import profile_preprocess


def test_evaluate_demo_predictions(tmp_path: Path) -> None:
    classes = ["person", "car"]
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, classes, image_count=6, seed=11)
    result = evaluate_yolo_predictions(dataset / "images", dataset / "labels", dataset / "predictions", len(classes))
    assert result["target_count"] > 0
    assert result["prediction_count"] == result["target_count"]
    assert result["map"] > 0.9


def test_profile_preprocess(tmp_path: Path) -> None:
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, ["person"], image_count=2, seed=13)
    result = profile_preprocess(dataset / "images", iterations=1, size=(64, 64))
    assert result["sample_count"] == 2
    assert result["latency_ms"]["p99"] >= 0.0


def test_export_detection_errors_has_no_errors_for_demo_predictions(tmp_path: Path) -> None:
    classes = ["person", "car"]
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, classes, image_count=4, seed=17)
    result = export_detection_errors(dataset / "images", dataset / "labels", dataset / "predictions", classes)
    assert result["error_count"] == 0
