from pathlib import Path

from dtflowcv.demo import create_demo_dataset
from dtflowcv.errors import export_detection_errors
from dtflowcv.evaluate import evaluate_yolo_predictions
from dtflowcv.profile import profile_preprocess
from PIL import Image


def test_evaluate_demo_predictions(tmp_path: Path) -> None:
    classes = ["person", "car"]
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, classes, image_count=6, seed=11)
    result = evaluate_yolo_predictions(dataset / "images", dataset / "labels", dataset / "predictions", len(classes))
    assert result["target_count"] > 0
    assert result["prediction_count"] == result["target_count"]
    assert result["map"] > 0.9
    assert result["map50_95"] > 0.7
    assert result["map75"] > 0.8
    assert "coco_metrics" in result


def test_evaluate_reports_out_of_schema_classes_without_crashing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    preds = tmp_path / "predictions"
    images.mkdir()
    labels.mkdir()
    preds.mkdir()
    Image.new("RGB", (100, 100), (255, 255, 255)).save(images / "one.jpg")
    (labels / "one.txt").write_text("0 0.5 0.5 0.4 0.4\n999 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (preds / "one.txt").write_text("999 0.5 0.5 0.4 0.4 0.99\n", encoding="utf-8")

    result = evaluate_yolo_predictions(images, labels, preds, class_count=1)

    assert result["invalid_target_class_id_count"] == 1
    assert result["invalid_target_class_ids"][0]["class_id"] == 999
    assert result["unknown_prediction_fp"] == 1
    assert result["invalid_prediction_class_ids"][0]["class_id"] == 999


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
