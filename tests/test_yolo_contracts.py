from pathlib import Path

import pytest
from dtflowcv.dataset import audit_dataset
from dtflowcv.errors import export_detection_errors
from dtflowcv.evaluate import evaluate_yolo_predictions
from dtflowcv.yolo import parse_yolo_label_file, related_label_path
from PIL import Image


def test_yolo_parser_rejects_extra_columns(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly 5 columns"):
        parse_yolo_label_file(label)


def test_yolo_parser_rejects_non_integer_class_id(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("1.5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class id must be an integer"):
        parse_yolo_label_file(label)


def test_yolo_parser_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 0.05 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must stay inside image bounds"):
        parse_yolo_label_file(label)


def test_related_label_path_preserves_nested_image_layout(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image = image_root / "val2017" / "one.jpg"
    label = label_root / "val2017" / "one.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"not-used")
    label.write_text("", encoding="utf-8")

    assert related_label_path(image, image_root, label_root) == label


def test_nested_evaluation_and_error_export_use_relative_layout(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    pred_root = tmp_path / "predictions"
    image_dir = image_root / "val2017"
    label_dir = label_root / "val2017"
    pred_dir = pred_root / "val2017"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (20, 30, 40)).save(image_dir / "one.jpg")
    label_dir.joinpath("one.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
    pred_dir.joinpath("one.txt").write_text("0 0.500000 0.500000 0.200000 0.200000 0.990000\n", encoding="utf-8")

    metrics = evaluate_yolo_predictions(image_root, label_root, pred_root, class_count=1)
    errors = export_detection_errors(image_root, label_root, pred_root, ["person"])

    assert metrics["map"] == 1.0
    assert errors["error_count"] == 0


def test_audit_reports_invalid_label_files_without_crashing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (100, 100), (20, 30, 40)).save(images / "bad.jpg")
    labels.joinpath("bad.txt").write_text("0 0.050000 0.500000 0.200000 0.200000\n", encoding="utf-8")

    report = audit_dataset(dataset, ["person"])

    assert report["summary"]["invalid_label_files"] == 1
    assert report["summary"]["empty_label_files"] == 0
    assert report["label_errors"]
    assert "1_invalid_label_files" in report["warnings"]
