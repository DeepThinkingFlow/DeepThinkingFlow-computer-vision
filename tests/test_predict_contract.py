from pathlib import Path

from dtflowcv.predict import model_class_map, prediction_path_for_image, yolo_prediction_line


def test_model_class_map_normalizes_model_and_problem_names() -> None:
    mapping = model_class_map({0: "person", 2: "car", 9: "traffic light", 90: "toothbrush"}, ["car", "traffic_light"])

    assert mapping == {2: 0, 9: 1}


def test_prediction_path_for_image_preserves_nested_layout(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_path = image_root / "val2017" / "one.jpg"

    assert prediction_path_for_image(image_path, image_root, tmp_path / "preds") == tmp_path / "preds/val2017/one.txt"


def test_yolo_prediction_line_clips_to_image_bounds() -> None:
    line = yolo_prediction_line(1, [-10, 5, 110, 95], 0.75, 100, 100)

    assert line == "1 0.500000 0.500000 1.000000 0.900000 0.750000"


def test_yolo_prediction_line_drops_degenerate_boxes() -> None:
    assert yolo_prediction_line(1, [20, 20, 20, 40], 0.75, 100, 100) is None
