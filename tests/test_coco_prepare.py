from pathlib import Path

from PIL import Image

from dtflowcv.coco import prepare_coco_yolo
from dtflowcv.config import write_json
from dtflowcv.dataset import audit_dataset


def test_prepare_coco_yolo_keeps_empty_and_converts_target_boxes(tmp_path: Path) -> None:
    images = tmp_path / "val2017"
    images.mkdir()
    Image.new("RGB", (100, 50), (10, 20, 30)).save(images / "one.jpg")
    Image.new("RGB", (80, 80), (40, 50, 60)).save(images / "two.jpg")
    annotations = tmp_path / "instances_val2017.json"
    write_json(
        annotations,
        {
            "images": [
                {"id": 1, "file_name": "one.jpg", "width": 100, "height": 50},
                {"id": 2, "file_name": "two.jpg", "width": 80, "height": 80},
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 10, "name": "traffic light"},
                {"id": 90, "name": "toothbrush"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 40, 20], "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 90, "bbox": [1, 1, 5, 5], "iscrowd": 0},
            ],
        },
    )

    out = tmp_path / "coco_yolo"
    summary = prepare_coco_yolo(images, annotations, out, ["person", "traffic_light"], include_empty=True)
    assert summary["source_image_count"] == 2
    assert summary["written_image_count"] == 2
    assert summary["kept_annotation_count"] == 1
    assert summary["empty_label_image_count"] == 1

    report = audit_dataset(out, ["person", "traffic_light"])
    assert report["summary"]["image_count"] == 2
    assert report["summary"]["object_count"] == 1
    filtered = out / "instances_filtered_road.json"
    assert filtered.exists()
    assert '"category_id": 0' in filtered.read_text(encoding="utf-8")


def test_prepare_coco_yolo_can_drop_empty_images(tmp_path: Path) -> None:
    images = tmp_path / "val2017"
    images.mkdir()
    Image.new("RGB", (100, 50), (10, 20, 30)).save(images / "one.jpg")
    Image.new("RGB", (80, 80), (40, 50, 60)).save(images / "two.jpg")
    annotations = tmp_path / "instances_val2017.json"
    write_json(
        annotations,
        {
            "images": [
                {"id": 1, "file_name": "one.jpg", "width": 100, "height": 50},
                {"id": 2, "file_name": "two.jpg", "width": 80, "height": 80},
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 90, "name": "toothbrush"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 40, 20], "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 90, "bbox": [1, 1, 5, 5], "iscrowd": 0},
            ],
        },
    )

    out = tmp_path / "coco_yolo"
    summary = prepare_coco_yolo(images, annotations, out, ["person"], include_empty=False)
    assert summary["written_image_count"] == 1
    assert summary["empty_label_image_count"] == 0
    assert summary["dropped_empty_image_count"] == 1
    assert (out / "images" / "val2017" / "one.jpg").exists()
    assert not (out / "images" / "val2017" / "two.jpg").exists()
