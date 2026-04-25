from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from dtflowcv.config import read_json, write_json


def normalize_coco_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def prepare_coco_yolo(
    images_dir: str | Path,
    annotations_json: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    include_empty: bool = True,
    include_crowd: bool = False,
    min_box_area_ratio: float = 0.0,
) -> dict[str, Any]:
    images_root = Path(images_dir).resolve()
    ann_path = Path(annotations_json).resolve()
    out_root = Path(out_dir).resolve()
    out_images = out_root / "images" / images_root.name
    out_labels = out_root / "labels" / images_root.name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    payload = read_json(ann_path)
    categories = payload.get("categories", [])
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])

    class_to_id = {normalize_coco_name(name): idx for idx, name in enumerate(class_names)}
    category_id_to_name = {
        int(category["id"]): normalize_coco_name(str(category["name"]))
        for category in categories
    }
    category_to_class = {
        category_id: class_to_id[name]
        for category_id, name in category_id_to_name.items()
        if name in class_to_id
    }
    images_by_id = {int(image["id"]): image for image in images}
    labels_by_image: dict[int, list[str]] = {int(image["id"]): [] for image in images}
    filtered_annotations_by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in images}
    kept_by_class: Counter[int] = Counter()
    ignored = Counter()

    for annotation in annotations:
        image_id = int(annotation["image_id"])
        image = images_by_id.get(image_id)
        if image is None:
            ignored["missing_image"] += 1
            continue
        category_id = int(annotation["category_id"])
        class_id = category_to_class.get(category_id)
        if class_id is None:
            ignored["non_target_category"] += 1
            continue
        if int(annotation.get("iscrowd", 0)) and not include_crowd:
            ignored["crowd"] += 1
            continue
        yolo = _coco_bbox_to_yolo(annotation.get("bbox", []), int(image["width"]), int(image["height"]))
        if yolo is None:
            ignored["invalid_bbox"] += 1
            continue
        if yolo[2] * yolo[3] < min_box_area_ratio:
            ignored["too_small"] += 1
            continue
        labels_by_image[image_id].append(" ".join([str(class_id), *(f"{value:.6f}" for value in yolo)]))
        filtered_annotation = dict(annotation)
        filtered_annotation["category_id"] = class_id
        filtered_annotations_by_image[image_id].append(filtered_annotation)
        kept_by_class[class_id] += 1

    linked_images = 0
    empty_images = 0
    written_images = 0
    kept_image_ids: set[int] = set()
    for image in images:
        image_id = int(image["id"])
        labels = labels_by_image.get(image_id, [])
        if not labels and not include_empty:
            continue
        source = images_root / str(image["file_name"])
        if not source.exists():
            ignored["missing_file"] += 1
            continue
        target = out_images / source.name
        _link_or_copy(source, target)
        linked_images += 1
        label_path = out_labels / source.with_suffix(".txt").name
        label_path.write_text(("\n".join(labels) + "\n") if labels else "", encoding="utf-8")
        written_images += 1
        if not labels:
            empty_images += 1
        else:
            kept_image_ids.add(image_id)

    names = {idx: name for idx, name in enumerate(class_names)}
    dataset_yaml = {
        "path": str(out_root),
        "train": f"images/{images_root.name}",
        "val": f"images/{images_root.name}",
        "names": names,
    }
    with (out_root / "dataset_all.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dataset_yaml, fh, sort_keys=False)
    filtered_coco_path = out_root / "instances_filtered_road.json"
    filtered_coco = {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": [dict(image) for image in images if int(image["id"]) in kept_image_ids],
        "annotations": [
            annotation
            for image_id in sorted(kept_image_ids)
            for annotation in filtered_annotations_by_image.get(image_id, [])
        ],
        "categories": [
            {"id": idx, "name": name, "supercategory": "road_object"}
            for idx, name in enumerate(class_names)
        ],
    }
    write_json(filtered_coco_path, filtered_coco)

    summary = {
        "images_dir": str(images_root),
        "annotations_json": str(ann_path),
        "out_dir": str(out_root),
        "source_image_count": len(images),
        "written_image_count": written_images,
        "linked_image_count": linked_images,
        "empty_label_image_count": empty_images,
        "dropped_empty_image_count": len(images) - written_images,
        "annotation_count": len(annotations),
        "kept_annotation_count": sum(kept_by_class.values()),
        "kept_by_class": {class_names[class_id]: count for class_id, count in sorted(kept_by_class.items())},
        "ignored_annotations": dict(sorted(ignored.items())),
        "include_empty": include_empty,
        "include_crowd": include_crowd,
        "min_box_area_ratio": min_box_area_ratio,
        "dataset_yaml": str(out_root / "dataset_all.yaml"),
        "filtered_coco_json": str(filtered_coco_path),
    }
    write_json(out_root / "prepare_summary.json", summary)
    return summary


def write_ultralytics_dataset_yaml(
    output_path: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path | None,
    class_names: list[str],
) -> None:
    payload: dict[str, Any] = {
        "path": ".",
        "train": str(Path(train_manifest).resolve()),
        "val": str(Path(val_manifest).resolve()),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    if test_manifest is not None:
        payload["test"] = str(Path(test_manifest).resolve())
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def _coco_bbox_to_yolo(bbox: list[Any], image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return None
    x, y, width, height = (float(value) for value in bbox)
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(clipped_width / image_width, 0.0), 1.0),
        min(max(clipped_height / image_height, 0.0), 1.0),
    )


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)
