from __future__ import annotations

import csv
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from dtflowcv.config import write_json
from dtflowcv.yolo import YoloBox, iter_images, label_path_for_image, parse_yolo_label_file


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    width: int
    height: int
    brightness: float
    boxes: tuple[YoloBox, ...]
    label_file_exists: bool

    @property
    def class_ids(self) -> set[int]:
        return {box.class_id for box in self.boxes}


def load_records(dataset_root: str | Path) -> list[ImageRecord]:
    root = Path(dataset_root)
    records: list[ImageRecord] = []
    for image_path in iter_images(root / "images" if (root / "images").exists() else root):
        label_path = label_path_for_image(image_path, root)
        label_exists = label_path.exists()
        boxes = tuple(parse_yolo_label_file(label_path))
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            brightness = _mean_brightness(rgb)
        records.append(ImageRecord(image_path, label_path, width, height, brightness, boxes, label_exists))
    return records


def audit_dataset(dataset_root: str | Path, class_names: list[str]) -> dict[str, Any]:
    records = load_records(dataset_root)
    class_counter: Counter[int] = Counter()
    bbox_areas: list[float] = []
    aspect_ratios: list[float] = []

    for record in records:
        for box in record.boxes:
            class_counter[box.class_id] += 1
            bbox_areas.append(box.area_ratio)
        aspect_ratios.append(record.width / max(record.height, 1))

    widths = [record.width for record in records]
    heights = [record.height for record in records]
    brightness = [record.brightness for record in records]

    report = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "summary": {
            "image_count": len(records),
            "annotated_image_count": sum(1 for record in records if record.boxes),
            "object_count": sum(class_counter.values()),
            "missing_label_files": sum(1 for record in records if not record.label_file_exists),
            "empty_label_files": sum(1 for record in records if record.label_file_exists and not record.boxes),
        },
        "class_frequency": {
            class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}": count
            for class_id, count in sorted(class_counter.items())
        },
        "image_width": _describe(widths),
        "image_height": _describe(heights),
        "image_aspect_ratio": _describe(aspect_ratios),
        "brightness": _describe(brightness),
        "bbox_area_ratio": _describe(bbox_areas),
        "warnings": _warnings(records, class_counter, class_names),
    }
    return report


def write_audit_report(report: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", report)
    _write_class_csv(out / "class_frequency.csv", report["class_frequency"])
    _write_svg_histogram(out / "brightness.svg", "Brightness", report["brightness"].get("histogram", []))
    _write_svg_histogram(out / "bbox_area_ratio.svg", "BBox Area Ratio", report["bbox_area_ratio"].get("histogram", []))


def stratified_split_records(
    records: list[ImageRecord],
    ratios: dict[str, float],
    seed: int = 1337,
) -> dict[str, list[ImageRecord]]:
    if not records:
        return {"train": [], "val": [], "test": []}

    split_names = ["train", "val", "test"]
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    total_by_class: Counter[int] = Counter()
    for record in shuffled:
        total_by_class.update(record.class_ids)

    target_images = _target_counts(len(records), ratios, split_names)
    target_by_class = {
        split: {class_id: total * ratios[split] for class_id, total in total_by_class.items()}
        for split in split_names
    }
    current_by_class = {split: Counter() for split in split_names}
    splits: dict[str, list[ImageRecord]] = {split: [] for split in split_names}

    shuffled.sort(key=lambda record: (_rarity_score(record, total_by_class), len(record.boxes)), reverse=True)
    for record in shuffled:
        best_split = max(
            split_names,
            key=lambda split: _split_score(
                split,
                record,
                splits,
                target_images,
                current_by_class,
                target_by_class,
            ),
        )
        splits[best_split].append(record)
        current_by_class[best_split].update(record.class_ids)

    return splits


def write_split_manifests(splits: dict[str, list[ImageRecord]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for split, records in splits.items():
        manifest_path = out / f"{split}.txt"
        with manifest_path.open("w", encoding="utf-8") as fh:
            for record in sorted(records, key=lambda item: str(item.image_path)):
                fh.write(str(record.image_path.absolute()))
                fh.write("\n")
        class_counter: Counter[int] = Counter()
        for record in records:
            class_counter.update(record.class_ids)
        summary[split] = {
            "image_count": len(records),
            "class_presence": {str(class_id): count for class_id, count in sorted(class_counter.items())},
            "manifest": str(manifest_path),
        }
    write_json(out / "split_summary.json", summary)


def qa_sample(records: list[ImageRecord], review_rate: float, seed: int = 1337) -> list[ImageRecord]:
    if not records:
        return []
    sample_size = max(1, math.ceil(len(records) * review_rate))
    rng = random.Random(seed)
    return rng.sample(records, min(sample_size, len(records)))


def _mean_brightness(image: Image.Image) -> float:
    small = image.convert("L").resize((32, 32))
    pixels = list(small.getdata())
    return float(sum(pixels) / max(len(pixels), 1))


def _describe(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "histogram": []}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _percentile(ordered, 0.10),
        "p50": _percentile(ordered, 0.50),
        "mean": statistics.fmean(ordered),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
        "histogram": _histogram(ordered),
    }


def _percentile(ordered_values: list[float], q: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    index = q * (len(ordered_values) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered_values[low]
    weight = index - low
    return ordered_values[low] * (1.0 - weight) + ordered_values[high] * weight


def _warnings(records: list[ImageRecord], class_counter: Counter[int], class_names: list[str]) -> list[str]:
    warnings: list[str] = []
    if not records:
        warnings.append("no_images_found")
        return warnings
    missing = sum(1 for record in records if not record.label_file_exists)
    if missing:
        warnings.append(f"{missing}_missing_label_files")
    empty = sum(1 for record in records if record.label_file_exists and not record.boxes)
    if empty:
        warnings.append(f"{empty}_empty_label_files")
    for class_id, name in enumerate(class_names):
        if class_counter[class_id] == 0:
            warnings.append(f"class_{name}_has_zero_objects")
    if class_counter:
        counts = list(class_counter.values())
        if max(counts) / max(min(counts), 1) >= 10:
            warnings.append("class_imbalance_ratio_at_least_10x")
    return warnings


def _write_class_csv(path: Path, class_frequency: dict[str, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_name", "count"])
        for class_name, count in class_frequency.items():
            writer.writerow([class_name, count])


def _write_svg_histogram(path: Path, title: str, histogram: list[dict[str, float | int]]) -> None:
    width = 720
    height = 260
    margin = 32
    if not histogram:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="32" y="40">{title}: no data</text></svg>\n'
        path.write_text(svg, encoding="utf-8")
        return
    counts = [int(bucket["count"]) for bucket in histogram]
    low = float(histogram[0]["low"])
    high = float(histogram[-1]["high"])
    max_count = max(counts) or 1
    bar_width = (width - margin * 2) / len(histogram)
    bars = []
    for idx, count in enumerate(counts):
        bar_height = (height - margin * 2 - 32) * count / max_count
        x = margin + idx * bar_width
        y = height - margin - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="#3b82f6"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="{margin}" y="24" font-family="sans-serif" font-size="16">{title}</text>'
        + "".join(bars)
        + f'<text x="{margin}" y="{height - 8}" font-family="sans-serif" font-size="12">min={low:.4f} max={high:.4f}</text>'
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def _histogram(ordered_values: list[float], bins: int = 16) -> list[dict[str, float | int]]:
    low = ordered_values[0]
    high = ordered_values[-1]
    step = (high - low) / bins if high > low else 1.0
    counts = [0 for _ in range(bins)]
    for value in ordered_values:
        index = min(int((value - low) / step), bins - 1)
        counts[index] += 1
    return [
        {
            "low": low + idx * step,
            "high": low + (idx + 1) * step,
            "count": count,
        }
        for idx, count in enumerate(counts)
    ]


def _target_counts(total: int, ratios: dict[str, float], split_names: list[str]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in split_names}
    counts = {split: int(math.floor(raw[split])) for split in split_names}
    remaining = total - sum(counts.values())
    for split in sorted(split_names, key=lambda name: raw[name] - counts[name], reverse=True):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1
    return counts


def _rarity_score(record: ImageRecord, total_by_class: Counter[int]) -> float:
    if not record.class_ids:
        return 0.0
    return sum(1.0 / max(total_by_class[class_id], 1) for class_id in record.class_ids)


def _split_score(
    split: str,
    record: ImageRecord,
    splits: dict[str, list[ImageRecord]],
    target_images: dict[str, int],
    current_by_class: dict[str, Counter[int]],
    target_by_class: dict[str, dict[int, float]],
) -> float:
    capacity_left = target_images[split] - len(splits[split])
    if capacity_left <= 0:
        return -1_000_000.0
    class_improvement = 0.0
    for class_id in record.class_ids:
        target = target_by_class[split].get(class_id, 0.0)
        current = current_by_class[split][class_id]
        before = abs(target - current)
        after = abs(target - (current + 1))
        class_improvement += before - after
    size_before = abs(target_images[split] - len(splits[split]))
    size_after = abs(target_images[split] - (len(splits[split]) + 1))
    size_improvement = size_before - size_after
    return class_improvement * 10.0 + size_improvement * 0.01
