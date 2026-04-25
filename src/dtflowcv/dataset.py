from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat

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
    label_errors: tuple[str, ...] = ()
    image_errors: tuple[str, ...] = ()

    @property
    def class_ids(self) -> set[int]:
        return {box.class_id for box in self.boxes}


def _load_single_record(image_path: Path, root: Path, strict_labels: bool, strict_images: bool) -> ImageRecord:
    label_path = label_path_for_image(image_path, root)
    label_exists = label_path.exists()
    label_errors: tuple[str, ...] = ()
    try:
        boxes = tuple(parse_yolo_label_file(label_path))
    except ValueError as exc:
        if strict_labels:
            raise
        boxes = ()
        label_errors = (str(exc),)
    image_errors: tuple[str, ...] = ()
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            brightness = _mean_brightness(rgb)
    except Exception as exc:
        if strict_images:
            raise
        width = 0
        height = 0
        brightness = 0.0
        boxes = ()
        image_errors = (str(exc),)
    return ImageRecord(
        image_path, label_path, width, height, brightness, boxes, label_exists, label_errors, image_errors
    )


def load_records(
    dataset_root: str | Path,
    strict_labels: bool = True,
    strict_images: bool = True,
) -> list[ImageRecord]:
    root = Path(dataset_root)
    image_dir = root / "images" if (root / "images").exists() else root
    image_paths = iter_images(image_dir)

    if not image_paths:
        return []

    # Use thread pool for I/O-bound image reading
    max_workers = min(8, len(image_paths))
    if max_workers > 1 and len(image_paths) > 4:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_single_record, p, root, strict_labels, strict_images) for p in image_paths]
            return [f.result() for f in futures]

    return [_load_single_record(p, root, strict_labels, strict_images) for p in image_paths]


def audit_dataset(dataset_root: str | Path, class_names: list[str]) -> dict[str, Any]:
    records = load_records(dataset_root, strict_labels=False, strict_images=False)
    class_counter: Counter[int] = Counter()
    unknown_class_counter: Counter[int] = Counter()
    bbox_areas: list[float] = []
    aspect_ratios: list[float] = []
    num_classes = len(class_names)

    for record in records:
        for box in record.boxes:
            if 0 <= box.class_id < num_classes:
                class_counter[box.class_id] += 1
            else:
                unknown_class_counter[box.class_id] += 1
            bbox_areas.append(box.area_ratio)
        aspect_ratios.append(record.width / max(record.height, 1))

    widths = [record.width for record in records]
    heights = [record.height for record in records]
    brightness = [record.brightness for record in records]

    report = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "summary": {
            "image_count": len(records),
            "corrupt_image_count": sum(1 for record in records if record.image_errors),
            "annotated_image_count": sum(1 for record in records if record.boxes),
            "object_count": sum(class_counter.values()),
            "unknown_class_object_count": sum(unknown_class_counter.values()),
            "missing_label_files": sum(1 for record in records if not record.label_file_exists),
            "empty_label_files": sum(
                1 for record in records if record.label_file_exists and not record.boxes and not record.label_errors
            ),
            "invalid_label_files": sum(1 for record in records if record.label_errors),
        },
        "class_frequency": {
            class_names[class_id]: count
            for class_id, count in sorted(class_counter.items())
        },
        "unknown_class_frequency": {str(class_id): count for class_id, count in sorted(unknown_class_counter.items())},
        "image_width": _describe(widths),
        "image_height": _describe(heights),
        "image_aspect_ratio": _describe(aspect_ratios),
        "brightness": _describe(brightness),
        "bbox_area_ratio": _describe(bbox_areas),
        "duplicate_groups": _duplicate_groups(records),
        "split_leakage_candidates": _split_leakage_candidates(records),
        "corrupt_images": _image_errors(records),
        "bbox_outliers": _bbox_outliers(records),
        "class_imbalance_severity": _class_imbalance_severity(class_counter),
        "label_errors": _label_errors(records),
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
        lines = [str(record.image_path.absolute()) for record in sorted(records, key=lambda item: str(item.image_path))]
        manifest_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
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
    # Downscale to 8x8 — 16x fewer pixels than 32x32, still accurate for brightness
    small = image.convert("L").resize((8, 8))
    return float(ImageStat.Stat(small).mean[0])


def _describe(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "histogram": []}
    arr = np.array(values, dtype=np.float64)
    arr.sort()
    return {
        "count": len(arr),
        "min": float(arr[0]),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr[-1]),
        "histogram": _histogram_np(arr),
    }


def _histogram_np(sorted_arr: np.ndarray, bins: int = 16) -> list[dict[str, float | int]]:
    low = float(sorted_arr[0])
    high = float(sorted_arr[-1])
    if high <= low:
        return [{"low": low, "high": low + 1.0, "count": len(sorted_arr)}]
    counts, edges = np.histogram(sorted_arr, bins=bins, range=(low, high))
    return [
        {"low": float(edges[i]), "high": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(bins)
    ]


def _warnings(records: list[ImageRecord], class_counter: Counter[int], class_names: list[str]) -> list[str]:
    warnings: list[str] = []
    if not records:
        warnings.append("no_images_found")
        return warnings
    invalid = sum(1 for record in records if record.label_errors)
    if invalid:
        warnings.append(f"{invalid}_invalid_label_files")
    corrupt = sum(1 for record in records if record.image_errors)
    if corrupt:
        warnings.append(f"{corrupt}_corrupt_images")
    missing = sum(1 for record in records if not record.label_file_exists)
    if missing:
        warnings.append(f"{missing}_missing_label_files")
    empty = sum(1 for record in records if record.label_file_exists and not record.boxes and not record.label_errors)
    if empty:
        warnings.append(f"{empty}_empty_label_files")
    for class_id, name in enumerate(class_names):
        if class_counter[class_id] == 0:
            warnings.append(f"class_{name}_has_zero_objects")
    if class_counter:
        counts = list(class_counter.values())
        if max(counts) / max(min(counts), 1) >= 10:
            warnings.append("class_imbalance_ratio_at_least_10x")
    unknown = sorted(
        {
            box.class_id
            for record in records
            for box in record.boxes
            if box.class_id >= len(class_names)
        }
    )
    if unknown:
        warnings.append("unknown_class_ids:" + ",".join(str(class_id) for class_id in unknown))
    return warnings


def _duplicate_groups(records: list[ImageRecord], limit: int = 50) -> list[dict[str, Any]]:
    by_hash: dict[str, list[Path]] = {}
    for record in records:
        if record.image_errors:
            continue
        digest = _file_sha256(record.image_path)
        by_hash.setdefault(digest, []).append(record.image_path)
    groups = [
        {"sha256": digest, "images": [str(path) for path in sorted(paths)]}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]
    return groups[:limit]


def _split_leakage_candidates(records: list[ImageRecord], limit: int = 50) -> list[dict[str, Any]]:
    by_hash: dict[str, list[tuple[str, Path]]] = {}
    for record in records:
        if record.image_errors:
            continue
        split = _split_name(record.image_path)
        if split is None:
            continue
        digest = _file_sha256(record.image_path)
        by_hash.setdefault(digest, []).append((split, record.image_path))
    leaks: list[dict[str, Any]] = []
    for digest, items in sorted(by_hash.items()):
        splits = sorted({split for split, _ in items})
        if len(splits) > 1:
            leaks.append({
                "sha256": digest,
                "splits": splits,
                "images": [str(path) for _, path in sorted(items, key=lambda item: str(item[1]))],
            })
            if len(leaks) >= limit:
                break
    return leaks


def _split_name(path: Path) -> str | None:
    for part in path.parts:
        if part in {"train", "val", "test"}:
            return part
    return None


def _image_errors(records: list[ImageRecord], limit: int = 50) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for record in records:
        for error in record.image_errors:
            errors.append({"image": str(record.image_path), "error": error})
            if len(errors) >= limit:
                return errors
    return errors


def _bbox_outliers(records: list[ImageRecord], limit: int = 50) -> list[dict[str, Any]]:
    outliers: list[dict[str, Any]] = []
    for record in records:
        for box in record.boxes:
            box_aspect = box.width / max(box.height, 1e-9)
            reasons = []
            if box.area_ratio < 0.0001:
                reasons.append("tiny_box_area")
            if box.area_ratio > 0.90:
                reasons.append("huge_box_area")
            if box_aspect >= 10.0 or box_aspect <= 0.10:
                reasons.append("extreme_box_aspect")
            if reasons:
                outliers.append({
                    "image": str(record.image_path),
                    "class_id": box.class_id,
                    "area_ratio": box.area_ratio,
                    "box_aspect_ratio": box_aspect,
                    "reasons": reasons,
                })
                if len(outliers) >= limit:
                    return outliers
    return outliers


def _class_imbalance_severity(class_counter: Counter[int]) -> dict[str, Any]:
    counts = [count for count in class_counter.values() if count > 0]
    if not counts:
        return {"level": "none", "max_to_min_ratio": 0.0}
    ratio = max(counts) / max(min(counts), 1)
    if ratio >= 20:
        level = "severe"
    elif ratio >= 10:
        level = "high"
    elif ratio >= 3:
        level = "moderate"
    else:
        level = "low"
    return {"level": level, "max_to_min_ratio": ratio}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_errors(records: list[ImageRecord], limit: int = 50) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for record in records:
        for error in record.label_errors:
            errors.append({"image": str(record.image_path), "label": str(record.label_path), "error": error})
            if len(errors) >= limit:
                return errors
    return errors


def _write_class_csv(path: Path, class_frequency: dict[str, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_name", "count"])
        writer.writerows(class_frequency.items())


def _write_svg_histogram(path: Path, title: str, histogram: list[dict[str, float | int]]) -> None:
    width = 720
    height = 260
    margin = 32
    if not histogram:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="32" y="40">{title}: no data</text></svg>\n'
        )
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
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" '
            f'height="{bar_height:.1f}" fill="#3b82f6"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="{margin}" y="24" font-family="sans-serif" font-size="16">{title}</text>'
        + "".join(bars)
        + (
            f'<text x="{margin}" y="{height - 8}" font-family="sans-serif" font-size="12">'
            f"min={low:.4f} max={high:.4f}</text>"
        )
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


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
