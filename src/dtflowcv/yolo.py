from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
BOUND_EPSILON = 1e-6

# Module-level cache for iter_images
_image_cache: dict[str, list[Path]] = {}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None

    @property
    def area_ratio(self) -> float:
        return self.width * self.height


def iter_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [root_path]
    cache_key = str(root_path.resolve())
    cached = _image_cache.get(cache_key)
    if cached is not None:
        return cached
    result = sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    _image_cache[cache_key] = result
    return result


def clear_image_cache() -> None:
    _image_cache.clear()


def label_path_for_image(image_path: str | Path, dataset_root: str | Path) -> Path:
    image = Path(image_path)
    root = Path(dataset_root)
    try:
        rel = image.relative_to(root)
    except ValueError:
        return image.with_suffix(".txt")

    parts = list(rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return root / "labels" / image.with_suffix(".txt").name


def related_label_path(image_path: str | Path, images_root: str | Path, labels_root: str | Path) -> Path:
    image = Path(image_path)
    image_base = Path(images_root)
    label_base = Path(labels_root)
    candidates: list[Path] = []

    for rel in _relative_image_paths(image, image_base):
        candidates.append(label_base / rel.with_suffix(".txt"))
    candidates.append(label_base / image.with_suffix(".txt").name)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return unique_candidates[0]


def parse_yolo_label_file(path: str | Path, with_confidence: bool = False) -> list[YoloBox]:
    label_path = Path(path)
    if not label_path.exists():
        return []

    # Batch read entire file, then parse
    content = label_path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    expected = 6 if with_confidence else 5
    boxes: list[YoloBox] = []

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != expected:
            raise ValueError(f"{label_path}:{line_number} expected exactly {expected} columns")
        class_id = _parse_class_id(parts[0], label_path, line_number)
        x_center = _parse_float_fast(parts[1], label_path, line_number)
        y_center = _parse_float_fast(parts[2], label_path, line_number)
        width = _parse_float_fast(parts[3], label_path, line_number)
        height = _parse_float_fast(parts[4], label_path, line_number)
        confidence = _parse_float_fast(parts[5], label_path, line_number) if with_confidence else None
        box = YoloBox(class_id, x_center, y_center, width, height, confidence)
        _validate_box(label_path, line_number, box, with_confidence)
        boxes.append(box)
    return boxes


def yolo_to_xyxy(box: YoloBox, image_width: int = 1, image_height: int = 1) -> tuple[float, float, float, float]:
    half_w = box.width * 0.5
    half_h = box.height * 0.5
    x1 = (box.x_center - half_w) * image_width
    y1 = (box.y_center - half_h) * image_height
    x2 = (box.x_center + half_w) * image_width
    y2 = (box.y_center + half_h) * image_height
    return x1, y1, x2, y2


def _relative_image_paths(image: Path, image_base: Path) -> list[Path]:
    if image_base.is_file():
        return [Path(image.name)]
    try:
        return [image.relative_to(image_base)]
    except ValueError:
        return [Path(image.name)]


def _parse_class_id(raw: str, path: Path, line_number: int) -> int:
    try:
        class_id = int(raw)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} class id must be an integer") from exc
    if class_id < 0:
        raise ValueError(f"{path}:{line_number} class id must be non-negative")
    return class_id


def _parse_float_fast(raw: str, path: Path, line_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} expected numeric box value") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}:{line_number} numeric values must be finite")
    return value


def _validate_box(path: Path, line_number: int, box: YoloBox, with_confidence: bool) -> None:
    xc, yc, w, h = box.x_center, box.y_center, box.width, box.height
    if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0 or w < 0.0 or w > 1.0 or h < 0.0 or h > 1.0:
        raise ValueError(f"{path}:{line_number} normalized box values must be in [0, 1]")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"{path}:{line_number} box width and height must be positive")
    half_w = w * 0.5
    half_h = h * 0.5
    x1 = xc - half_w
    y1 = yc - half_h
    x2 = xc + half_w
    y2 = yc + half_h
    if x1 < -BOUND_EPSILON or y1 < -BOUND_EPSILON or x2 > 1.0 + BOUND_EPSILON or y2 > 1.0 + BOUND_EPSILON:
        raise ValueError(f"{path}:{line_number} normalized box must stay inside image bounds")
    if with_confidence and box.confidence is not None and not 0.0 <= box.confidence <= 1.0:
        raise ValueError(f"{path}:{line_number} confidence must be in [0, 1]")
