from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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


def parse_yolo_label_file(path: str | Path, with_confidence: bool = False) -> list[YoloBox]:
    label_path = Path(path)
    if not label_path.exists():
        return []

    boxes: list[YoloBox] = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            expected = 6 if with_confidence else 5
            if len(parts) < expected:
                raise ValueError(f"{label_path}:{line_number} expected at least {expected} columns")
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
            confidence = float(parts[5]) if with_confidence and len(parts) >= 6 else None
            box = YoloBox(class_id, x_center, y_center, width, height, confidence)
            _validate_box(label_path, line_number, box, with_confidence)
            boxes.append(box)
    return boxes


def yolo_to_xyxy(box: YoloBox, image_width: int = 1, image_height: int = 1) -> tuple[float, float, float, float]:
    x1 = (box.x_center - box.width / 2.0) * image_width
    y1 = (box.y_center - box.height / 2.0) * image_height
    x2 = (box.x_center + box.width / 2.0) * image_width
    y2 = (box.y_center + box.height / 2.0) * image_height
    return x1, y1, x2, y2


def _validate_box(path: Path, line_number: int, box: YoloBox, with_confidence: bool) -> None:
    values = [box.x_center, box.y_center, box.width, box.height]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{path}:{line_number} normalized box values must be in [0, 1]")
    if box.width <= 0.0 or box.height <= 0.0:
        raise ValueError(f"{path}:{line_number} box width and height must be positive")
    if with_confidence and box.confidence is not None and not 0.0 <= box.confidence <= 1.0:
        raise ValueError(f"{path}:{line_number} confidence must be in [0, 1]")
