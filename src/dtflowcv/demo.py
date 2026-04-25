from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

PALETTE = [
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (234, 88, 12),
    (147, 51, 234),
    (8, 145, 178),
    (202, 138, 4),
    (219, 39, 119),
]


def create_demo_dataset(root: str | Path, class_names: list[str], image_count: int = 24, seed: int = 1337) -> None:
    out = Path(root)
    images_dir = out / "images"
    labels_dir = out / "labels"
    preds_dir = out / "predictions"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for idx in range(image_count):
        width = rng.choice([480, 640, 800])
        height = rng.choice([360, 480, 600])
        base = 40 + (idx * 7) % 160
        image = Image.new("RGB", (width, height), (base, base + 10, min(base + 25, 255)))
        draw = ImageDraw.Draw(image)
        object_count = 1 + (idx % 3)
        labels: list[str] = []
        preds: list[str] = []

        for obj_idx in range(object_count):
            class_id = (idx + obj_idx) % len(class_names)
            box_w = rng.randint(max(24, width // 12), max(48, width // 4))
            box_h = rng.randint(max(24, height // 12), max(48, height // 4))
            x1 = rng.randint(0, max(width - box_w - 1, 1))
            y1 = rng.randint(0, max(height - box_h - 1, 1))
            x2 = x1 + box_w
            y2 = y1 + box_h
            color = PALETTE[class_id % len(PALETTE)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

            yolo = _xyxy_to_yolo(x1, y1, x2, y2, width, height)
            labels.append(_format_label(class_id, yolo))

            jitter = rng.uniform(-0.012, 0.012)
            pred = (
                min(max(yolo[0] + jitter, 0.001), 0.999),
                min(max(yolo[1] - jitter, 0.001), 0.999),
                min(max(yolo[2] * rng.uniform(0.96, 1.04), 0.001), 0.999),
                min(max(yolo[3] * rng.uniform(0.96, 1.04), 0.001), 0.999),
            )
            confidence = 0.82 + rng.random() * 0.15
            preds.append(_format_label(class_id, _clamp_yolo_box(pred), confidence))

        stem = f"demo_{idx:04d}"
        image.save(images_dir / f"{stem}.jpg", quality=90)
        (labels_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        (preds_dir / f"{stem}.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[float, float, float, float]:
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    return x_center, y_center, box_w, box_h


def _format_label(class_id: int, box: tuple[float, float, float, float], confidence: float | None = None) -> str:
    values = [str(class_id), *(f"{value:.6f}" for value in box)]
    if confidence is not None:
        values.append(f"{confidence:.6f}")
    return " ".join(values)


def _clamp_yolo_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_center, y_center, box_w, box_h = box
    box_w = min(max(box_w, 0.001), 0.999)
    box_h = min(max(box_h, 0.001), 0.999)
    x_center = min(max(x_center, box_w / 2.0), 1.0 - box_w / 2.0)
    y_center = min(max(y_center, box_h / 2.0), 1.0 - box_h / 2.0)
    return x_center, y_center, box_w, box_h
