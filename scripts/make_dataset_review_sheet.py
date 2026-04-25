from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dtflowcv.config import load_yaml
from dtflowcv.specs import class_names
from dtflowcv.yolo import iter_images, parse_yolo_label_file, yolo_to_xyxy
from PIL import Image, ImageDraw

BOX_COLOR = (34, 197, 94)
TEXT_BG = (15, 23, 42)
TEXT_FG = (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem", type=Path, default=Path("configs/problem.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--empty-only", action="store_true")
    args = parser.parse_args()

    names = class_names(load_yaml(args.problem))
    images = iter_images(args.dataset / "images")
    records = []
    for image_path in images:
        label_path = label_path_for_dataset(args.dataset, image_path)
        boxes = parse_yolo_label_file(label_path)
        if args.empty_only and boxes:
            continue
        if not args.empty_only and not boxes:
            continue
        records.append((image_path, label_path, boxes))

    if not records:
        raise ValueError("no matching images to review")

    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.count, len(records)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    review_items = []
    tiles = []
    for image_path, label_path, boxes in selected:
        tiles.append(render_tile(image_path, label_path, boxes, names))
        review_items.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "box_count": len(boxes),
                "classes": sorted(
                    {
                        names[box.class_id] if 0 <= box.class_id < len(names) else f"unknown_{box.class_id}"
                        for box in boxes
                    }
                ),
            }
        )

    for sheet_idx, start in enumerate(range(0, len(tiles), 10)):
        sheet = make_sheet(tiles[start : start + 10], columns=2)
        sheet.save(args.out_dir / f"sheet_{sheet_idx:02d}.jpg", quality=90)
    (args.out_dir / "review_items.json").write_text(
        json.dumps(review_items, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "reviewed_images": len(review_items)}, indent=2, sort_keys=True))


def label_path_for_dataset(dataset_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(dataset_root / "images")
    return dataset_root / "labels" / rel.with_suffix(".txt")


def render_tile(image_path: Path, label_path: Path, boxes, names: list[str]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(480 / width, 320 / height)
    tile = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    header = Image.new("RGB", (tile.width, 56), TEXT_BG)
    out = Image.new("RGB", (tile.width, tile.height + header.height), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(tile, (0, header.height))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), image_path.name, fill=TEXT_FG)
    draw.text((8, 28), f"boxes:{len(boxes)} label:{label_path.name}", fill=TEXT_FG)
    for box in boxes:
        label = names[box.class_id] if 0 <= box.class_id < len(names) else f"unknown_{box.class_id}"
        draw_box(draw, yolo_to_xyxy(box, tile.width, tile.height), label)
    return out


def draw_box(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], label: str) -> None:
    x1, y1, x2, y2 = xyxy
    y1 += 56
    y2 += 56
    draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=2)
    text_w = max(40, len(label) * 7)
    draw.rectangle((x1, max(56, y1 - 16), x1 + text_w, max(72, y1)), fill=TEXT_BG)
    draw.text((x1 + 2, max(56, y1 - 15)), label, fill=BOX_COLOR)


def make_sheet(tiles: list[Image.Image], columns: int) -> Image.Image:
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (241, 245, 249))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % columns) * tile_w, (idx // columns) * tile_h))
    return sheet


if __name__ == "__main__":
    main()
