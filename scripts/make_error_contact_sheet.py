from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dtflowcv.config import load_yaml
from dtflowcv.specs import class_names
from dtflowcv.yolo import parse_yolo_label_file, yolo_to_xyxy


GT_COLOR = (34, 197, 94)
PRED_COLOR = (239, 68, 68)
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--problem", type=Path, default=Path("configs/problem.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/coco_val2017/error_review"))
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    names = class_names(load_yaml(args.problem))
    errors = json.loads(args.errors.read_text(encoding="utf-8"))
    examples = errors["examples"]
    paths_by_id = {Path(line.strip()).stem: Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()}

    by_image: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_image[example["image_id"]].append(example)

    selected = sorted(by_image, key=lambda image_id: len(by_image[image_id]), reverse=True)[: args.count]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    review_items = []
    tiles = []
    for image_id in selected:
        image_path = paths_by_id[image_id]
        tile = render_tile(image_path, args.labels / f"{image_id}.txt", args.preds / f"{image_id}.txt", names, by_image[image_id])
        tiles.append(tile)
        review_items.append(
            {
                "image_id": image_id,
                "image": str(image_path),
                "error_count": len(by_image[image_id]),
                "error_kinds": dict(Counter(error["kind"] for error in by_image[image_id])),
                "sheet": str(args.out_dir / f"sheet_{len(tiles) // 10:02d}.jpg"),
            }
        )

    for sheet_idx, start in enumerate(range(0, len(tiles), 10)):
        sheet = make_sheet(tiles[start : start + 10], columns=2)
        sheet.save(args.out_dir / f"sheet_{sheet_idx:02d}.jpg", quality=90)
    (args.out_dir / "review_items.json").write_text(json.dumps(review_items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "reviewed_images": len(review_items)}, indent=2))


def render_tile(image_path: Path, label_path: Path, pred_path: Path, names: list[str], errors: list[dict]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(480 / width, 320 / height)
    tile = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    draw = ImageDraw.Draw(tile)

    for box in parse_yolo_label_file(label_path):
        draw_box(draw, yolo_to_xyxy(box, tile.width, tile.height), GT_COLOR, f"GT {safe_name(names, box.class_id)}")
    for box in parse_yolo_label_file(pred_path, with_confidence=True):
        label = f"P {safe_name(names, box.class_id)} {box.confidence:.2f}" if box.confidence is not None else f"P {safe_name(names, box.class_id)}"
        draw_box(draw, yolo_to_xyxy(box, tile.width, tile.height), PRED_COLOR, label)

    header = Image.new("RGB", (tile.width, 56), (17, 24, 39))
    hdraw = ImageDraw.Draw(header)
    kinds = Counter(error["kind"] for error in errors)
    hdraw.text((8, 6), image_path.name, fill=TEXT_FG)
    hdraw.text((8, 28), " ".join(f"{key}:{value}" for key, value in sorted(kinds.items())), fill=TEXT_FG)
    out = Image.new("RGB", (tile.width, tile.height + header.height), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(tile, (0, header.height))
    return out


def draw_box(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = xyxy
    y1 += 56
    y2 += 56
    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    text_w = max(40, len(label) * 7)
    draw.rectangle((x1, max(56, y1 - 16), x1 + text_w, max(72, y1)), fill=TEXT_BG)
    draw.text((x1 + 2, max(56, y1 - 15)), label, fill=color)


def make_sheet(tiles: list[Image.Image], columns: int) -> Image.Image:
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (241, 245, 249))
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
        sheet.paste(tile, (x, y))
    return sheet


def safe_name(names: list[str], class_id: int) -> str:
    return names[class_id] if 0 <= class_id < len(names) else f"unknown_{class_id}"


if __name__ == "__main__":
    main()
