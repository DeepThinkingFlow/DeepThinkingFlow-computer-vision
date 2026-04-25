from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import dtflowcv_native
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from dtflowcv.config import load_yaml
from dtflowcv.specs import class_names
from dtflowcv.yolo import parse_yolo_label_file, yolo_to_xyxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--problem", type=Path, default=Path("configs/problem.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/native_yolo_e2e/overlay.jpg"))
    parser.add_argument("--summary", type=Path, default=Path("reports/native_yolo_e2e/summary.json"))
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    names = class_names(load_yaml(args.problem))
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = rgb.shape[:2]
    resized = np.ascontiguousarray(cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_LINEAR))
    chw = np.asarray(dtflowcv_native.normalize_hwc_u8_to_chw_f32(resized, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    tensor = torch.from_numpy(chw).unsqueeze(0)

    model = YOLO(str(args.model))
    result = model.predict(tensor, verbose=False, conf=args.conf)[0]

    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in parse_yolo_label_file(args.label):
        draw_box(
            draw,
            yolo_to_xyxy(box, original_width, original_height),
            (34, 197, 94),
            f"GT {safe_name(names, box.class_id)}",
        )

    predictions = []
    x_scale = original_width / args.size
    y_scale = original_height / args.size
    for xyxy, cls, conf in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        strict=False,
    ):
        class_id = int(cls)
        scaled = (
            float(xyxy[0] * x_scale),
            float(xyxy[1] * y_scale),
            float(xyxy[2] * x_scale),
            float(xyxy[3] * y_scale),
        )
        predictions.append(
            {
                "class_id": class_id,
                "class_name": safe_name(names, class_id),
                "confidence": float(conf),
                "xyxy": scaled,
            }
        )
        draw_box(draw, scaled, (239, 68, 68), f"P {safe_name(names, class_id)} {float(conf):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, quality=90)
    payload = {
        "image": str(args.image),
        "label": str(args.label),
        "model": str(args.model),
        "overlay": str(args.out),
        "native_capabilities": dtflowcv_native.capabilities(),
        "tensor_shape": list(tensor.shape),
        "tensor_min": float(tensor.min()),
        "tensor_max": float(tensor.max()),
        "prediction_count": len(predictions),
        "predictions": predictions,
        "ground_truth_count": len(parse_yolo_label_file(args.label)),
        "methodology_limit": (
            "This verifies native RGB HWC u8 to CHW float 0..1 tensor inference alignment by visual overlay. "
            "It does not verify Ultralytics letterbox parity or ImageNet mean/std preprocessing."
        ),
    }
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def draw_box(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text_w = max(44, len(label) * 7)
    draw.rectangle((x1, max(0, y1 - 18), x1 + text_w, max(18, y1)), fill=(15, 23, 42))
    draw.text((x1 + 2, max(1, y1 - 17)), label, fill=color)


def safe_name(names: list[str], class_id: int) -> str:
    return names[class_id] if 0 <= class_id < len(names) else f"unknown_{class_id}"


if __name__ == "__main__":
    main()
