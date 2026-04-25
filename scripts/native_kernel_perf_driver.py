from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import dtflowcv_native
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/native_kernel_perf/summary.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    image_paths = sorted(args.images.rglob("*.jpg"))[: args.limit]
    if not image_paths:
        raise ValueError(f"no jpg images under {args.images}")

    inputs = [load_resized_rgb(path, args.size) for path in image_paths]
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]

    for index in range(args.warmup):
        dtflowcv_native.normalize_hwc_u8_to_chw_f32(inputs[index % len(inputs)], mean, std)

    started = time.perf_counter()
    checksum = 0.0
    for index in range(args.iterations):
        out = np.asarray(dtflowcv_native.normalize_hwc_u8_to_chw_f32(inputs[index % len(inputs)], mean, std))
        checksum += float(out[0, 0, 0])
    elapsed = time.perf_counter() - started

    pixels_per_image = args.size * args.size
    input_bytes = pixels_per_image * 3
    output_bytes = pixels_per_image * 3 * 4
    bytes_per_call = input_bytes + output_bytes
    ops_per_pixel_estimate = 6
    flops_per_call_estimate = pixels_per_image * 3 * ops_per_pixel_estimate
    payload = {
        "image_count_loaded": len(inputs),
        "resize": args.size,
        "warmup_calls": args.warmup,
        "timed_calls": args.iterations,
        "elapsed_seconds": elapsed,
        "mean_ms_per_call": elapsed * 1000.0 / args.iterations,
        "estimated_bytes_per_call": bytes_per_call,
        "estimated_flops_per_call": flops_per_call_estimate,
        "estimated_arithmetic_intensity_flop_per_byte": flops_per_call_estimate / bytes_per_call,
        "estimated_effective_bandwidth_gb_s": (bytes_per_call * args.iterations) / elapsed / 1e9,
        "checksum": checksum,
        "perf_command": (
            "perf stat -e cache-misses,cache-references,instructions,cycles .venv/bin/python "
            "scripts/native_kernel_perf_driver.py --images "
            "data/coco/prepared/val2017_road_target_only_yolo/images/val2017 --limit 100 --warmup 20 "
            "--iterations 500"
        ),
        "methodology_limit": (
            "This isolates native normalization on predecoded resized RGB arrays. Hardware counters were not "
            "collected on this host because perf is not installed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_resized_rgb(path: Path, size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR))


if __name__ == "__main__":
    main()
