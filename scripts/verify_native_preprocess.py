from __future__ import annotations

import argparse
import json
import timeit
from pathlib import Path

import cv2
import dtflowcv_native
import numpy as np

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb_cv2(path: Path, size: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized)


def numpy_reference(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def native_normalize(image: np.ndarray) -> np.ndarray:
    return np.asarray(dtflowcv_native.normalize_hwc_u8_to_chw_f32(image, MEAN.tolist(), STD.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/native_verify.json"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    image = load_rgb_cv2(args.image, (args.width, args.height))
    ref = numpy_reference(image)
    native = native_normalize(image)
    abs_diff = np.abs(ref - native)

    ref_timer = timeit.Timer(lambda: numpy_reference(image))
    native_timer = timeit.Timer(lambda: native_normalize(image))
    ref_times = ref_timer.repeat(repeat=5, number=args.iterations)
    native_times = native_timer.repeat(repeat=5, number=args.iterations)

    payload = {
        "image": str(args.image),
        "shape_hwc": list(image.shape),
        "native_capabilities": dtflowcv_native.capabilities(),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "allclose_atol_1e-6": bool(np.allclose(ref, native, rtol=0.0, atol=1e-6)),
        "reference_numpy_ms_per_call_best": min(ref_times) * 1000.0 / args.iterations,
        "native_ms_per_call_best": min(native_times) * 1000.0 / args.iterations,
        "iterations_per_repeat": args.iterations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
