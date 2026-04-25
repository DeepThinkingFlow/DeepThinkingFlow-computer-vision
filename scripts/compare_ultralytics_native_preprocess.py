from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.augment import LetterBox

import dtflowcv_native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/native_yolo_e2e/preprocess_parity.json"))
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {args.image}")

    reference_letterbox = ultralytics_tensor_from_bgr(bgr, args.size)
    native_letterbox = native_tensor_from_rgb(letterbox_bgr(bgr, args.size)[:, :, ::-1])
    native_straight = native_tensor_from_rgb(cv2.resize(bgr[:, :, ::-1], (args.size, args.size), interpolation=cv2.INTER_LINEAR))

    payload = {
        "image": str(args.image),
        "original_shape_hwc": list(bgr.shape),
        "native_capabilities": dtflowcv_native.capabilities(),
        "letterbox_native_matches_reference": bool(np.allclose(reference_letterbox, native_letterbox, rtol=0.0, atol=1e-6)),
        "letterbox_native_max_abs_diff": float(np.max(np.abs(reference_letterbox - native_letterbox))),
        "straight_resize_matches_ultralytics_letterbox": bool(np.allclose(reference_letterbox, native_straight, rtol=0.0, atol=1e-6)),
        "straight_resize_max_abs_diff": float(np.max(np.abs(reference_letterbox - native_straight))),
        "decision": "Native normalization is correct after Ultralytics letterbox. Straight 640x640 resize is not parity with Ultralytics preprocessing and must not be used for YOLO baseline inference.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def letterbox_bgr(bgr: np.ndarray, size: int) -> np.ndarray:
    return LetterBox(new_shape=(size, size), auto=False, scale_fill=False, scaleup=True, center=True, stride=32)(image=bgr)


def ultralytics_tensor_from_bgr(bgr: np.ndarray, size: int) -> np.ndarray:
    boxed = letterbox_bgr(bgr, size)
    rgb = boxed[:, :, ::-1]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0


def native_tensor_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb)
    return np.asarray(dtflowcv_native.normalize_hwc_u8_to_chw_f32(rgb, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))


if __name__ == "__main__":
    main()
