# DeepThinkingFlow Computer Vision

This repo is a computer-vision pipeline for object detection. It is not a trained model and it is not production-ready by README claim alone. The verified core surface is package code for problem-spec validation, synthetic smoke data, strict YOLO parsing, dataset audit, split manifests, prediction-file evaluation, error export, preprocessing profiling, dependency-gated training, and native build contracts.

## Problem Contract

Default task: road-scene object detection on RGB images.

- Input: RGB uint8 images, usually 640-1280 px wide, from static images or extracted video frames.
- Output: object class plus bounding box in YOLO `class x_center y_center width height confidence` format.
- Primary metric: `mAP@0.5`.
- Secondary metrics: `mAP@0.5:0.95`, class-wise AP, FPS, and latency P99.
- Public benchmark reference: COCO val2017 filtered to the configured classes. COCO8 and generated synthetic data are smoke tests only.

The source contract is [configs/problem.yaml](configs/problem.yaml).

## Architecture

- `dtflowcv.specs`: validates the problem contract before pipeline commands run.
- `dtflowcv.yolo`: strict YOLO parsing, normalized box validation, and nested image/label path resolution.
- `dtflowcv.dataset`: image record loading, label health audit, duplicate/leakage checks, reports, and split manifests.
- `dtflowcv.evaluate` and `dtflowcv.metrics`: mAP and confusion matrix with explicit unknown-class handling.
- `dtflowcv.errors`: false positive, false negative, duplicate, and class-confusion export.
- `dtflowcv.predict`: Ultralytics checkpoint predictions mapped into the project class schema.
- `dtflowcv.benchmark`: prediction-file benchmark plus preprocessing profile; not full model inference.
- `dtflowcv.inference_benchmark`: full runtime command for checkpoint inference latency/FPS.
- `dtflowcv.train`: dependency-gated Ultralytics/MLflow baseline training.
- `dtflowcv.infer`, `dtflowcv.video`, `dtflowcv.tracking`: optional image/video inference and class-aware tracking.
- `dtflowcv.health`: machine-readable doctor report.
- [native/dtflowcv_native](native/dtflowcv_native): optional Rust/C extension.

## Claim Boundaries

| Claim | CI proved? | Condition |
| --- | --- | --- |
| Synthetic smoke dataset | Yes | Core deps |
| Problem spec validation | Yes | Core deps |
| Dataset audit and split manifests | Yes | Core deps |
| Prediction-file evaluation and benchmark gate | Yes | Core deps, supplied prediction files |
| Dataset card | Yes | Core deps, dirty labels do not crash the card |
| Visualization and frame extraction | Yes | `viz`/`video` extras |
| Image/video YOLO inference | Blocked contract only in core CI | Requires `ultralytics` plus video/runtime deps |
| Native Rust/C source | Cargo test only | `maturin develop` must be run before Python-native claims |
| Full inference latency/FPS | Command exists, not core-smoked | Requires `train` runtime and a checkpoint |
| Model quality | No | Needs locked real benchmark predictions and accepted metrics |

## Install

```bash
python -m pip install -e ".[dev]" -c constraints-dev.txt
```

Optional lanes:

```bash
python -m pip install -e ".[dev,viz,video]" -c constraints-dev.txt -c constraints-benchmark.txt
python -m pip install -e ".[train]" -c constraints-train-cpu.txt
python -m pip install -e ".[export,deploy]"
python -m pip install -e ".[native]" -c constraints-native.txt
```

Missing optional dependencies must return compact JSON with `status: blocked` and `build_blockers`; public CLI commands should not expose raw `ModuleNotFoundError` stacktraces.

## Core Smoke

```bash
PYTHONPATH=src python -m dtflowcv check-spec configs/problem.yaml
PYTHONPATH=src python -m dtflowcv make-demo-dataset --out data/demo --images 24
PYTHONPATH=src python -m dtflowcv audit-dataset data/demo --problem configs/problem.yaml --out reports/demo_audit
PYTHONPATH=src python -m dtflowcv split-dataset data/demo --problem configs/problem.yaml --out data/demo_splits
PYTHONPATH=src python -m dtflowcv evaluate-yolo --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --problem configs/problem.yaml
PYTHONPATH=src python -m dtflowcv benchmark-yolo --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --profile-iterations 1 --width 64 --height 64
PYTHONPATH=src python -m dtflowcv export-errors --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --problem configs/problem.yaml --out reports/demo_errors.json
PYTHONPATH=src python -m dtflowcv dataset-card data/demo --problem configs/problem.yaml --out reports/demo_card
```

## Benchmarking

`benchmark-yolo` evaluates existing prediction label files and separately profiles preprocessing. It does not measure model forward pass, NMS, device transfer, or end-to-end runtime latency.

```bash
dtflowcv benchmark-yolo \
  --images data/demo/images \
  --labels data/demo/labels \
  --preds data/demo/predictions \
  --problem configs/problem.yaml \
  --profile-iterations 1
```

Use `benchmark-inference` when the runtime stack is installed and a checkpoint is available:

```bash
dtflowcv benchmark-inference \
  --images data/demo/images \
  --model yolov8n.pt \
  --problem configs/problem.yaml \
  --device cpu \
  --warmup 5 \
  --runs 30 \
  --batch 1
```

## COCO Prepare

`prepare-coco` writes `dataset_smoke_all.yaml`, where `train` and `val` point to the same image folder. That file is for smoke verification only. Do not use it for real training, validation, or benchmark claims without a separate split.

## Native Lane

```bash
cargo test --manifest-path native/dtflowcv_native/Cargo.toml
python -m pip install -e ".[native]" -c constraints-native.txt
maturin develop --manifest-path native/dtflowcv_native/Cargo.toml
dtflowcv native-info
```

Native performance is unclaimed until the extension is built and benchmarked on the target machine.

## Test

```bash
ruff check src tests scripts notebooks
PYTHONPATH=src python -m compileall -q src tests scripts
PYTHONPATH=src python -m pytest -q
cargo test --manifest-path native/dtflowcv_native/Cargo.toml
```

CI separates core, visualization/video, heavy optional blocked behavior, native, and docs hygiene. Real model quality still requires a fixed dataset split, generated predictions, and an accepted benchmark report.
