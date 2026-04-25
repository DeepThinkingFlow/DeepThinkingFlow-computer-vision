# DeepThinkingFlow Computer Vision Data Science Project

This repo is a production-shaped computer-vision data science scaffold for object detection. It is not a trained model yet. The current verified surface is the local pipeline code: problem-spec validation, synthetic smoke data generation, YOLO-format dataset audit, stratified split, prediction evaluation, preprocessing profiling, and dependency-gated baseline/native entrypoints.

## Problem Contract

Default task: road-scene object detection on RGB images.

- Input: RGB uint8 images, usually 640-1280 px wide, from static images or extracted video frames.
- Output: object class plus bounding box in YOLO `class x_center y_center width height confidence` format, with COCO JSON export planned through the same class schema.
- Primary metric: `mAP@0.5`.
- Secondary metrics: `mAP@0.5:0.95`, class-wise AP, FPS, and latency P99.
- Public benchmark reference: COCO val2017 filtered to the configured road-scene classes. `COCO8` or the generated synthetic dataset is only a smoke test, not a real benchmark.

The exact contract lives in [configs/problem.yaml](/home/danggiaminh/data-science-deepthinkingflow-computer-vision/configs/problem.yaml).

## Phase Coverage

- Phase 1: `dtflowcv check-spec` validates input, output, metric, benchmark, and acceptance criteria before training.
- Phase 2: `dtflowcv audit-dataset` and `dtflowcv split-dataset` inspect class balance, image sizes, brightness, bounding-box ratios, missing labels, and stratified train/val/test splits.
- Phase 3: `dtflowcv train-baseline` wraps Ultralytics YOLO with MLflow logging. It fails closed if `ultralytics` or `mlflow` is missing.
- Phase 4: augmentation and model-capacity choices are encoded in [configs/baseline.yolo.yaml](/home/danggiaminh/data-science-deepthinkingflow-computer-vision/configs/baseline.yolo.yaml), not hidden in scripts.
- Phase 5: `dtflowcv profile-preprocess` measures preprocessing P50/P95/P99. Native Rust/C source is under [native/dtflowcv_native](/home/danggiaminh/data-science-deepthinkingflow-computer-vision/native/dtflowcv_native), with runtime capability reporting through `dtflowcv native-info`.
- Phase 6: `dtflowcv evaluate-yolo` computes mAP@0.5, and `dtflowcv export-errors` writes false-positive, false-negative, duplicate, and class-confusion examples without touching the test set during development.

## Quick Smoke Run

```bash
PYTHONPATH=src python3 -m dtflowcv make-demo-dataset --out data/demo --images 24
PYTHONPATH=src python3 -m dtflowcv check-spec configs/problem.yaml
PYTHONPATH=src python3 -m dtflowcv audit-dataset data/demo --problem configs/problem.yaml --out reports/demo_audit
PYTHONPATH=src python3 -m dtflowcv split-dataset data/demo --problem configs/problem.yaml --out data/demo_splits
PYTHONPATH=src python3 -m dtflowcv evaluate-yolo --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --problem configs/problem.yaml
PYTHONPATH=src python3 -m dtflowcv export-errors --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --problem configs/problem.yaml --out reports/demo_errors.json
PYTHONPATH=src python3 -m dtflowcv profile-preprocess --images data/demo/images --iterations 2
```

## Real Baseline Training

Use Python 3.11 or 3.12 for the training environment unless your Torch/Ultralytics wheels explicitly support your newer Python version. The local smoke lane was verified on Python 3.14, but the train extras are heavier and more version-sensitive.

Install the train extras first:

```bash
python3 -m pip install -e ".[train]"
```

Then point [configs/dataset.example.yaml](/home/danggiaminh/data-science-deepthinkingflow-computer-vision/configs/dataset.example.yaml) at a real YOLO dataset and run:

```bash
dtflowcv train-baseline --problem configs/problem.yaml --dataset configs/dataset.example.yaml --train-config configs/baseline.yolo.yaml
```

This command is intentionally dependency-gated. A missing model runtime, tracking backend, dataset path, or invalid spec is reported as a blocker instead of being silently skipped.

## Native Lane

The native extension is source-present but not built by default:

```bash
python3 -m pip install -e ".[native]"
maturin develop --manifest-path native/dtflowcv_native/Cargo.toml
dtflowcv native-info
```

On the current host, NASM is not installed and AVX-512 is not expected on the CPU class, so AVX-512/NASM performance is not claimed. The Python profiler still runs and reports whether the native extension is importable.

## Test

```bash
PYTHONPATH=src python3 -m pytest
```

The test suite uses generated synthetic images, so it does not need COCO, OpenCV, MLflow, or Ultralytics.
