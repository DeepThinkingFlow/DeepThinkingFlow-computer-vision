# DeepThinkingFlow Computer Vision Data Science Project

This repo is a computer-vision data science pipeline for object detection. It is not a trained model yet. The verified surface is the local package code: problem-spec validation, synthetic smoke data generation, strict YOLO label contracts, dataset audit, stratified split manifests, prediction evaluation, error export, preprocessing profiling, and dependency-gated baseline/native entrypoints.

## Problem Contract

Default task: road-scene object detection on RGB images.

- Input: RGB uint8 images, usually 640-1280 px wide, from static images or extracted video frames.
- Output: object class plus bounding box in YOLO `class x_center y_center width height confidence` format, with COCO JSON export planned through the same class schema.
- Primary metric: `mAP@0.5`.
- Secondary metrics: `mAP@0.5:0.95`, class-wise AP, FPS, and latency P99.
- Public benchmark reference: COCO val2017 filtered to the configured road-scene classes. `COCO8` or the generated synthetic dataset is only a smoke test, not a real benchmark.

The exact contract lives in [configs/problem.yaml](/home/danggiaminh/data-science-deepthinkingflow-computer-vision/configs/problem.yaml).

## Architecture

- `dtflowcv.specs`: validates the problem contract before pipeline commands run.
- `dtflowcv.yolo`: owns strict YOLO parsing, normalized box validation, and nested image/label path resolution.
- `dtflowcv.dataset`: loads image records, audits label health, writes reports, and creates stratified manifests.
- `dtflowcv.evaluate` and `dtflowcv.errors`: compute mAP@IoU and export false positive/negative, duplicate, and class-confusion cases.
- `dtflowcv.predict`: runs an Ultralytics checkpoint and maps model class names into the problem class schema.
- `dtflowcv.benchmark`: combines detection quality and preprocessing speed into an acceptance-gated benchmark report.
- `dtflowcv.train`: gates Ultralytics/MLflow baseline training on dataset, dependency, and runtime readiness.
- `dtflowcv.health`: emits a machine-readable doctor report covering spec, train dependencies, native extension state, and large-file push risk.
- `native/dtflowcv_native`: optional Rust/C preprocessing extension. Native performance is unclaimed until built and benchmarked on target hardware.

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
PYTHONPATH=src python3 -m dtflowcv benchmark-yolo --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --profile-iterations 1 --width 64 --height 64
PYTHONPATH=src python3 -m dtflowcv export-errors --images data/demo/images --labels data/demo/labels --preds data/demo/predictions --problem configs/problem.yaml --out reports/demo_errors.json
PYTHONPATH=src python3 -m dtflowcv profile-preprocess --images data/demo/images --iterations 2
```

CLI commands emit compact JSON on stdout so they can be used in shell gates and CI logs.

## Operational Doctor

Use this in a fully provisioned environment to check the spec, train extras, native extension, and tracked large-file risk:

```bash
.venv/bin/python -m dtflowcv doctor
```

Missing train extras or an unbuilt native extension are reported as blockers or claim boundaries. A clean package-only smoke run should use the commands above; a production training run should pass `doctor` in the intended training environment.

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

## Prediction And Benchmark Gate

Generate mapped predictions from an Ultralytics checkpoint:

```bash
dtflowcv predict-yolo \
  --images data/coco/prepared/val2017_person_car_target_only_yolo/images/val2017 \
  --problem configs/problem.coco2_reliable.yaml \
  --model yolov8n.pt \
  --out artifacts/predictions/person_car_val2017 \
  --device cpu \
  --conf 0.25
```

Then benchmark the exact prediction set:

```bash
dtflowcv benchmark-yolo \
  --images data/coco/prepared/val2017_person_car_target_only_yolo/images/val2017 \
  --labels data/coco/prepared/val2017_person_car_target_only_yolo/labels/val2017 \
  --preds artifacts/predictions/person_car_val2017 \
  --problem configs/problem.coco2_reliable.yaml \
  --profile-iterations 1 \
  --max-images 100
```

The command exits non-zero when any acceptance threshold fails. On this host, a 10-image CPU sample with `yolov8n.pt` and `conf=0.25` failed the configured `map50_min=0.50` gate (`map50=0.3729`), so the current checkpoint must not be described as accepted on the real person/car benchmark.

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
ruff check src tests scripts notebooks
PYTHONPATH=src python3 -m pytest
PYTHONPATH=src python3 -m compileall -q src tests scripts
cargo test --manifest-path native/dtflowcv_native/Cargo.toml
```

The GitHub Actions workflow under `.github/workflows/ci.yml` runs lint, compile, tests, synthetic benchmark smoke, and the native Rust/C build contract. The Python test suite uses generated synthetic images, so it does not need COCO, OpenCV, MLflow, or Ultralytics. Full training still needs the `train` extras and a real dataset YAML.
