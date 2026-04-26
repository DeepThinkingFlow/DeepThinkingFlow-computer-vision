# DeepThinkingFlow Computer Vision

This repo is a computer-vision pipeline for object detection. It is not a trained model and it is not production-ready by README claim alone. The verified core surface is package code for problem-spec validation, synthetic smoke data, strict YOLO parsing, dataset audit, split manifests, prediction-file evaluation, error export, preprocessing profiling, dependency-gated training, and native build contracts.

## License And Vendor Boundary

The core project is licensed under Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE),
[LICENSE_POLICY.md](LICENSE_POLICY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No third-party source code is currently vendored. Vendoring is governed by
[VENDOR_POLICY.md](VENDOR_POLICY.md) and [third_party/MANIFEST.json](third_party/MANIFEST.json).
Do not vendor Ultralytics, Torch, OpenCV, ONNXRuntime, NumPy, Pandas, Matplotlib, or MLflow into this
repo. Optional Ultralytics integration remains dependency-based and may carry AGPL-3.0 obligations unless
covered by an upstream enterprise license.

## Problem Contract

Default task: road-scene object detection on RGB images.

- Input: RGB uint8 images, usually 640-1280 px wide, from static images or extracted video frames.
- Output: object class plus bounding box in YOLO `class x_center y_center width height confidence` format.
- Primary metric: `mAP@0.5`.
- Secondary metrics: `mAP@0.5:0.95`, class-wise AP, FPS, and latency P99.
- Public benchmark reference: COCO val2017 filtered to the configured classes. COCO8 and generated synthetic data are smoke tests only.

The source contract is [configs/problem.yaml](configs/problem.yaml).

## Architecture

- `dtflowcv.commands`: Typer CLI implementation; `dtflowcv.cli` is a compatibility shim.
- `dtflowcv.core`: config, JSON IO, dependency gates, hashing, and schema helpers.
- `dtflowcv.data`: YOLO/COCO ingestion, audit, split, dataset card, and registry boundaries.
- `dtflowcv.evaluation`: IoU, AP, COCO-style AP/AR, confusion matrix, PR curves, and report adapters.
- `dtflowcv.models`: prediction, training, export, export validation, model card, and registry boundaries.
- `dtflowcv.models.backend`: backend protocol for dependency-managed inference adapters.
- `dtflowcv.models.backends.ultralytics`: optional Ultralytics adapter; no upstream source is copied.
- `dtflowcv.models.backends.onnxruntime`: ONNXRuntime adapter skeleton with blocked behavior when missing.
- `dtflowcv.models.backends.torchscript`: TorchScript load-path skeleton; postprocess parity is not claimed.
- `dtflowcv.models.registry`: local model metadata registry; it stores hashes and paths, not model weights.
- `dtflowcv.runtime`: image/video inference, video IO, tracking, visualization, preprocessing, and native status.
- `dtflowcv.reports`: benchmark, dataset-card, and model-card evidence adapters.
- `dtflowcv.serving`: reserved namespace; no serving API is claimed yet.
- [native/dtflowcv_native](native/dtflowcv_native): optional Rust/C extension.

Detailed boundaries are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md).

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
| Full inference latency/FPS | Scheduled/manual heavy workflow | Requires `train` runtime and a checkpoint |
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

## Governance Checks

```bash
PYTHONPATH=src python -m dtflowcv vendor-check
PYTHONPATH=src python -m dtflowcv license-check
PYTHONPATH=src python -m dtflowcv dependency-check
```

`license-check` and `vendor-check` are CI-guarded. They enforce the Apache-2.0 project license files,
the third-party manifest contract, forbidden-license denylist rules, and the no-tracked-model-artifact rule.

Model artifacts are intentionally ignored by git: `*.pt`, `*.pth`, `*.onnx`, `*.engine`, `*.safetensors`,
and `*.torchscript` must not be committed. Register existing local artifacts with:

```bash
PYTHONPATH=src python -m dtflowcv model-register \
  --name road-smoke \
  --version 0.1.0 \
  --path artifacts/models/road-smoke/model.onnx \
  --license Apache-2.0 \
  --source local
```

`model-download` never fetches network URLs. It only stages local files after explicit `--accept-license`.

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

`benchmark-yolo` evaluates existing prediction label files and separately profiles preprocessing. It reports AP50 plus internal COCO-style AP/AR fields such as `map50_95`, `map75`, `ap_small`, `ap_medium`, `ap_large`, `ar_1`, `ar_10`, and `ar_100`. It does not measure model forward pass, NMS, device transfer, or end-to-end runtime latency.

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

The scheduled/manual `.github/workflows/heavy-runtime.yml` workflow installs heavy runtime extras and smokes real `infer-images`, `benchmark-inference`, and ONNX export with `yolov8n.pt`. That workflow proves runtime viability only; it does not prove model quality.
