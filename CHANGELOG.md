# Changelog

## 0.2.0

- License the core project as Apache-2.0 and add NOTICE/license policy documents.
- Add vendor governance with `third_party/MANIFEST.json`, `vendor-check`, `license-check`, and CI guards.
- Add dependency governance with `dependency-check`.
- Add backend abstraction plus dependency-managed Ultralytics and ONNXRuntime backend adapters.
- Add local model registry/downloader command foundation without auto-download or committed weights.
- Add optional dependency extras for video, visualization, export, deploy, and all runtime lanes.
- Expand CI into core, visualization/video, heavy-blocked, native, and docs jobs.
- Harden evaluator behavior for out-of-schema target and prediction class IDs.
- Add COCO-style AP/AR metrics alongside AP50 evaluation.
- Add `benchmark-inference` for checkpoint runtime latency smoke measurements.
- Split the package into compatibility-safe `core`, `commands`, `data`, `evaluation`, `models`,
  `runtime`, `reports`, and `serving` boundaries.
- Add dataset audit duplicate, leakage, corrupt image, bbox outlier, and class imbalance reporting.
- Add class-aware tracking by default and correct the tracking assignment contract wording.
- Make dataset cards survive invalid label files and report invalid label counts.
- Rename COCO all-in-one smoke manifest to `dataset_smoke_all.yaml` to avoid leakage ambiguity.
- Fix native Rust/C FFI and PyO3 0.27 build issues so `cargo test` passes.
