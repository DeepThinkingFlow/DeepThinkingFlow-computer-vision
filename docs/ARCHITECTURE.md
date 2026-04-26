# DTFlowCV Architecture

DTFlowCV is organized as a compatibility-first computer vision pipeline.

The public CLI contract remains `dtflowcv.cli:app`. The implementation now lives at
`dtflowcv.commands.app`, and `dtflowcv.cli` is a shim so existing entrypoints keep working.

## Package Boundaries

| Package | Responsibility | Current maturity |
| --- | --- | --- |
| `dtflowcv.core` | Config loading, JSON IO, dependency gates, hashing, schema helpers | Thin stable adapters |
| `dtflowcv.commands` | Typer CLI app and command-facing group boundaries | CLI implementation lives in `commands.app` |
| `dtflowcv.data` | YOLO/COCO ingestion, audit, split, dataset card, registry hooks | Compatibility adapters over existing modules |
| `dtflowcv.evaluation` | IoU, AP, COCO-style AP, confusion matrix, PR curves, evaluation reports | Compatibility adapters plus evaluator protocol |
| `dtflowcv.models` | Prediction, training, export, export validation, model card, registry | Compatibility adapters over existing modules |
| `dtflowcv.runtime` | Image/video inference, tracking, video IO, visualization, preprocessing, native status | Compatibility adapters over existing modules |
| `dtflowcv.reports` | Benchmark, dataset card, and model card report adapters | Thin evidence boundary |
| `dtflowcv.serving` | Future serving API namespace | Reserved; no server is implemented yet |

## Compatibility Rule

Old imports remain supported. New code should prefer the package boundary matching its concern:

- Dataset work: `dtflowcv.data.*`
- Metric work: `dtflowcv.evaluation.*`
- Model lifecycle work: `dtflowcv.models.*`
- Runtime inference work: `dtflowcv.runtime.*`
- CLI additions: `dtflowcv.commands.*`

The root modules such as `dtflowcv.dataset`, `dtflowcv.metrics`, `dtflowcv.infer`, and
`dtflowcv.export` remain compatibility surfaces until implementation migration is complete.

## License And Vendor Boundary

The core repo is Apache-2.0. Third-party source, if ever vendored, retains its upstream license and must be
listed in `third_party/MANIFEST.json`.

Large CV/runtime dependencies stay dependency-managed:

- Ultralytics
- Torch
- OpenCV
- ONNXRuntime
- NumPy
- Pandas
- Matplotlib
- MLflow

Model backends are isolated behind `dtflowcv.models.backend` and `dtflowcv.models.backends.*` so runtime
performance work can move toward ONNXRuntime/TorchScript/native paths without copying upstream framework
source. `dtflowcv.models.registry` records model metadata and hashes only; model weights stay outside git.

## Claim Boundary

This refactor creates stable architectural boundaries. It does not by itself create:

- a production serving API;
- a full model registry lifecycle;
- a real model-quality benchmark artifact;
- export parity strong enough for deployment approval;
- near-duplicate leakage detection.

Those are roadmap items, not hidden claims.
