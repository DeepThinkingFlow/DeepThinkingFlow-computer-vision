# Refactor Plan

## v0.2.0 Boundary Split

Goal: separate architecture boundaries without changing CLI behavior or JSON contracts.

Status:

- `dtflowcv.commands.app` owns the Typer app.
- `dtflowcv.commands.data`, `eval`, `model`, `runtime`, and `system` expose command group boundaries.
- `dtflowcv.cli` remains a shim for `dtflowcv = "dtflowcv.cli:app"`.
- `core`, `data`, `evaluation`, `models`, `runtime`, `reports`, and `serving` packages exist.
- Existing root modules remain compatibility surfaces.
- New package imports are covered by tests.

## v0.3.0 Evaluation And Benchmark Evidence

Planned work:

- strengthen COCO-style AP/AR implementation and optionally compare against `pycocotools`;
- generate committed smoke benchmark artifacts;
- add PR curve artifacts and threshold tuning reports;
- attach model, dataset, schema, and git hashes to accepted benchmark reports.

## v0.4.0 Dataset QA

Planned work:

- annotation heatmaps;
- object-count histograms;
- manifest-aware split leakage;
- optional perceptual near-duplicate detection;
- QA review queue generation.

## v0.5.0 Model Lifecycle

Planned work:

- harden the initial model registry commands;
- full model cards;
- promotion gates;
- export parity reports;
- dataset registry links.

## v0.6.0 Serving

Planned work:

- `dtflowcv serve`;
- `/healthz`, `/readyz`, `/predict`, `/model-info`, and `/metrics`;
- runtime telemetry;
- deployment docs and container contract.

## Non-Goals For v0.2.0

- no new serving API;
- no hidden model-quality claim;
- no mandatory heavy dependency lane in core tests;
- no deletion of legacy import paths.
