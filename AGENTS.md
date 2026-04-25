# Repository Guidelines

## Project Structure & Module Organization

`src/dtflowcv/` contains the Python package and Typer CLI for dataset audit, YOLO contracts, evaluation, profiling, prediction, deployment, and health checks. `configs/` stores YAML problem, dataset, baseline, and deploy contracts. `tests/` contains pytest contract and pipeline tests. `scripts/` holds one-off benchmark, verification, and report helpers. `native/dtflowcv_native/` is the optional Rust/PyO3 plus C preprocessing extension. `notebooks/` is for Colab/GPU experiments. `data/raw/`, `data/processed/`, `reports/`, and `artifacts/` keep `.gitkeep` placeholders; generated contents are ignored.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"`: install the package with pytest and Ruff.
- `python -m pip install -e ".[train,analysis,native]"`: install heavier optional lanes when needed.
- `PYTHONPATH=src python -m dtflowcv check-spec configs/problem.yaml`: validate the problem contract.
- `PYTHONPATH=src python -m dtflowcv make-demo-dataset --out data/demo --images 24`: create local smoke data.
- `ruff check src tests scripts notebooks`: run lint checks configured in `pyproject.toml`.
- `PYTHONPATH=src python -m pytest`: run the Python test suite.
- `python -m compileall -q src tests scripts`: catch syntax/import packaging issues.
- `cargo test --manifest-path native/dtflowcv_native/Cargo.toml`: verify the native build contract.
- `maturin develop --manifest-path native/dtflowcv_native/Cargo.toml`: build the optional native extension locally.

## Coding Style & Naming Conventions

Target Python 3.11+. Use 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and explicit names for CLI/report fields. Ruff uses line length 120 and lint families `E`, `F`, `I`, `UP`, `B`, and `SIM`; keep imports sorted and avoid broad rewrites. CLI commands should keep compact JSON on stdout and send human diagnostics to stderr.

## Testing Guidelines

Tests use pytest and generated synthetic images, so they must not require COCO, MLflow, Ultralytics, or GPU hardware unless explicitly marked as an optional lane. Name files `test_*.py` and add focused contract tests when changing YAML schemas, CLI outputs, metric calculations, native capability reporting, or acceptance gates.

## Commit & Pull Request Guidelines

The current history is minimal, so use concise imperative commits such as `Add dataset audit contract tests`. PRs should include the behavioral change, commands run, linked issue if any, and whether generated data/model artifacts were created. Do not commit datasets, `reports/*`, `artifacts/*`, `mlruns/`, `runs/`, `*.pt`, `*.onnx`, or `*.engine`.

## Operational Boundaries

Do not claim native, GPU, training, or benchmark readiness unless the exact command was run on the target environment. Keep local dataset paths and credentials out of tracked config files.
