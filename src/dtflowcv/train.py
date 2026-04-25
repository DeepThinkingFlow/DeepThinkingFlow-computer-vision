from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any

from dtflowcv.config import load_yaml
from dtflowcv.specs import class_names, validate_problem_spec


def train_yolo_baseline(
    problem_path: str | Path,
    dataset_path: str | Path,
    train_config_path: str | Path,
) -> dict[str, Any]:
    problem = load_yaml(problem_path)
    config = load_yaml(train_config_path)
    errors = validate_problem_spec(problem)
    blockers = [f"invalid_problem_spec:{error}" for error in errors]
    blockers.extend(_dataset_blockers(dataset_path, expected_class_count=len(class_names(problem))))
    blockers.extend(_module_blockers())
    blockers.extend(_runtime_blockers(config))
    if blockers:
        raise RuntimeError("build_blockers: " + "; ".join(blockers))

    ultralytics = _require("ultralytics", "python3 -m pip install -e '.[train]'")
    mlflow = _require("mlflow", "python3 -m pip install -e '.[train]'")

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    tracking_cfg = config.get("tracking", {})
    augmentation_cfg = config.get("augmentation", {})

    tracking_uri = tracking_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(tracking_cfg.get("experiment_name", "dtflowcv-baseline"))
    run_name = tracking_cfg.get("run_name_prefix", "baseline")
    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        assert run is not None
        mlflow.log_params(
            {
                "problem": str(problem_path),
                "dataset": str(dataset_path),
                "checkpoint": model_cfg.get("checkpoint", "yolov8n.pt"),
                "torch_cuda_available": _ultralytics_cuda_available(ultralytics),
                **{f"train_{key}": value for key, value in training_cfg.items()},
                **{f"aug_{key}": value for key, value in augmentation_cfg.items()},
            }
        )
        model = ultralytics.YOLO(model_cfg.get("checkpoint", "yolov8n.pt"))
        train_kwargs = {"data": str(dataset_path), **augmentation_cfg}
        if "image_size" in model_cfg:
            train_kwargs["imgsz"] = int(model_cfg["image_size"])
        for key in ("epochs", "batch", "workers", "seed", "device", "patience", "optimizer", "lr0", "lrf"):
            if key in training_cfg:
                train_kwargs[key] = training_cfg[key]
        if tracking_cfg.get("runs_dir"):
            train_kwargs["project"] = str(tracking_cfg["runs_dir"])
        if tracking_cfg.get("run_dir_name"):
            train_kwargs["name"] = str(tracking_cfg["run_dir_name"])

        started = time.perf_counter()
        result = model.train(**train_kwargs)
        elapsed_seconds = time.perf_counter() - started
        metrics = _extract_metrics(result)
        metrics["train_elapsed_seconds"] = elapsed_seconds
        metrics["max_rss_mb"] = _max_rss_mb()
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, int | float)})
        mlflow.log_param("result_type", type(result).__name__)
        if hasattr(result, "save_dir"):
            mlflow.log_param("ultralytics_save_dir", str(result.save_dir))
        info = run.info
    return {
        "status": "completed",
        "experiment_id": info.experiment_id,
        "run_id": info.run_id,
        "tracking_uri": mlflow.get_tracking_uri(),
        "result_type": type(result).__name__,
        "metrics": metrics,
    }


def _require(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"{module_name} is required for this command. Install with: {install_hint}") from exc


def _module_blockers() -> list[str]:
    blockers: list[str] = []
    for module_name in ("torch", "ultralytics", "mlflow"):
        if importlib.util.find_spec(module_name) is None:
            blockers.append(f"missing_python_module:{module_name}: install with python3 -m pip install -e '.[train]'")
    return blockers


def _runtime_blockers(config: dict[str, Any]) -> list[str]:
    training_cfg = config.get("training", {})
    device = training_cfg.get("device")
    if device is None:
        return []
    if _is_cpu_device(device) or str(device).lower() == "mps":
        return []
    if importlib.util.find_spec("torch") is None:
        return ["gpu_device_requested_but_torch_missing"]
    import torch

    if not torch.cuda.is_available():
        return [f"gpu_device_requested_but_cuda_unavailable:device={device}"]
    return []


def _is_cpu_device(device: Any) -> bool:
    return str(device).strip().lower() == "cpu"


def _dataset_blockers(dataset_path: str | Path, expected_class_count: int | None = None) -> list[str]:
    path = Path(dataset_path)
    if not path.exists():
        return [f"dataset_yaml_missing:{path}"]
    data = load_yaml(path)
    blockers: list[str] = []
    root_raw = data.get("path")
    if not root_raw:
        blockers.append("dataset.path_missing")
        return blockers
    root = Path(str(root_raw))
    if not root.is_absolute():
        root = path.parent / root
    if not root.exists():
        blockers.append(f"dataset_root_missing:{root}")
    for split in ("train", "val"):
        split_raw = data.get(split)
        if not split_raw:
            blockers.append(f"dataset.{split}_missing")
            continue
        split_path = Path(str(split_raw))
        if not split_path.is_absolute():
            split_path = root / split_path
        if not split_path.exists():
            blockers.append(f"dataset_{split}_path_missing:{split_path}")
            continue
        blockers.extend(_manifest_blockers(split, split_path))
    names = data.get("names")
    if not _valid_names_payload(names):
        blockers.append("dataset.names_missing")
    elif expected_class_count is not None and _names_count(names) != expected_class_count:
        blockers.append(f"dataset.names_count_mismatch:expected={expected_class_count}:actual={_names_count(names)}")
    return blockers


def _manifest_blockers(split: str, path: Path) -> list[str]:
    if path.suffix.lower() != ".txt":
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [f"dataset_{split}_manifest_empty:{path}"]
    missing = [line for line in lines[:20] if not Path(line).exists()]
    if missing:
        return [f"dataset_{split}_manifest_missing_images:{len(missing)}_of_first_{min(len(lines), 20)}"]
    return []


def _valid_names_payload(names: Any) -> bool:
    if isinstance(names, list):
        return bool(names) and all(isinstance(name, str) and name.strip() for name in names)
    if isinstance(names, dict):
        return bool(names) and all(
            str(key).strip() and isinstance(value, str) and value.strip()
            for key, value in names.items()
        )
    return False


def _names_count(names: Any) -> int:
    if isinstance(names, dict | list):
        return len(names)
    return 0


def _ultralytics_cuda_available(ultralytics: Any) -> bool:
    torch_module = getattr(ultralytics, "torch", None)
    return bool(torch_module and torch_module.cuda.is_available())


def _extract_metrics(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    result_dict = getattr(result, "results_dict", None)
    if isinstance(result_dict, dict):
        for key, value in result_dict.items():
            if isinstance(value, int | float):
                clean_key = str(key).replace("metrics/", "").replace("(", "").replace(")", "").replace(":", "_")
                metrics[clean_key] = float(value)
    return metrics


def _max_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return float(rss / 1024 / 1024)
        return float(rss / 1024)
    except Exception:
        return 0.0
