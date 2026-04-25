from __future__ import annotations

import json
from pathlib import Path

import typer

from dtflowcv.benchmark import benchmark_yolo_pipeline
from dtflowcv.coco import prepare_coco_yolo, write_ultralytics_dataset_yaml
from dtflowcv.config import load_yaml, write_json
from dtflowcv.dataset import (
    audit_dataset,
    load_records,
    qa_sample,
    stratified_split_records,
    write_audit_report,
    write_split_manifests,
)
from dtflowcv.demo import create_demo_dataset
from dtflowcv.errors import export_detection_errors
from dtflowcv.evaluate import evaluate_yolo_predictions
from dtflowcv.health import project_health
from dtflowcv.native import hardware_report, native_status
from dtflowcv.predict import predict_ultralytics_yolo
from dtflowcv.profile import profile_preprocess
from dtflowcv.specs import class_names, split_ratios, validate_problem_spec
from dtflowcv.train import train_yolo_baseline

app = typer.Typer(no_args_is_help=True)


def _emit(payload: dict | list) -> None:
    typer.echo(json.dumps(payload, sort_keys=True, default=str))


def _exit_with_errors(errors: list[str], status: str = "failed", code: int = 1) -> None:
    _emit({"status": status, "errors": errors})
    raise typer.Exit(code)


# ── Existing Commands ────────────────────────────────────────

@app.command("check-spec")
def check_spec(problem: Path = typer.Argument(..., help="Problem YAML path")) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    _emit({"status": "ok", "problem": str(problem)})


@app.command("make-demo-dataset")
def make_demo_dataset(
    out: Path = typer.Option(Path("data/demo"), help="Output dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    images: int = typer.Option(24, min=1, help="Number of synthetic images"),
    seed: int = typer.Option(1337, help="Random seed"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    create_demo_dataset(out, class_names(spec), image_count=images, seed=seed)
    _emit({"status": "ok", "dataset": str(out), "images": images})


@app.command("audit-dataset")
def audit_dataset_cmd(
    dataset: Path = typer.Argument(..., help="YOLO dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/audit"), help="Output report directory"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    report = audit_dataset(dataset, class_names(spec))
    write_audit_report(report, out)
    _emit({"status": "ok", "report": str(out / "audit.json"), "summary": report["summary"]})


@app.command("split-dataset")
def split_dataset_cmd(
    dataset: Path = typer.Argument(..., help="YOLO dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("data/splits"), help="Output split directory"),
    seed: int = typer.Option(1337, help="Random seed"),
    max_train: int | None = typer.Option(None, min=0, help="Optional cap for train images"),
    max_val: int | None = typer.Option(None, min=0, help="Optional cap for val images"),
    max_test: int | None = typer.Option(None, min=0, help="Optional cap for test images"),
    keep_empty: bool = typer.Option(True, help="Keep images with no target labels"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    records = load_records(dataset)
    if not keep_empty:
        records = [record for record in records if record.boxes]
    splits = stratified_split_records(records, split_ratios(spec), seed=seed)
    splits = {
        "train": _cap_records(splits["train"], max_train),
        "val": _cap_records(splits["val"], max_val),
        "test": _cap_records(splits["test"], max_test),
    }
    write_split_manifests(splits, out)
    write_ultralytics_dataset_yaml(
        out / "dataset.yaml",
        out / "train.txt",
        out / "val.txt",
        out / "test.txt",
        class_names(spec),
    )
    qa_rate = spec.get("dataset", {}).get("annotation_schema", {}).get("qa_review_rate", 0.15)
    qa = qa_sample(records, qa_rate, seed=seed)
    write_json(
        out / "qa_sample.json",
        [{"image": str(record.image_path), "label": str(record.label_path)} for record in qa],
    )
    _emit({"status": "ok", "out": str(out), "splits": {key: len(value) for key, value in splits.items()}})


@app.command("evaluate-yolo")
def evaluate_yolo_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    labels: Path = typer.Option(..., help="Ground-truth labels directory"),
    preds: Path = typer.Option(..., help="Prediction labels directory with confidence"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path | None = typer.Option(None, help="Optional metrics JSON output"),
    iou: float = typer.Option(0.5, help="IoU threshold"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    names = class_names(spec)
    result = evaluate_yolo_predictions(images, labels, preds, len(names), iou_threshold=iou, class_names=names)
    if out:
        write_json(out, result)
    _emit(result)


@app.command("benchmark-yolo")
def benchmark_yolo_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    labels: Path = typer.Option(..., help="Ground-truth labels directory"),
    preds: Path = typer.Option(..., help="Prediction labels directory with confidence"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path | None = typer.Option(None, help="Optional benchmark JSON output"),
    iou: float = typer.Option(0.5, help="IoU threshold"),
    profile_iterations: int = typer.Option(3, min=1, help="Preprocess profiler iterations"),
    width: int = typer.Option(640, min=1, help="Preprocess resize width"),
    height: int = typer.Option(640, min=1, help="Preprocess resize height"),
    manifest: Path | None = typer.Option(None, help="Optional image manifest to restrict benchmark"),
    max_images: int | None = typer.Option(None, min=1, help="Optional image limit"),
) -> None:
    result = benchmark_yolo_pipeline(
        images, labels, preds, problem,
        iou_threshold=iou, profile_iterations=profile_iterations,
        profile_size=(width, height), image_manifest=manifest, max_images=max_images,
    )
    if out:
        write_json(out, result)
    _emit(result)
    if result["status"] != "passed":
        raise typer.Exit(2)


@app.command("predict-yolo")
def predict_yolo_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    model: Path = typer.Option(Path("yolov8n.pt"), help="Ultralytics YOLO checkpoint"),
    out: Path = typer.Option(Path("artifacts/predictions"), help="Prediction output directory"),
    conf: float = typer.Option(0.001, min=0.0, max=1.0, help="Prediction confidence threshold"),
    iou: float = typer.Option(0.7, min=0.0, max=1.0, help="NMS IoU threshold"),
    device: str | None = typer.Option(None, help="Optional Ultralytics device"),
    max_images: int | None = typer.Option(None, min=1, help="Optional image limit"),
) -> None:
    result = predict_ultralytics_yolo(
        images, problem, model_path=model, out_dir=out,
        conf=conf, iou=iou, device=device, max_images=max_images,
    )
    _emit(result)
    if result["status"] != "ok":
        raise typer.Exit(2)


@app.command("profile-preprocess")
def profile_preprocess_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    iterations: int = typer.Option(3, min=1, help="Iterations over all images"),
    width: int = typer.Option(640, min=1, help="Resize width"),
    height: int = typer.Option(640, min=1, help="Resize height"),
    out: Path | None = typer.Option(None, help="Optional profile JSON output"),
    manifest: Path | None = typer.Option(None, help="Optional image manifest"),
    max_images: int | None = typer.Option(None, min=1, help="Optional image limit"),
) -> None:
    result = profile_preprocess(images, iterations=iterations, size=(width, height),
                                image_manifest=manifest, max_images=max_images)
    if out:
        write_json(out, result)
    _emit(result)


@app.command("export-errors")
def export_errors_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    labels: Path = typer.Option(..., help="Ground-truth labels directory"),
    preds: Path = typer.Option(..., help="Prediction labels directory with confidence"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/errors.json"), help="Output error JSON"),
    iou: float = typer.Option(0.5, help="IoU threshold"),
    manifest: Path | None = typer.Option(None, help="Optional image manifest"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    min_area = float(spec.get("dataset", {}).get("annotation_schema", {}).get("min_box_area_ratio", 0.0001))
    result = export_detection_errors(
        images, labels, preds, class_names(spec),
        iou_threshold=iou, small_area_ratio=min_area, image_manifest=manifest,
    )
    write_json(out, result)
    _emit({"status": "ok", "out": str(out), "summary": result["by_kind"]})


@app.command("native-info")
def native_info_cmd() -> None:
    _emit(native_status())


@app.command("doctor")
def doctor_cmd(
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    dataset: Path | None = typer.Option(None, help="Optional Ultralytics dataset YAML path"),
    train_config: Path = typer.Option(Path("configs/baseline.yolo.yaml"), help="Baseline config path"),
) -> None:
    report = project_health(problem, dataset, train_config, Path.cwd())
    _emit(report)
    if report["status"] != "ok":
        raise typer.Exit(2)


@app.command("prepare-coco")
def prepare_coco_cmd(
    images: Path = typer.Option(..., help="COCO images directory"),
    annotations: Path = typer.Option(..., help="COCO instances JSON path"),
    out: Path = typer.Option(Path("data/coco_road_yolo"), help="Output YOLO dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    include_empty: bool = typer.Option(True, help="Include images with no target-class boxes"),
    include_crowd: bool = typer.Option(False, help="Include COCO iscrowd boxes"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        _exit_with_errors(errors)
    min_area = float(spec.get("dataset", {}).get("annotation_schema", {}).get("min_box_area_ratio", 0.0))
    summary = prepare_coco_yolo(
        images, annotations, out, class_names(spec),
        include_empty=include_empty, include_crowd=include_crowd, min_box_area_ratio=min_area,
    )
    _emit({"status": "ok", "summary": summary})


@app.command("train-baseline")
def train_baseline_cmd(
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    dataset: Path = typer.Option(..., help="Ultralytics dataset YAML path"),
    train_config: Path = typer.Option(Path("configs/baseline.yolo.yaml"), help="Baseline config path"),
) -> None:
    try:
        result = train_yolo_baseline(problem, dataset, train_config)
    except RuntimeError as exc:
        _emit({"status": "blocked", "reason": str(exc)})
        raise typer.Exit(2) from exc
    _emit(result)


@app.command("hwinfo")
def hwinfo_cmd(
    out: Path = typer.Option(Path("reports/hwinfo.json"), help="Output JSON path"),
) -> None:
    report = hardware_report()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    _emit(report)


# ── New Commands ─────────────────────────────────────────────

@app.command("visualize-errors")
def visualize_errors_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    labels: Path = typer.Option(..., help="Ground-truth labels directory"),
    preds: Path = typer.Option(..., help="Prediction labels directory"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/visual_errors"), help="Output directory"),
    iou: float = typer.Option(0.5, help="IoU threshold"),
    max_images: int | None = typer.Option(None, min=1, help="Optional image limit"),
) -> None:
    from dtflowcv.visualize import save_annotated_errors
    spec = load_yaml(problem)
    result = save_annotated_errors(images, labels, preds, out, class_names(spec),
                                   iou_threshold=iou, max_images=max_images)
    _emit({"status": "ok", **result})


@app.command("error-grid")
def error_grid_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    labels: Path = typer.Option(..., help="Ground-truth labels directory"),
    preds: Path = typer.Option(..., help="Prediction labels directory"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/error_grid.jpg"), help="Output grid image"),
) -> None:
    from dtflowcv.visualize import make_error_grid
    spec = load_yaml(problem)
    result = make_error_grid(images, labels, preds, out, class_names(spec))
    _emit({"status": "ok", **result})


@app.command("infer-video")
def infer_video_cmd(
    source: str = typer.Option(..., help="Video file, RTSP URL, or device ID"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    model: Path = typer.Option(Path("yolov8n.pt"), help="YOLO checkpoint"),
    out_video: Path | None = typer.Option(None, help="Output annotated video"),
    out_json: Path | None = typer.Option(None, help="Output per-frame JSON"),
    conf: float = typer.Option(0.25, help="Detection confidence"),
    iou: float = typer.Option(0.45, help="NMS IoU"),
    tracking: bool = typer.Option(True, help="Enable tracking"),
    max_frames: int | None = typer.Option(None, min=1, help="Max frames"),
    sample_fps: float | None = typer.Option(None, help="Sample FPS"),
) -> None:
    from dtflowcv.infer import infer_video
    result = infer_video(
        source, problem, model_path=model,
        output_video=out_video, output_json=out_json,
        conf=conf, iou=iou, enable_tracking=tracking,
        max_frames=max_frames, sample_fps=sample_fps,
    )
    _emit(result)
    if result.get("status") != "ok":
        raise typer.Exit(2)


@app.command("infer-images")
def infer_images_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    model: Path = typer.Option(Path("yolov8n.pt"), help="YOLO checkpoint"),
    out: Path = typer.Option(Path("artifacts/annotated"), help="Annotated output dir"),
    conf: float = typer.Option(0.25, help="Detection confidence"),
    max_images: int | None = typer.Option(None, min=1, help="Max images"),
) -> None:
    from dtflowcv.infer import infer_images
    result = infer_images(images, problem, model_path=model, output_dir=out,
                          conf=conf, max_images=max_images)
    _emit(result)


@app.command("extract-frames")
def extract_frames_cmd(
    source: str = typer.Option(..., help="Video file path"),
    out: Path = typer.Option(Path("data/frames"), help="Output frame directory"),
    sample_fps: float | None = typer.Option(None, help="Sample FPS (null=native)"),
    max_frames: int | None = typer.Option(None, min=1, help="Max frames"),
    format: str = typer.Option("jpg", help="Image format"),
) -> None:
    from dtflowcv.video import extract_frames
    result = extract_frames(source, out, sample_fps=sample_fps, max_frames=max_frames, format=format)
    _emit({"status": "ok", **result})


@app.command("export-model")
def export_model_cmd(
    model: Path = typer.Option(..., help="YOLO .pt checkpoint"),
    out: Path = typer.Option(Path("artifacts/models/best.onnx"), help="Output path"),
    format: str = typer.Option("onnx", help="Export format: onnx, torchscript, engine"),
    input_size: int = typer.Option(640, help="Input image size"),
    half: bool = typer.Option(False, help="FP16 export"),
) -> None:
    from dtflowcv.export import export_engine, export_onnx, export_torchscript
    if format == "onnx":
        result = export_onnx(model, out, input_size=input_size, half=half)
    elif format == "torchscript":
        result = export_torchscript(model, out, input_size=input_size, half=half)
    elif format == "engine":
        result = export_engine(model, out, input_size=input_size, half=half)
    else:
        _emit({"status": "failed", "reason": f"Unknown format: {format}"})
        raise typer.Exit(2)
    _emit(result)
    if result.get("status") != "ok":
        raise typer.Exit(2)


@app.command("dataset-card")
def dataset_card_cmd(
    dataset: Path = typer.Argument(..., help="Dataset root directory"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/dataset_card"), help="Output directory"),
    name: str = typer.Option("dtflowcv-dataset", help="Dataset name"),
    verify: str | None = typer.Option(None, help="Expected SHA-256 to verify against"),
) -> None:
    from dtflowcv.data_card import build_dataset_card, verify_dataset_integrity, write_dataset_card
    spec = load_yaml(problem)
    names = class_names(spec)
    if verify:
        result = verify_dataset_integrity(dataset, verify)
        _emit(result)
        if not result["verified"]:
            raise typer.Exit(2)
    else:
        card = build_dataset_card(dataset, name, names)
        paths = write_dataset_card(card, out)
        _emit({"status": "ok", **paths, "sha256": card.sha256})


@app.command("env-check")
def env_check_cmd() -> None:
    from dtflowcv.deploy import environment_check
    result = environment_check()
    _emit(result)
    if not result["ready"]:
        raise typer.Exit(2)


def _cap_records(records: list, cap: int | None) -> list:
    if cap is None:
        return records
    return records[:cap]
