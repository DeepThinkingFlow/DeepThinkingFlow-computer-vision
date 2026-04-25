from __future__ import annotations

import json
from pathlib import Path

import typer

from dtflowcv.config import load_yaml, write_json
from dtflowcv.coco import prepare_coco_yolo, write_ultralytics_dataset_yaml
from dtflowcv.dataset import audit_dataset, load_records, qa_sample, stratified_split_records, write_audit_report, write_split_manifests
from dtflowcv.demo import create_demo_dataset
from dtflowcv.errors import export_detection_errors
from dtflowcv.evaluate import evaluate_yolo_predictions
from dtflowcv.native import native_status
from dtflowcv.profile import profile_preprocess
from dtflowcv.specs import class_names, split_ratios, validate_problem_spec
from dtflowcv.train import train_yolo_baseline

app = typer.Typer(no_args_is_help=True)


@app.command("check-spec")
def check_spec(problem: Path = typer.Argument(..., help="Problem YAML path")) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        typer.echo(json.dumps({"status": "failed", "errors": errors}, indent=2))
        raise typer.Exit(1)
    typer.echo(json.dumps({"status": "ok", "problem": str(problem)}, indent=2))


@app.command("make-demo-dataset")
def make_demo_dataset(
    out: Path = typer.Option(Path("data/demo"), help="Output dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    images: int = typer.Option(24, min=1, help="Number of synthetic images"),
    seed: int = typer.Option(1337, help="Random seed"),
) -> None:
    spec = load_yaml(problem)
    create_demo_dataset(out, class_names(spec), image_count=images, seed=seed)
    typer.echo(json.dumps({"status": "ok", "dataset": str(out), "images": images}, indent=2))


@app.command("audit-dataset")
def audit_dataset_cmd(
    dataset: Path = typer.Argument(..., help="YOLO dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/audit"), help="Output report directory"),
) -> None:
    spec = load_yaml(problem)
    errors = validate_problem_spec(spec)
    if errors:
        typer.echo(json.dumps({"status": "failed", "errors": errors}, indent=2))
        raise typer.Exit(1)
    report = audit_dataset(dataset, class_names(spec))
    write_audit_report(report, out)
    typer.echo(json.dumps({"status": "ok", "report": str(out / "audit.json"), "summary": report["summary"]}, indent=2))


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
    qa = qa_sample(records, spec.get("dataset", {}).get("annotation_schema", {}).get("qa_review_rate", 0.15), seed=seed)
    write_json(out / "qa_sample.json", [{"image": str(record.image_path), "label": str(record.label_path)} for record in qa])
    typer.echo(json.dumps({"status": "ok", "out": str(out), "splits": {key: len(value) for key, value in splits.items()}}, indent=2))


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
    result = evaluate_yolo_predictions(images, labels, preds, len(class_names(spec)), iou_threshold=iou)
    if out:
        write_json(out, result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("profile-preprocess")
def profile_preprocess_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    iterations: int = typer.Option(3, min=1, help="Iterations over all images"),
    width: int = typer.Option(640, min=1, help="Resize width"),
    height: int = typer.Option(640, min=1, help="Resize height"),
    out: Path | None = typer.Option(None, help="Optional profile JSON output"),
) -> None:
    result = profile_preprocess(images, iterations=iterations, size=(width, height))
    if out:
        write_json(out, result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("export-errors")
def export_errors_cmd(
    images: Path = typer.Option(..., help="Image directory"),
    labels: Path = typer.Option(..., help="Ground-truth labels directory"),
    preds: Path = typer.Option(..., help="Prediction labels directory with confidence"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    out: Path = typer.Option(Path("reports/errors.json"), help="Output error JSON"),
    iou: float = typer.Option(0.5, help="IoU threshold"),
    manifest: Path | None = typer.Option(None, help="Optional image manifest to restrict evaluation"),
) -> None:
    spec = load_yaml(problem)
    min_area = float(spec.get("dataset", {}).get("annotation_schema", {}).get("min_box_area_ratio", 0.0001))
    result = export_detection_errors(
        images,
        labels,
        preds,
        class_names(spec),
        iou_threshold=iou,
        small_area_ratio=min_area,
        image_manifest=manifest,
    )
    write_json(out, result)
    typer.echo(json.dumps({"status": "ok", "out": str(out), "summary": result["by_kind"]}, indent=2, sort_keys=True))


@app.command("native-info")
def native_info_cmd() -> None:
    typer.echo(json.dumps(native_status(), indent=2, sort_keys=True))


@app.command("prepare-coco")
def prepare_coco_cmd(
    images: Path = typer.Option(..., help="COCO images directory, for example val2017"),
    annotations: Path = typer.Option(..., help="COCO instances JSON path"),
    out: Path = typer.Option(Path("data/coco_road_yolo"), help="Output YOLO dataset root"),
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    include_empty: bool = typer.Option(True, help="Include images with no target-class boxes"),
    include_crowd: bool = typer.Option(False, help="Include COCO iscrowd boxes"),
) -> None:
    spec = load_yaml(problem)
    min_area = float(spec.get("dataset", {}).get("annotation_schema", {}).get("min_box_area_ratio", 0.0))
    summary = prepare_coco_yolo(
        images,
        annotations,
        out,
        class_names(spec),
        include_empty=include_empty,
        include_crowd=include_crowd,
        min_box_area_ratio=min_area,
    )
    typer.echo(json.dumps({"status": "ok", "summary": summary}, indent=2, sort_keys=True))


@app.command("train-baseline")
def train_baseline_cmd(
    problem: Path = typer.Option(Path("configs/problem.yaml"), help="Problem YAML path"),
    dataset: Path = typer.Option(..., help="Ultralytics dataset YAML path"),
    train_config: Path = typer.Option(Path("configs/baseline.yolo.yaml"), help="Baseline config path"),
) -> None:
    try:
        result = train_yolo_baseline(problem, dataset, train_config)
    except RuntimeError as exc:
        typer.echo(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2))
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def _cap_records(records: list, cap: int | None) -> list:
    if cap is None:
        return records
    return records[:cap]
