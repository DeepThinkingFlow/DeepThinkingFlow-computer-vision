from pathlib import Path

from dtflowcv.benchmark import benchmark_yolo_pipeline
from dtflowcv.config import write_json
from dtflowcv.demo import create_demo_dataset


def test_benchmark_yolo_pipeline_passes_demo_predictions(tmp_path: Path) -> None:
    problem = tmp_path / "problem.yaml"
    problem.write_text(
        """
project:
  task_type: detection
phase_1:
  input:
    modality: image
    color_space: RGB
    dtype: uint8
    average_resolution: smoke
    noise_profile: synthetic
    lighting: synthetic
    source: generated
  output:
    type: bounding_box
    training_format: yolo
    prediction_format: yolo_conf
    coordinate_system: normalized
  metrics:
    primary: map50
    secondary: [fps, latency_p99_ms]
    acceptance:
      map50_min: 0.90
      latency_p99_ms_max: 1000.0
      fps_min: 1.0
dataset:
  public_benchmark:
    name: synthetic smoke
  annotation_schema:
    min_box_area_ratio: 0.0001
  split:
    train: 0.70
    val: 0.15
    test: 0.15
classes:
  - person
  - car
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, ["person", "car"], image_count=4, seed=19)

    result = benchmark_yolo_pipeline(
        dataset / "images",
        dataset / "labels",
        dataset / "predictions",
        problem,
        profile_iterations=1,
        profile_size=(64, 64),
    )

    assert result["status"] == "passed"
    assert result["acceptance"]["passed"] is True
    assert result["detection"]["map"] >= 0.90


def test_benchmark_yolo_pipeline_blocks_invalid_spec(tmp_path: Path) -> None:
    problem = tmp_path / "invalid.yaml"
    write_json(problem, {"project": {"task_type": "classification"}})

    result = benchmark_yolo_pipeline(tmp_path, tmp_path, tmp_path, problem)

    assert result["status"] == "blocked"
    assert result["build_blockers"]


def test_benchmark_yolo_pipeline_respects_max_images(tmp_path: Path) -> None:
    problem = tmp_path / "problem.yaml"
    problem.write_text(
        """
project:
  task_type: detection
phase_1:
  input:
    modality: image
    color_space: RGB
    dtype: uint8
    average_resolution: smoke
    noise_profile: synthetic
    lighting: synthetic
    source: generated
  output:
    type: bounding_box
    training_format: yolo
    prediction_format: yolo_conf
    coordinate_system: normalized
  metrics:
    primary: map50
    secondary: [fps, latency_p99_ms]
    acceptance:
      map50_min: 0.90
      latency_p99_ms_max: 1000.0
      fps_min: 1.0
dataset:
  public_benchmark:
    name: synthetic smoke
  annotation_schema:
    min_box_area_ratio: 0.0001
  split:
    train: 0.70
    val: 0.15
    test: 0.15
classes:
  - person
  - car
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, ["person", "car"], image_count=5, seed=23)

    result = benchmark_yolo_pipeline(
        dataset / "images",
        dataset / "labels",
        dataset / "predictions",
        problem,
        profile_iterations=1,
        profile_size=(64, 64),
        max_images=2,
    )

    assert result["detection"]["image_count"] == 2
    assert result["preprocess"]["image_count"] == 2
