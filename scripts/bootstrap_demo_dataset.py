from __future__ import annotations

from pathlib import Path

from dtflowcv.config import load_yaml
from dtflowcv.demo import create_demo_dataset
from dtflowcv.specs import class_names


def main() -> None:
    problem = load_yaml("configs/problem.yaml")
    create_demo_dataset(Path("data/demo"), class_names(problem), image_count=24)


if __name__ == "__main__":
    main()
