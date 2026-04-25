from dtflowcv.config import load_yaml
from dtflowcv.specs import class_names, validate_problem_spec


def test_problem_spec_is_complete() -> None:
    spec = load_yaml("configs/problem.yaml")
    assert validate_problem_spec(spec) == []
    assert class_names(spec)[:3] == ["person", "bicycle", "car"]
