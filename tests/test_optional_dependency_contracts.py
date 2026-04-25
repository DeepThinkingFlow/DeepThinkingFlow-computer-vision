from dtflowcv.deps import blocked_payload, missing_optional_blockers


def test_missing_optional_dependency_blocker_shape() -> None:
    blockers = missing_optional_blockers(["definitely_missing_dtflowcv_dependency"])

    assert blockers == [
        "missing_python_module:definitely_missing_dtflowcv_dependency: install with install the required extra"
    ]
    assert blocked_payload(blockers)["status"] == "blocked"
