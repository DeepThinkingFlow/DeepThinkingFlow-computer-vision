from dtflowcv.train import _runtime_blockers


def test_runtime_blockers_allow_cpu_device() -> None:
    assert _runtime_blockers({"training": {"device": "cpu"}}) == []


def test_runtime_blockers_fail_closed_for_requested_cuda_without_cuda() -> None:
    blockers = _runtime_blockers({"training": {"device": 0}})
    assert not blockers or blockers[0].startswith("gpu_device_requested_but_")
