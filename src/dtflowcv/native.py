from __future__ import annotations

import importlib.util
import platform
import shutil
from typing import Any


def native_status() -> dict[str, Any]:
    spec = importlib.util.find_spec("dtflowcv_native")
    module_available = spec is not None
    status: dict[str, Any] = {
        "python_extension_importable": module_available,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nasm_available": shutil.which("nasm") is not None,
        "cpu_avx512f_flag": _linux_cpu_flag("avx512f"),
        "claim_boundary": "Native source is present; performance is unclaimed until extension build and benchmark pass on target hardware.",
    }
    if module_available:
        import dtflowcv_native  # type: ignore

        if hasattr(dtflowcv_native, "capabilities"):
            status["extension_capabilities"] = dtflowcv_native.capabilities()
    return status


def _linux_cpu_flag(flag: str) -> bool | None:
    cpuinfo = "/proc/cpuinfo"
    try:
        with open(cpuinfo, "r", encoding="utf-8") as fh:
            return flag in fh.read().split()
    except OSError:
        return None
