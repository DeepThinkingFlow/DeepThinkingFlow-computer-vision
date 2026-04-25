from __future__ import annotations

import importlib.util
import platform
import shutil
from typing import Any

NATIVE_CLAIM_BOUNDARY = (
    "Native source is present; performance is unclaimed until extension build and benchmark pass on target hardware."
)


def native_status() -> dict[str, Any]:
    spec = importlib.util.find_spec("dtflowcv_native")
    module_available = spec is not None
    status: dict[str, Any] = {
        "python_extension_importable": module_available,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nasm_available": shutil.which("nasm") is not None,
        "cpu_avx512f_flag": _linux_cpu_flag("avx512f"),
        "cpu_sse2_flag": _linux_cpu_flag("sse2"),
        "claim_boundary": NATIVE_CLAIM_BOUNDARY,
    }
    if module_available:
        import dtflowcv_native  # type: ignore

        if hasattr(dtflowcv_native, "capabilities"):
            status["extension_capabilities"] = dtflowcv_native.capabilities()
    return status


def hardware_report() -> dict[str, Any]:
    """Full hardware detection report via native C hwinfo.
    Falls back to Python-only detection if native module is unavailable."""
    spec = importlib.util.find_spec("dtflowcv_native")
    if spec is not None:
        import dtflowcv_native  # type: ignore

        if hasattr(dtflowcv_native, "hardware_info"):
            return dtflowcv_native.hardware_info()

    # Fallback: Python-only detection
    return _python_hw_fallback()


def _python_hw_fallback() -> dict[str, Any]:
    """Basic hardware detection using only Python stdlib."""
    import os

    cpu: dict[str, Any] = {
        "arch": platform.machine(),
        "vendor": "unknown",
        "brand": platform.processor() or "unknown",
        "physical_cores": os.cpu_count() or 1,
        "logical_cores": os.cpu_count() or 1,
        "simd": {},
        "cache": {},
    }

    # Linux: parse /proc/cpuinfo
    flags = _linux_cpu_flags()
    if flags is not None:
        simd_names = [
            "sse2", "sse3", "ssse3", "sse4_1", "sse4_2",
            "avx", "avx2", "fma",
            "avx512f", "avx512bw", "avx512vl",
            "neon", "asimd", "sve", "sve2",
        ]
        cpu["simd"] = {name: name in flags for name in simd_names}
        vendor = _linux_cpu_field("vendor_id")
        if vendor:
            cpu["vendor"] = vendor
        brand = _linux_cpu_field("model name")
        if brand:
            cpu["brand"] = brand

    # Memory
    mem: dict[str, Any] = {"total_bytes": 0, "available_bytes": 0}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem["total_bytes"] = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem["available_bytes"] = int(line.split()[1]) * 1024
    except OSError:
        pass

    return {
        "cpu": cpu,
        "memory": mem,
        "gpu": {"cuda_device_count": 0},
        "os": {
            "name": platform.system(),
            "release": platform.release(),
            "hostname": platform.node(),
        },
        "suitability": {"status": "UNKNOWN", "messages": ["native module not available"]},
        "recommendations": "Build native module for full hardware detection.",
    }


def _linux_cpu_flag(flag: str) -> bool | None:
    cpuinfo = "/proc/cpuinfo"
    try:
        with open(cpuinfo, encoding="utf-8") as fh:
            return flag in fh.read().split()
    except OSError:
        return None


def _linux_cpu_flags() -> set[str] | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    colon = line.index(":")
                    return set(line[colon + 1:].split())
    except (OSError, ValueError):
        pass
    return None


def _linux_cpu_field(name: str) -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(name):
                    colon = line.index(":")
                    return line[colon + 1:].strip()
    except (OSError, ValueError):
        pass
    return None
