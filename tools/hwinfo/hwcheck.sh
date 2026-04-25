#!/bin/bash
#
# hwcheck.sh — Quick hardware check for dtflowcv pipeline
#
# Usage: bash tools/hwinfo/hwcheck.sh
#
# Works on any Linux system. No dependencies except bash and /proc.
# Also provides basic macOS support.
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}[ OK ]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

STATUS=0  # 0=OK, 1=WARN, 2=FAIL

echo "================================================================"
echo "  dtflowcv Hardware Check"
echo "================================================================"
echo

# ── OS ──────────────────────────────────────────────────────

echo "--- System ---"
if [[ -f /etc/os-release ]]; then
    source /etc/os-release 2>/dev/null || true
    echo "  OS:             ${PRETTY_NAME:-$(uname -s)}"
else
    echo "  OS:             $(uname -s) $(uname -r)"
fi
echo "  Kernel:         $(uname -r)"
echo "  Arch:           $(uname -m)"
echo "  Hostname:       $(hostname 2>/dev/null || echo unknown)"
echo

# ── CPU ─────────────────────────────────────────────────────

echo "--- CPU ---"
ARCH=$(uname -m)
CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)
echo "  Cores:          ${CORES}"

if [[ -f /proc/cpuinfo ]]; then
    BRAND=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || echo "unknown")
    VENDOR=$(grep -m1 'vendor_id' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || echo "unknown")
    echo "  Vendor:         ${VENDOR}"
    echo "  Brand:          ${BRAND}"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    echo "  Brand:          ${BRAND}"
fi

# Frequency
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ]]; then
    FREQ_KHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
    echo "  Max freq:       $((FREQ_KHZ / 1000)) MHz"
elif [[ -f /proc/cpuinfo ]]; then
    FREQ=$(grep -m1 'cpu MHz' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || echo "")
    if [[ -n "$FREQ" ]]; then
        echo "  Freq:           ${FREQ} MHz"
    fi
fi

echo

# ── SIMD Features ──────────────────────────────────────────

echo "--- SIMD Features ---"
if [[ -f /proc/cpuinfo ]]; then
    FLAGS=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo "")

    check_flag() {
        local flag=$1
        local display=$2
        if echo "$FLAGS" | grep -qw "$flag"; then
            echo "  ${display}: YES"
        else
            echo "  ${display}: NO"
        fi
    }

    if [[ "$ARCH" == "x86_64" || "$ARCH" == "x86" ]]; then
        check_flag "sse2"    "SSE2     "
        check_flag "sse3"    "SSE3     "
        check_flag "ssse3"   "SSSE3    "
        check_flag "sse4_1"  "SSE4.1   "
        check_flag "sse4_2"  "SSE4.2   "
        check_flag "avx"     "AVX      "
        check_flag "avx2"    "AVX2     "
        check_flag "fma"     "FMA3     "
        check_flag "avx512f" "AVX-512F "
        check_flag "avx512bw" "AVX-512BW"
        check_flag "avx512vl" "AVX-512VL"
    elif [[ "$ARCH" == "aarch64" ]]; then
        check_flag "asimd"   "NEON/ASIMD"
        check_flag "fp"      "FP       "
        check_flag "fphp"    "FP16     "
        check_flag "sve"     "SVE      "
        check_flag "sve2"    "SVE2     "
    fi
else
    echo "  (no /proc/cpuinfo — cannot detect SIMD)"
fi

echo

# ── Cache ──────────────────────────────────────────────────

echo "--- Cache ---"
CACHE_BASE="/sys/devices/system/cpu/cpu0/cache"
if [[ -d "$CACHE_BASE" ]]; then
    for idx in $(seq 0 7); do
        INDEX_DIR="${CACHE_BASE}/index${idx}"
        [[ -d "$INDEX_DIR" ]] || break
        TYPE=$(cat "${INDEX_DIR}/type" 2>/dev/null || echo "")
        SIZE=$(cat "${INDEX_DIR}/size" 2>/dev/null || echo "")
        LEVEL=$(cat "${INDEX_DIR}/level" 2>/dev/null || echo "")
        LINE=$(cat "${INDEX_DIR}/coherency_line_size" 2>/dev/null || echo "")
        echo "  L${LEVEL} ${TYPE}: ${SIZE} (line: ${LINE} bytes)"
    done
else
    echo "  (no sysfs cache info available)"
fi
echo

# ── Memory ─────────────────────────────────────────────────

echo "--- Memory ---"
if [[ -f /proc/meminfo ]]; then
    TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    AVAIL_KB=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
    TOTAL_MB=$((TOTAL_KB / 1024))
    AVAIL_MB=$((AVAIL_KB / 1024))
    echo "  Total:          ${TOTAL_MB} MB"
    echo "  Available:      ${AVAIL_MB} MB"

    # Huge pages
    HP_SIZE=$(grep Hugepagesize /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "")
    if [[ -n "$HP_SIZE" && "$HP_SIZE" -gt 0 ]] 2>/dev/null; then
        echo "  Huge pages:     ${HP_SIZE} KB"
    fi
elif [[ "$(uname -s)" == "Darwin" ]]; then
    TOTAL_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    TOTAL_MB=$((TOTAL_BYTES / 1024 / 1024))
    AVAIL_MB=0
    echo "  Total:          ${TOTAL_MB} MB"
fi
echo

# ── NUMA ───────────────────────────────────────────────────

if [[ -f /sys/devices/system/node/online ]]; then
    NUMA_ONLINE=$(cat /sys/devices/system/node/online)
    echo "--- NUMA ---"
    echo "  Nodes:          ${NUMA_ONLINE}"
    echo
fi

# ── GPU ────────────────────────────────────────────────────

echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [[ -n "$GPU_INFO" ]]; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        echo "  CUDA devices:   ${GPU_COUNT}"
        echo "  Device 0:       ${GPU_INFO}"
    else
        echo "  CUDA:           nvidia-smi found but no devices"
    fi
else
    echo "  CUDA:           nvidia-smi not found"
fi

# ROCm
if command -v rocm-smi &>/dev/null; then
    echo "  ROCm:           detected"
fi

echo

# ── Suitability ────────────────────────────────────────────

echo "--- Suitability ---"
echo

# Cores check
if [[ "$CORES" -lt 2 ]]; then
    fail "CPU cores: ${CORES}. Minimum: 2."
    STATUS=2
elif [[ "$CORES" -lt 4 ]]; then
    warn "CPU cores: ${CORES}. Recommend >= 4."
    [[ "$STATUS" -lt 1 ]] && STATUS=1
else
    pass "CPU cores: ${CORES}"
fi

# RAM check
if [[ -n "${TOTAL_MB:-}" ]]; then
    if [[ "$TOTAL_MB" -lt 2048 ]]; then
        fail "RAM: ${TOTAL_MB} MB. Minimum: 2048 MB."
        STATUS=2
    elif [[ "$TOTAL_MB" -lt 8192 ]]; then
        warn "RAM: ${TOTAL_MB} MB. Recommend >= 8192 MB for training."
        [[ "$STATUS" -lt 1 ]] && STATUS=1
    else
        pass "RAM: ${TOTAL_MB} MB"
    fi
fi

# SIMD check
if [[ "$ARCH" == "x86_64" ]]; then
    if echo "${FLAGS:-}" | grep -qw "sse2"; then
        pass "SSE2 available"
    else
        fail "SSE2 not detected"
        STATUS=2
    fi
    if echo "${FLAGS:-}" | grep -qw "avx2"; then
        pass "AVX2 available"
    else
        warn "No AVX2 — SSE2 kernel will be used"
        [[ "$STATUS" -lt 1 ]] && STATUS=1
    fi
    if echo "${FLAGS:-}" | grep -qw "avx512f"; then
        info "AVX-512 detected"
    fi
elif [[ "$ARCH" == "aarch64" ]]; then
    pass "AArch64 — NEON mandatory"
    if echo "${FLAGS:-}" | grep -qw "sve"; then
        info "SVE available"
    fi
fi

echo
case $STATUS in
    0) echo -e "  ${GREEN}Overall: READY${NC}" ;;
    1) echo -e "  ${YELLOW}Overall: READY (with warnings)${NC}" ;;
    2) echo -e "  ${RED}Overall: NOT READY${NC}" ;;
esac

echo

# ── Recommendations ────────────────────────────────────────

echo "--- Recommendations ---"
echo

WORKERS=$((CORES / 2))
[[ "$WORKERS" -lt 1 ]] && WORKERS=1
[[ "$WORKERS" -gt 8 ]] && WORKERS=8
echo "  workers:        ${WORKERS}"

BATCH=16
if [[ -n "${TOTAL_MB:-}" ]]; then
    if [[ "$TOTAL_MB" -ge 32768 ]]; then BATCH=64
    elif [[ "$TOTAL_MB" -ge 16384 ]]; then BATCH=32
    elif [[ "$TOTAL_MB" -ge 8192 ]]; then BATCH=16
    else BATCH=8; fi
fi
echo "  batch_size:     ${BATCH}"

IMGSZ=640
[[ -n "${TOTAL_MB:-}" && "$TOTAL_MB" -lt 4096 ]] && IMGSZ=416
echo "  image_size:     ${IMGSZ}"

if [[ "$ARCH" == "x86_64" ]]; then
    if echo "${FLAGS:-}" | grep -qw "avx512f"; then echo "  simd_kernel:    avx512"
    elif echo "${FLAGS:-}" | grep -qw "avx2"; then echo "  simd_kernel:    avx2"
    else echo "  simd_kernel:    sse2"; fi
elif [[ "$ARCH" == "aarch64" ]]; then
    if echo "${FLAGS:-}" | grep -qw "sve"; then echo "  simd_kernel:    sve"
    else echo "  simd_kernel:    neon"; fi
else
    echo "  simd_kernel:    scalar"
fi

echo

exit $STATUS
