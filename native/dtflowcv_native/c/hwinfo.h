/*
 * hwinfo.h — Cross-platform hardware detection for dtflowcv
 *
 * Supports: x86-64 (Intel/AMD), AArch64 (ARM64), generic POSIX
 * Compilers: GCC, Clang, MSVC
 * OS: Linux, macOS, Windows
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

/* ── CPU identification ──────────────────────────────────── */

typedef struct {
    char vendor[64];            /* e.g. "GenuineIntel", "AuthenticAMD" */
    char brand[128];            /* full brand string                   */
    char arch[32];              /* "x86_64", "aarch64", "unknown"      */
    int  physical_cores;
    int  logical_cores;
    int  numa_nodes;            /* NUMA node count (0 if unknown)      */

    /* x86-64 SIMD feature flags */
    int  has_sse2;
    int  has_sse3;
    int  has_ssse3;
    int  has_sse41;
    int  has_sse42;
    int  has_avx;
    int  has_avx2;
    int  has_fma3;
    int  has_avx512f;
    int  has_avx512bw;
    int  has_avx512vl;

    /* AArch64 feature flags */
    int  has_neon;
    int  has_sve;
    int  has_sve2;
    int  has_fp16;

    /* Cache sizes in bytes (0 if unknown) */
    size_t l1d_cache_bytes;
    size_t l1i_cache_bytes;
    size_t l2_cache_bytes;
    size_t l3_cache_bytes;
    size_t cache_line_bytes;

    /* Base frequency MHz (0 if unknown) */
    int  base_freq_mhz;
} dtflowcv_cpu_info_t;

/* ── Memory information ──────────────────────────────────── */

typedef struct {
    size_t total_ram_bytes;
    size_t available_ram_bytes;
    size_t page_size_bytes;
    int    huge_pages_supported;
    size_t huge_page_size_bytes;
} dtflowcv_mem_info_t;

/* ── GPU information (basic) ─────────────────────────────── */

typedef struct {
    int    cuda_device_count;
    char   cuda_device_name[128];
    size_t cuda_total_mem_bytes;
    int    cuda_compute_major;
    int    cuda_compute_minor;
    int    opencl_available;
} dtflowcv_gpu_info_t;

/* ── OS information ──────────────────────────────────────── */

typedef struct {
    char os_name[64];           /* "Linux", "Darwin", "Windows" */
    char os_release[128];
    char hostname[128];
    int  pid;
} dtflowcv_os_info_t;

/* ── Pipeline suitability ────────────────────────────────── */

#define DTFLOWCV_HW_OK           0
#define DTFLOWCV_HW_WARN         1
#define DTFLOWCV_HW_INSUFFICIENT 2

typedef struct {
    int    overall_status;      /* DTFLOWCV_HW_OK / WARN / INSUFFICIENT */
    int    cpu_ok;
    int    ram_ok;
    int    simd_ok;
    char   messages[8][256];
    int    message_count;
} dtflowcv_suitability_t;

/* ── API ─────────────────────────────────────────────────── */

void dtflowcv_detect_cpu(dtflowcv_cpu_info_t *out);
void dtflowcv_detect_mem(dtflowcv_mem_info_t *out);
void dtflowcv_detect_gpu(dtflowcv_gpu_info_t *out);
void dtflowcv_detect_os(dtflowcv_os_info_t *out);

/* Evaluate if system meets minimum requirements for this pipeline. */
void dtflowcv_check_suitability(
    const dtflowcv_cpu_info_t *cpu,
    const dtflowcv_mem_info_t *mem,
    dtflowcv_suitability_t *out);

/* Print full hardware report to stdout. */
void dtflowcv_print_hw_report(void);

/*
 * Recommended pipeline settings based on detected hardware.
 * output buffer must be >= 4096 bytes.
 */
void dtflowcv_hw_recommendations(
    const dtflowcv_cpu_info_t *cpu,
    const dtflowcv_mem_info_t *mem,
    const dtflowcv_gpu_info_t *gpu,
    char *output,
    size_t output_size);
