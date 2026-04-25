/*
 * hwinfo.c — Cross-platform hardware detection implementation
 *
 * Portable across: x86-64, AArch64, Linux, macOS, Windows
 * Compilers:       GCC ≥4.9, Clang ≥3.8, MSVC ≥2019
 */
#define _POSIX_C_SOURCE 200809L
#include "hwinfo.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── Platform includes ───────────────────────────────────── */

#if defined(_WIN32) || defined(_WIN64)
  #define DTFLOWCV_WINDOWS 1
  #include <windows.h>
  #include <intrin.h>
#else
  #define DTFLOWCV_POSIX 1
  #include <unistd.h>
  #include <sys/utsname.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define DTFLOWCV_X86 1
  #if !defined(DTFLOWCV_WINDOWS)
    #include <cpuid.h>
  #endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
  #define DTFLOWCV_AARCH64 1
  #if defined(__linux__)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
  #endif
#endif

/* ── Helpers ─────────────────────────────────────────────── */

static void safe_copy(char *dst, const char *src, size_t dst_size) {
    if (!src || !dst || dst_size == 0) return;
    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

static void append_msg(dtflowcv_suitability_t *s, const char *msg) {
    if (s->message_count < 8) {
        safe_copy(s->messages[s->message_count], msg, 256);
        s->message_count++;
    }
}

/* ── x86 CPUID wrapper ──────────────────────────────────── */

#if defined(DTFLOWCV_X86)
static void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
{
#if defined(DTFLOWCV_WINDOWS)
    int info[4];
    __cpuidex(info, (int)leaf, (int)subleaf);
    *eax = (uint32_t)info[0];
    *ebx = (uint32_t)info[1];
    *ecx = (uint32_t)info[2];
    *edx = (uint32_t)info[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}
#endif

/* ── CPU Detection ──────────────────────────────────────── */

void dtflowcv_detect_cpu(dtflowcv_cpu_info_t *out) {
    memset(out, 0, sizeof(*out));

    /* Architecture */
#if defined(__x86_64__) || defined(_M_X64)
    safe_copy(out->arch, "x86_64", sizeof(out->arch));
#elif defined(__aarch64__) || defined(_M_ARM64)
    safe_copy(out->arch, "aarch64", sizeof(out->arch));
#elif defined(__arm__) || defined(_M_ARM)
    safe_copy(out->arch, "arm32", sizeof(out->arch));
#elif defined(__riscv)
    safe_copy(out->arch, "riscv", sizeof(out->arch));
#elif defined(__powerpc64__)
    safe_copy(out->arch, "ppc64", sizeof(out->arch));
#else
    safe_copy(out->arch, "unknown", sizeof(out->arch));
#endif

    /* ── x86 CPUID-based detection ───────────────────── */
#if defined(DTFLOWCV_X86)
    {
        uint32_t eax, ebx, ecx, edx;

        /* Vendor string: CPUID leaf 0 */
        cpuid(0, 0, &eax, &ebx, &ecx, &edx);
        uint32_t max_leaf = eax;
        char vendor_buf[13];
        memcpy(vendor_buf + 0, &ebx, 4);
        memcpy(vendor_buf + 4, &edx, 4);
        memcpy(vendor_buf + 8, &ecx, 4);
        vendor_buf[12] = '\0';
        safe_copy(out->vendor, vendor_buf, sizeof(out->vendor));

        /* Feature flags: CPUID leaf 1 */
        if (max_leaf >= 1) {
            cpuid(1, 0, &eax, &ebx, &ecx, &edx);
            out->has_sse2  = (edx >> 26) & 1;  /* bit 26 of EDX */
            out->has_sse3  = (ecx >>  0) & 1;  /* bit  0 of ECX */
            out->has_ssse3 = (ecx >>  9) & 1;  /* bit  9 of ECX */
            out->has_sse41 = (ecx >> 19) & 1;  /* bit 19 of ECX */
            out->has_sse42 = (ecx >> 20) & 1;  /* bit 20 of ECX */
            out->has_avx   = (ecx >> 28) & 1;  /* bit 28 of ECX */
            out->has_fma3  = (ecx >> 12) & 1;  /* bit 12 of ECX */
        }

        /* Extended features: CPUID leaf 7 */
        if (max_leaf >= 7) {
            cpuid(7, 0, &eax, &ebx, &ecx, &edx);
            out->has_avx2     = (ebx >>  5) & 1;  /* bit  5 of EBX */
            out->has_avx512f  = (ebx >> 16) & 1;  /* bit 16 of EBX */
            out->has_avx512bw = (ebx >> 30) & 1;  /* bit 30 of EBX */
            out->has_avx512vl = (ebx >> 31) & 1;  /* bit 31 of EBX */
        }

        /* Brand string: CPUID leaves 0x80000002–0x80000004 */
        cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);
        if (eax >= 0x80000004) {
            uint32_t brand[12];
            cpuid(0x80000002, 0, &brand[0], &brand[1], &brand[2], &brand[3]);
            cpuid(0x80000003, 0, &brand[4], &brand[5], &brand[6], &brand[7]);
            cpuid(0x80000004, 0, &brand[8], &brand[9], &brand[10], &brand[11]);
            char brand_str[49];
            memcpy(brand_str, brand, 48);
            brand_str[48] = '\0';
            /* Trim leading spaces */
            const char *start = brand_str;
            while (*start == ' ') start++;
            safe_copy(out->brand, start, sizeof(out->brand));
        }

        /* Cache info: CPUID leaf 4 (Intel deterministic cache) */
        if (max_leaf >= 4) {
            for (int idx = 0; idx < 16; idx++) {
                cpuid(4, (uint32_t)idx, &eax, &ebx, &ecx, &edx);
                int type = eax & 0x1F;
                if (type == 0) break;  /* no more caches */
                int level = (eax >> 5) & 0x7;
                uint32_t line_size    = (ebx & 0xFFF) + 1;
                uint32_t partitions   = ((ebx >> 12) & 0x3FF) + 1;
                uint32_t ways         = ((ebx >> 22) & 0x3FF) + 1;
                uint32_t sets         = ecx + 1;
                size_t cache_size = (size_t)line_size * partitions * ways * sets;

                out->cache_line_bytes = line_size;
                if (level == 1 && type == 1)      out->l1d_cache_bytes = cache_size;
                else if (level == 1 && type == 2)  out->l1i_cache_bytes = cache_size;
                else if (level == 2)               out->l2_cache_bytes  = cache_size;
                else if (level == 3)               out->l3_cache_bytes  = cache_size;
            }
        }

        /* AMD cache info: CPUID leaf 0x8000001D */
        if (out->l1d_cache_bytes == 0) {
            cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);
            if (eax >= 0x8000001D) {
                for (int idx = 0; idx < 16; idx++) {
                    cpuid(0x8000001D, (uint32_t)idx, &eax, &ebx, &ecx, &edx);
                    int type = eax & 0x1F;
                    if (type == 0) break;
                    int level = (eax >> 5) & 0x7;
                    uint32_t line_size  = (ebx & 0xFFF) + 1;
                    uint32_t partitions = ((ebx >> 12) & 0x3FF) + 1;
                    uint32_t ways       = ((ebx >> 22) & 0x3FF) + 1;
                    uint32_t sets       = ecx + 1;
                    size_t cache_size = (size_t)line_size * partitions * ways * sets;

                    out->cache_line_bytes = line_size;
                    if (level == 1 && type == 1)      out->l1d_cache_bytes = cache_size;
                    else if (level == 1 && type == 2)  out->l1i_cache_bytes = cache_size;
                    else if (level == 2)               out->l2_cache_bytes  = cache_size;
                    else if (level == 3)               out->l3_cache_bytes  = cache_size;
                }
            }
        }
    }
#endif

    /* ── AArch64 feature detection ───────────────────── */
#if defined(DTFLOWCV_AARCH64)
    safe_copy(out->vendor, "ARM", sizeof(out->vendor));
    out->has_neon = 1;  /* NEON is mandatory on AArch64 */
  #if defined(__linux__)
    {
        unsigned long hwcap = getauxval(AT_HWCAP);
        out->has_fp16 = (hwcap & HWCAP_FPHP) ? 1 : 0;
        out->has_sve  = (hwcap & HWCAP_SVE)  ? 1 : 0;
    }
    {
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
        out->has_sve2 = (hwcap2 & HWCAP2_SVE2) ? 1 : 0;
    }
  #elif defined(__APPLE__)
    out->has_fp16 = 1;  /* Apple Silicon always has fp16 */
  #endif

    /* Read brand from /proc/cpuinfo on Linux */
  #if defined(__linux__)
    {
        FILE *f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "model name", 10) == 0 ||
                    strncmp(line, "Hardware", 8) == 0 ||
                    strncmp(line, "CPU implementer", 15) == 0) {
                    char *colon = strchr(line, ':');
                    if (colon) {
                        colon++;
                        while (*colon == ' ') colon++;
                        /* Remove trailing newline */
                        char *nl = strchr(colon, '\n');
                        if (nl) *nl = '\0';
                        safe_copy(out->brand, colon, sizeof(out->brand));
                        break;
                    }
                }
            }
            fclose(f);
        }
    }
  #endif

    /* Cache sizes from sysfs */
  #if defined(__linux__)
    {
        const char *base = "/sys/devices/system/cpu/cpu0/cache";
        for (int idx = 0; idx < 8; idx++) {
            char path_type[256], path_size[256], path_line[256], path_level[256];
            snprintf(path_type, sizeof(path_type), "%s/index%d/type", base, idx);
            snprintf(path_size, sizeof(path_size), "%s/index%d/size", base, idx);
            snprintf(path_line, sizeof(path_line), "%s/index%d/coherency_line_size", base, idx);
            snprintf(path_level, sizeof(path_level), "%s/index%d/level", base, idx);

            FILE *ft = fopen(path_type, "r");
            if (!ft) break;
            char type_str[32] = {0};
            if (fgets(type_str, sizeof(type_str), ft)) {
                char *nl = strchr(type_str, '\n');
                if (nl) *nl = '\0';
            }
            fclose(ft);

            FILE *fs = fopen(path_size, "r");
            size_t size_bytes = 0;
            if (fs) {
                char size_str[32] = {0};
                if (fgets(size_str, sizeof(size_str), fs)) {
                    size_bytes = (size_t)atol(size_str) * 1024;  /* Size is in KiB */
                }
                fclose(fs);
            }

            FILE *fl = fopen(path_level, "r");
            int level = 0;
            if (fl) {
                char level_str[8] = {0};
                if (fgets(level_str, sizeof(level_str), fl)) level = atoi(level_str);
                fclose(fl);
            }

            FILE *fc = fopen(path_line, "r");
            if (fc) {
                char cl_str[16] = {0};
                if (fgets(cl_str, sizeof(cl_str), fc)) out->cache_line_bytes = (size_t)atol(cl_str);
                fclose(fc);
            }

            if (level == 1 && strncmp(type_str, "Data", 4) == 0)        out->l1d_cache_bytes = size_bytes;
            else if (level == 1 && strncmp(type_str, "Instruction", 11) == 0) out->l1i_cache_bytes = size_bytes;
            else if (level == 2)  out->l2_cache_bytes = size_bytes;
            else if (level == 3)  out->l3_cache_bytes = size_bytes;
        }
    }
  #endif
#endif

    /* ── Core count (cross-platform) ─────────────────── */
#if defined(DTFLOWCV_WINDOWS)
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        out->logical_cores = (int)si.dwNumberOfProcessors;
        out->physical_cores = out->logical_cores;  /* Approximate */
    }
#elif defined(DTFLOWCV_POSIX)
    {
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        out->logical_cores = (nproc > 0) ? (int)nproc : 1;
        out->physical_cores = out->logical_cores;

        /* Try to get physical cores from /proc/cpuinfo on Linux */
      #if defined(__linux__)
        {
            FILE *f = fopen("/proc/cpuinfo", "r");
            if (f) {
                int max_core_id = -1;
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    if (strncmp(line, "core id", 7) == 0) {
                        char *colon = strchr(line, ':');
                        if (colon) {
                            int cid = atoi(colon + 1);
                            if (cid > max_core_id) max_core_id = cid;
                        }
                    }
                }
                fclose(f);
                if (max_core_id >= 0)
                    out->physical_cores = max_core_id + 1;
            }
        }
      #endif

      #if defined(__linux__)
        /* NUMA nodes */
        {
            FILE *f = fopen("/sys/devices/system/node/online", "r");
            if (f) {
                char buf[64] = {0};
                if (fgets(buf, sizeof(buf), f)) {
                    /* Format: "0-N" or "0" */
                    char *dash = strchr(buf, '-');
                    if (dash) out->numa_nodes = atoi(dash + 1) + 1;
                    else      out->numa_nodes = 1;
                }
                fclose(f);
            }
        }
      #endif
    }
#endif

    /* ── CPU frequency ───────────────────────────────── */
#if defined(__linux__)
    {
        FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
        if (f) {
            char buf[32] = {0};
            if (fgets(buf, sizeof(buf), f))
                out->base_freq_mhz = atoi(buf) / 1000;
            fclose(f);
        }
        if (out->base_freq_mhz == 0) {
            FILE *f2 = fopen("/proc/cpuinfo", "r");
            if (f2) {
                char line[256];
                while (fgets(line, sizeof(line), f2)) {
                    if (strncmp(line, "cpu MHz", 7) == 0) {
                        char *colon = strchr(line, ':');
                        if (colon) out->base_freq_mhz = (int)atof(colon + 1);
                        break;
                    }
                }
                fclose(f2);
            }
        }
    }
#endif
}

/* ── Memory Detection ───────────────────────────────────── */

void dtflowcv_detect_mem(dtflowcv_mem_info_t *out) {
    memset(out, 0, sizeof(*out));

#if defined(DTFLOWCV_WINDOWS)
    {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            out->total_ram_bytes     = (size_t)ms.ullTotalPhys;
            out->available_ram_bytes = (size_t)ms.ullAvailPhys;
        }
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        out->page_size_bytes = (size_t)si.dwPageSize;
    }
#elif defined(__linux__)
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0) {
            out->total_ram_bytes = (size_t)pages * (size_t)page_size;
            out->page_size_bytes = (size_t)page_size;
        }
        long avail = sysconf(_SC_AVPHYS_PAGES);
        if (avail > 0 && page_size > 0) {
            out->available_ram_bytes = (size_t)avail * (size_t)page_size;
        }
        /* Huge pages */
        FILE *f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "Hugepagesize:", 13) == 0) {
                    size_t kb = 0;
                    if (sscanf(line + 13, " %zu", &kb) == 1) {
                        out->huge_page_size_bytes = kb * 1024;
                        out->huge_pages_supported = 1;
                    }
                }
            }
            fclose(f);
        }
    }
#elif defined(__APPLE__)
    {
        long page_size = sysconf(_SC_PAGESIZE);
        out->page_size_bytes = (page_size > 0) ? (size_t)page_size : 4096;
        /* macOS: use sysctl for total memory */
        FILE *p = popen("sysctl -n hw.memsize 2>/dev/null", "r");
        if (p) {
            char buf[64] = {0};
            if (fgets(buf, sizeof(buf), p))
                out->total_ram_bytes = (size_t)atoll(buf);
            pclose(p);
        }
        out->available_ram_bytes = out->total_ram_bytes / 2;  /* Approximation */
    }
#endif
}

/* ── GPU Detection (basic) ──────────────────────────────── */

void dtflowcv_detect_gpu(dtflowcv_gpu_info_t *out) {
    memset(out, 0, sizeof(*out));

    /* Try nvidia-smi for CUDA info */
#if defined(DTFLOWCV_POSIX)
    {
        FILE *p = popen("nvidia-smi --query-gpu=name,memory.total,compute_cap "
                         "--format=csv,noheader,nounits 2>/dev/null", "r");
        if (p) {
            char buf[256] = {0};
            if (fgets(buf, sizeof(buf), p)) {
                /* Parse: "NAME, MEMORY_MB, COMPUTE_CAP" */
                char *name_end = strchr(buf, ',');
                if (name_end) {
                    *name_end = '\0';
                    safe_copy(out->cuda_device_name, buf, sizeof(out->cuda_device_name));
                    char *mem_str = name_end + 1;
                    while (*mem_str == ' ') mem_str++;
                    out->cuda_total_mem_bytes = (size_t)atol(mem_str) * 1024 * 1024;
                    char *cap_str = strchr(mem_str, ',');
                    if (cap_str) {
                        cap_str++;
                        while (*cap_str == ' ') cap_str++;
                        float cap = (float)atof(cap_str);
                        out->cuda_compute_major = (int)cap;
                        out->cuda_compute_minor = (int)((cap - (int)cap) * 10);
                    }
                    out->cuda_device_count = 1;
                }
            }
            int status = pclose(p);
            if (status != 0) out->cuda_device_count = 0;
        }
        /* Count devices */
        if (out->cuda_device_count > 0) {
            FILE *p2 = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l", "r");
            if (p2) {
                char buf2[16] = {0};
                if (fgets(buf2, sizeof(buf2), p2))
                    out->cuda_device_count = atoi(buf2);
                pclose(p2);
            }
        }
    }
#endif
}

/* ── OS Detection ───────────────────────────────────────── */

void dtflowcv_detect_os(dtflowcv_os_info_t *out) {
    memset(out, 0, sizeof(*out));

#if defined(DTFLOWCV_WINDOWS)
    safe_copy(out->os_name, "Windows", sizeof(out->os_name));
    {
        DWORD size = sizeof(out->hostname);
        GetComputerNameA(out->hostname, &size);
    }
    out->pid = (int)GetCurrentProcessId();
#elif defined(DTFLOWCV_POSIX)
    {
        struct utsname u;
        if (uname(&u) == 0) {
            safe_copy(out->os_name, u.sysname, sizeof(out->os_name));
            safe_copy(out->os_release, u.release, sizeof(out->os_release));
            safe_copy(out->hostname, u.nodename, sizeof(out->hostname));
        }
    }
    out->pid = (int)getpid();
#endif
}

/* ── Suitability Check ──────────────────────────────────── */

void dtflowcv_check_suitability(
    const dtflowcv_cpu_info_t *cpu,
    const dtflowcv_mem_info_t *mem,
    dtflowcv_suitability_t *out)
{
    memset(out, 0, sizeof(*out));
    out->overall_status = DTFLOWCV_HW_OK;
    out->cpu_ok = 1;
    out->ram_ok = 1;
    out->simd_ok = 1;

    /* Minimum: 2 cores */
    if (cpu->logical_cores < 2) {
        out->cpu_ok = 0;
        out->overall_status = DTFLOWCV_HW_INSUFFICIENT;
        append_msg(out, "CRITICAL: Less than 2 CPU cores detected. Pipeline requires >= 2 cores.");
    }

    /* Minimum: 2 GB RAM */
    size_t min_ram = (size_t)2 * 1024 * 1024 * 1024;
    if (mem->total_ram_bytes > 0 && mem->total_ram_bytes < min_ram) {
        out->ram_ok = 0;
        out->overall_status = DTFLOWCV_HW_INSUFFICIENT;
        append_msg(out, "CRITICAL: Less than 2 GB RAM detected. Pipeline requires >= 2 GB.");
    }

    /* Warning: < 8 GB for YOLO training */
    size_t rec_ram = (size_t)8 * 1024 * 1024 * 1024;
    if (mem->total_ram_bytes > 0 && mem->total_ram_bytes < rec_ram) {
        if (out->overall_status < DTFLOWCV_HW_WARN)
            out->overall_status = DTFLOWCV_HW_WARN;
        append_msg(out, "WARNING: Less than 8 GB RAM. YOLO training may run out of memory.");
    }

    /* SIMD: on x86 we need at least SSE2 */
    if (strcmp(cpu->arch, "x86_64") == 0 && !cpu->has_sse2) {
        out->simd_ok = 0;
        out->overall_status = DTFLOWCV_HW_INSUFFICIENT;
        append_msg(out, "CRITICAL: SSE2 not detected on x86_64. This should not happen.");
    }

    /* Recommendations */
    if (strcmp(cpu->arch, "x86_64") == 0) {
        if (!cpu->has_avx2) {
            if (out->overall_status < DTFLOWCV_HW_WARN)
                out->overall_status = DTFLOWCV_HW_WARN;
            append_msg(out, "INFO: AVX2 not available. SSE2 kernel will be used (slower).");
        }
        if (cpu->has_avx512f) {
            append_msg(out, "INFO: AVX-512 detected. Consider AVX-512 kernels for max throughput.");
        }
    }

    if (strcmp(cpu->arch, "aarch64") == 0) {
        append_msg(out, "INFO: ARM64 detected. NEON kernels are used automatically.");
        if (cpu->has_sve) {
            append_msg(out, "INFO: SVE detected. Consider SVE kernels for additional throughput.");
        }
    }

    if (cpu->logical_cores >= 8) {
        append_msg(out, "INFO: 8+ cores detected. Good for parallel data loading.");
    }
}

/* ── Hardware Recommendations ───────────────────────────── */

void dtflowcv_hw_recommendations(
    const dtflowcv_cpu_info_t *cpu,
    const dtflowcv_mem_info_t *mem,
    const dtflowcv_gpu_info_t *gpu,
    char *output,
    size_t output_size)
{
    if (!output || output_size < 128) return;
    output[0] = '\0';
    size_t pos = 0;

    #define APPEND(...) do { \
        int n = snprintf(output + pos, output_size - pos, __VA_ARGS__); \
        if (n > 0 && (size_t)n < output_size - pos) pos += (size_t)n; \
    } while(0)

    APPEND("=== dtflowcv Hardware Recommendations ===\n\n");

    /* Workers for data loading */
    int rec_workers = cpu->logical_cores / 2;
    if (rec_workers < 1) rec_workers = 1;
    if (rec_workers > 8) rec_workers = 8;
    APPEND("Data loader workers:      %d\n", rec_workers);

    /* Batch size based on RAM */
    int rec_batch = 16;
    if (mem->total_ram_bytes >= (size_t)32 * 1024 * 1024 * 1024)      rec_batch = 64;
    else if (mem->total_ram_bytes >= (size_t)16 * 1024 * 1024 * 1024)  rec_batch = 32;
    else if (mem->total_ram_bytes >= (size_t)8  * 1024 * 1024 * 1024)  rec_batch = 16;
    else                                                                rec_batch = 8;
    APPEND("Recommended batch size:   %d\n", rec_batch);

    /* Image size */
    int rec_imgsz = 640;
    if (mem->total_ram_bytes < (size_t)4 * 1024 * 1024 * 1024) rec_imgsz = 416;
    APPEND("Recommended image size:   %d\n", rec_imgsz);

    /* Preprocessing backend */
    if (strcmp(cpu->arch, "x86_64") == 0) {
        if (cpu->has_avx512f)
            APPEND("Preprocess kernel:        AVX-512 (optimal)\n");
        else if (cpu->has_avx2)
            APPEND("Preprocess kernel:        AVX2 (fast)\n");
        else
            APPEND("Preprocess kernel:        SSE2 (baseline)\n");
    } else if (strcmp(cpu->arch, "aarch64") == 0) {
        if (cpu->has_sve)
            APPEND("Preprocess kernel:        SVE (optimal)\n");
        else
            APPEND("Preprocess kernel:        NEON (baseline)\n");
    } else {
        APPEND("Preprocess kernel:        scalar (no SIMD)\n");
    }

    /* Training device */
    if (gpu->cuda_device_count > 0) {
        APPEND("Training device:          cuda:0 (%s, %zu MB)\n",
               gpu->cuda_device_name,
               gpu->cuda_total_mem_bytes / (1024 * 1024));
        /* GPU batch size */
        size_t gpu_gb = gpu->cuda_total_mem_bytes / (1024 * 1024 * 1024);
        int gpu_batch = 8;
        if (gpu_gb >= 24)      gpu_batch = 64;
        else if (gpu_gb >= 12) gpu_batch = 32;
        else if (gpu_gb >= 8)  gpu_batch = 16;
        else if (gpu_gb >= 4)  gpu_batch = 8;
        else                   gpu_batch = 4;
        APPEND("GPU training batch:       %d\n", gpu_batch);

        if (gpu->cuda_compute_major >= 7) {
            APPEND("FP16/AMP training:        yes (compute >= 7.0)\n");
        } else {
            APPEND("FP16/AMP training:        no (compute < 7.0)\n");
        }
    } else {
        APPEND("Training device:          cpu (no GPU detected)\n");
        APPEND("GPU training batch:       N/A\n");
        APPEND("FP16/AMP training:        no\n");
    }

    /* Cache optimization */
    if (cpu->l1d_cache_bytes > 0) {
        APPEND("L1D cache:                %zu KB — tile loops for < %zu KB working set\n",
               cpu->l1d_cache_bytes / 1024,
               cpu->l1d_cache_bytes / 1024);
    }
    if (cpu->l2_cache_bytes > 0) {
        APPEND("L2 cache:                 %zu KB — intermediate buffers < %zu KB ideal\n",
               cpu->l2_cache_bytes / 1024,
               cpu->l2_cache_bytes / 1024);
    }
    if (cpu->cache_line_bytes > 0) {
        APPEND("Cache line:               %zu bytes — align arrays to this boundary\n",
               cpu->cache_line_bytes);
    }

    /* NUMA */
    if (cpu->numa_nodes > 1) {
        APPEND("NUMA nodes:               %d — use numactl --interleave=all for training\n",
               cpu->numa_nodes);
    }

    /* Huge pages */
    if (mem->huge_pages_supported) {
        APPEND("Huge pages:               available (%zu KB) — enable for large datasets\n",
               mem->huge_page_size_bytes / 1024);
    }

    APPEND("\n");
    #undef APPEND
}

/* ── Print Full Report ──────────────────────────────────── */

void dtflowcv_print_hw_report(void) {
    dtflowcv_cpu_info_t cpu;
    dtflowcv_mem_info_t mem;
    dtflowcv_gpu_info_t gpu;
    dtflowcv_os_info_t  os;

    dtflowcv_detect_cpu(&cpu);
    dtflowcv_detect_mem(&mem);
    dtflowcv_detect_gpu(&gpu);
    dtflowcv_detect_os(&os);

    printf("================================================================\n");
    printf("  dtflowcv Hardware Report\n");
    printf("================================================================\n\n");

    printf("--- OS ---\n");
    printf("  System:         %s %s\n", os.os_name, os.os_release);
    printf("  Hostname:       %s\n", os.hostname);
    printf("  PID:            %d\n\n", os.pid);

    printf("--- CPU ---\n");
    printf("  Architecture:   %s\n", cpu.arch);
    printf("  Vendor:         %s\n", cpu.vendor);
    printf("  Brand:          %s\n", cpu.brand);
    printf("  Physical cores: %d\n", cpu.physical_cores);
    printf("  Logical cores:  %d\n", cpu.logical_cores);
    if (cpu.base_freq_mhz > 0)
        printf("  Max frequency:  %d MHz\n", cpu.base_freq_mhz);
    if (cpu.numa_nodes > 0)
        printf("  NUMA nodes:     %d\n", cpu.numa_nodes);
    printf("\n");

    if (strcmp(cpu.arch, "x86_64") == 0) {
        printf("--- x86-64 SIMD ---\n");
        printf("  SSE2:           %s\n", cpu.has_sse2    ? "yes" : "no");
        printf("  SSE3:           %s\n", cpu.has_sse3    ? "yes" : "no");
        printf("  SSSE3:          %s\n", cpu.has_ssse3   ? "yes" : "no");
        printf("  SSE4.1:         %s\n", cpu.has_sse41   ? "yes" : "no");
        printf("  SSE4.2:         %s\n", cpu.has_sse42   ? "yes" : "no");
        printf("  AVX:            %s\n", cpu.has_avx     ? "yes" : "no");
        printf("  AVX2:           %s\n", cpu.has_avx2    ? "yes" : "no");
        printf("  FMA3:           %s\n", cpu.has_fma3    ? "yes" : "no");
        printf("  AVX-512F:       %s\n", cpu.has_avx512f ? "yes" : "no");
        printf("  AVX-512BW:      %s\n", cpu.has_avx512bw? "yes" : "no");
        printf("  AVX-512VL:      %s\n", cpu.has_avx512vl? "yes" : "no");
        printf("\n");
    }

    if (strcmp(cpu.arch, "aarch64") == 0) {
        printf("--- AArch64 SIMD ---\n");
        printf("  NEON:           %s\n", cpu.has_neon ? "yes" : "no");
        printf("  FP16:           %s\n", cpu.has_fp16 ? "yes" : "no");
        printf("  SVE:            %s\n", cpu.has_sve  ? "yes" : "no");
        printf("  SVE2:           %s\n", cpu.has_sve2 ? "yes" : "no");
        printf("\n");
    }

    printf("--- Cache ---\n");
    if (cpu.l1d_cache_bytes)  printf("  L1 Data:        %zu KB\n", cpu.l1d_cache_bytes / 1024);
    if (cpu.l1i_cache_bytes)  printf("  L1 Instruction: %zu KB\n", cpu.l1i_cache_bytes / 1024);
    if (cpu.l2_cache_bytes)   printf("  L2:             %zu KB\n", cpu.l2_cache_bytes  / 1024);
    if (cpu.l3_cache_bytes)   printf("  L3:             %zu KB\n", cpu.l3_cache_bytes  / 1024);
    if (cpu.cache_line_bytes) printf("  Line size:      %zu bytes\n", cpu.cache_line_bytes);
    printf("\n");

    printf("--- Memory ---\n");
    if (mem.total_ram_bytes > 0)
        printf("  Total RAM:      %zu MB\n", mem.total_ram_bytes / (1024 * 1024));
    if (mem.available_ram_bytes > 0)
        printf("  Available RAM:  %zu MB\n", mem.available_ram_bytes / (1024 * 1024));
    if (mem.page_size_bytes > 0)
        printf("  Page size:      %zu bytes\n", mem.page_size_bytes);
    if (mem.huge_pages_supported)
        printf("  Huge pages:     %zu KB\n", mem.huge_page_size_bytes / 1024);
    printf("\n");

    printf("--- GPU ---\n");
    if (gpu.cuda_device_count > 0) {
        printf("  CUDA devices:   %d\n", gpu.cuda_device_count);
        printf("  Device 0:       %s\n", gpu.cuda_device_name);
        printf("  VRAM:           %zu MB\n", gpu.cuda_total_mem_bytes / (1024 * 1024));
        printf("  Compute:        %d.%d\n", gpu.cuda_compute_major, gpu.cuda_compute_minor);
    } else {
        printf("  CUDA devices:   none detected\n");
    }
    printf("\n");

    /* Suitability */
    dtflowcv_suitability_t suit;
    dtflowcv_check_suitability(&cpu, &mem, &suit);
    printf("--- Suitability ---\n");
    printf("  Status:         %s\n",
           suit.overall_status == DTFLOWCV_HW_OK ? "OK" :
           suit.overall_status == DTFLOWCV_HW_WARN ? "WARNING" : "INSUFFICIENT");
    for (int i = 0; i < suit.message_count; i++) {
        printf("  %s\n", suit.messages[i]);
    }
    printf("\n");

    /* Recommendations */
    char rec_buf[4096];
    dtflowcv_hw_recommendations(&cpu, &mem, &gpu, rec_buf, sizeof(rec_buf));
    printf("%s", rec_buf);
}
