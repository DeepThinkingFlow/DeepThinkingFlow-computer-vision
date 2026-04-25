/**
 * HardwareProfiler.java — Cross-platform hardware profiler for dtflowcv
 *
 * Compile: javac HardwareProfiler.java
 * Run:     java HardwareProfiler [--json]
 *
 * Java 8+ compatible. Uses only standard library.
 * Runs on ANY platform with a JRE: Windows, Linux, macOS, ARM, x86, RISC-V.
 */

import java.io.*;
import java.lang.management.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;

public class HardwareProfiler {

    /* ── CPU Info ─────────────────────────────────────── */

    static String getArch() {
        return System.getProperty("os.arch", "unknown");
    }

    static int getAvailableProcessors() {
        return Runtime.getRuntime().availableProcessors();
    }

    static String getCpuBrand() {
        // Linux: first "model name" line in /proc/cpuinfo
        try {
            List<String> lines = Files.readAllLines(Paths.get("/proc/cpuinfo"));
            for (String line : lines) {
                if (line.startsWith("model name")) {
                    int idx = line.indexOf(':');
                    if (idx >= 0) return line.substring(idx + 1).trim();
                }
            }
        } catch (Exception ignored) {}

        // macOS
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"sysctl", "-n", "machdep.cpu.brand_string"});
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = r.readLine();
                if (line != null && !line.isEmpty()) return line.trim();
            }
        } catch (Exception ignored) {}

        // Windows: WMIC
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"wmic", "cpu", "get", "name"});
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty() && !line.equalsIgnoreCase("Name")) return line;
                }
            }
        } catch (Exception ignored) {}

        return "unknown";
    }

    static String getCpuVendor() {
        try {
            List<String> lines = Files.readAllLines(Paths.get("/proc/cpuinfo"));
            for (String line : lines) {
                if (line.startsWith("vendor_id")) {
                    int idx = line.indexOf(':');
                    if (idx >= 0) return line.substring(idx + 1).trim();
                }
            }
        } catch (Exception ignored) {}
        return "unknown";
    }

    /* ── SIMD feature flags (Linux only via /proc/cpuinfo) ── */

    static Map<String, Boolean> getSimdFlags() {
        Map<String, Boolean> flags = new LinkedHashMap<>();
        String[] simdNames = {
            "sse2", "sse3", "ssse3", "sse4_1", "sse4_2",
            "avx", "avx2", "fma",
            "avx512f", "avx512bw", "avx512vl",
            "neon", "asimd", "sve", "sve2"
        };
        for (String name : simdNames) flags.put(name, false);

        try {
            List<String> lines = Files.readAllLines(Paths.get("/proc/cpuinfo"));
            for (String line : lines) {
                if (line.startsWith("flags") || line.startsWith("Features")) {
                    String flagLine = line.substring(line.indexOf(':') + 1).trim();
                    Set<String> cpuFlags = new HashSet<>(Arrays.asList(flagLine.split("\\s+")));
                    for (String name : simdNames) {
                        if (cpuFlags.contains(name)) flags.put(name, true);
                    }
                    break;
                }
            }
        } catch (Exception ignored) {}

        // AArch64: asimd implies neon
        if (flags.getOrDefault("asimd", false)) flags.put("neon", true);

        return flags;
    }

    /* ── Cache info ──────────────────────────────────── */

    static Map<String, Long> getCacheInfo() {
        Map<String, Long> cache = new LinkedHashMap<>();
        // Linux sysfs
        for (int idx = 0; idx < 8; idx++) {
            String base = "/sys/devices/system/cpu/cpu0/cache/index" + idx;
            Path typePath = Paths.get(base, "type");
            Path sizePath = Paths.get(base, "size");
            Path levelPath = Paths.get(base, "level");
            Path linePath = Paths.get(base, "coherency_line_size");

            if (!Files.exists(typePath)) break;
            try {
                String type = new String(Files.readAllBytes(typePath)).trim();
                String sizeStr = new String(Files.readAllBytes(sizePath)).trim();
                int level = Integer.parseInt(new String(Files.readAllBytes(levelPath)).trim());

                // Size format: "32K", "256K", "8192K", etc.
                long sizeKb = Long.parseLong(sizeStr.replaceAll("[^0-9]", ""));

                if (level == 1 && type.startsWith("Data")) cache.put("l1d_kb", sizeKb);
                else if (level == 1 && type.startsWith("Instruction")) cache.put("l1i_kb", sizeKb);
                else if (level == 2) cache.put("l2_kb", sizeKb);
                else if (level == 3) cache.put("l3_kb", sizeKb);

                if (Files.exists(linePath)) {
                    long lineSize = Long.parseLong(new String(Files.readAllBytes(linePath)).trim());
                    cache.put("line_bytes", lineSize);
                }
            } catch (Exception ignored) {}
        }
        return cache;
    }

    /* ── Memory info ─────────────────────────────────── */

    static long getTotalMemoryMB() {
        // JMX
        try {
            OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
            if (os instanceof com.sun.management.OperatingSystemMXBean) {
                long bytes = ((com.sun.management.OperatingSystemMXBean) os).getTotalPhysicalMemorySize();
                return bytes / 1024 / 1024;
            }
        } catch (Exception ignored) {}

        // Linux fallback
        try {
            List<String> lines = Files.readAllLines(Paths.get("/proc/meminfo"));
            for (String line : lines) {
                if (line.startsWith("MemTotal:")) {
                    String val = line.replaceAll("[^0-9]", "");
                    return Long.parseLong(val) / 1024;
                }
            }
        } catch (Exception ignored) {}

        return Runtime.getRuntime().maxMemory() / 1024 / 1024;
    }

    static long getAvailableMemoryMB() {
        try {
            OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
            if (os instanceof com.sun.management.OperatingSystemMXBean) {
                long bytes = ((com.sun.management.OperatingSystemMXBean) os).getFreePhysicalMemorySize();
                return bytes / 1024 / 1024;
            }
        } catch (Exception ignored) {}

        try {
            List<String> lines = Files.readAllLines(Paths.get("/proc/meminfo"));
            for (String line : lines) {
                if (line.startsWith("MemAvailable:")) {
                    String val = line.replaceAll("[^0-9]", "");
                    return Long.parseLong(val) / 1024;
                }
            }
        } catch (Exception ignored) {}

        return Runtime.getRuntime().freeMemory() / 1024 / 1024;
    }

    /* ── GPU detection ───────────────────────────────── */

    static Map<String, String> getGpuInfo() {
        Map<String, String> gpu = new LinkedHashMap<>();
        gpu.put("count", "0");
        gpu.put("name", "none");
        gpu.put("vram_mb", "0");
        gpu.put("compute", "0.0");

        try {
            Process p = Runtime.getRuntime().exec(new String[]{
                "nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits"
            });
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = r.readLine();
                if (line != null) {
                    String[] parts = line.split(",");
                    if (parts.length >= 1) gpu.put("name", parts[0].trim());
                    if (parts.length >= 2) gpu.put("vram_mb", parts[1].trim());
                    if (parts.length >= 3) gpu.put("compute", parts[2].trim());
                    gpu.put("count", "1");
                }
            }
            int exitCode = p.waitFor();
            if (exitCode != 0) {
                gpu.put("count", "0");
                gpu.put("name", "none");
            }
        } catch (Exception ignored) {}

        // Count devices
        if (Integer.parseInt(gpu.get("count")) > 0) {
            try {
                Process p = Runtime.getRuntime().exec(new String[]{
                    "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
                });
                try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    int count = 0;
                    while (r.readLine() != null) count++;
                    gpu.put("count", String.valueOf(count));
                }
            } catch (Exception ignored) {}
        }

        return gpu;
    }

    /* ── NUMA info ───────────────────────────────────── */

    static int getNumaNodes() {
        try {
            String content = new String(Files.readAllBytes(Paths.get("/sys/devices/system/node/online")));
            content = content.trim();
            if (content.contains("-")) {
                String[] parts = content.split("-");
                return Integer.parseInt(parts[parts.length - 1]) + 1;
            }
            return 1;
        } catch (Exception ignored) { return 0; }
    }

    /* ── Suitability check ───────────────────────────── */

    static void checkSuitability(int cores, long ramMb, Map<String, Boolean> simd, String arch) {
        System.out.println("--- Suitability for dtflowcv ---\n");
        int status = 0;

        if (cores < 2) {
            System.out.println("  [FAIL] CPU cores: " + cores + ". Minimum: 2.");
            status = 2;
        } else if (cores < 4) {
            System.out.println("  [WARN] CPU cores: " + cores + ". Recommend >= 4.");
            if (status < 1) status = 1;
        } else {
            System.out.println("  [ OK ] CPU cores: " + cores);
        }

        if (ramMb < 2048) {
            System.out.println("  [FAIL] RAM: " + ramMb + " MB. Minimum: 2048 MB.");
            status = 2;
        } else if (ramMb < 8192) {
            System.out.println("  [WARN] RAM: " + ramMb + " MB. Recommend >= 8192 MB for training.");
            if (status < 1) status = 1;
        } else {
            System.out.println("  [ OK ] RAM: " + ramMb + " MB");
        }

        if (arch.contains("amd64") || arch.contains("x86_64")) {
            boolean sse2 = simd.getOrDefault("sse2", false);
            boolean avx2 = simd.getOrDefault("avx2", false);
            if (!sse2) {
                System.out.println("  [FAIL] SSE2 not detected.");
                status = 2;
            } else {
                System.out.println("  [ OK ] SSE2 available");
            }
            if (avx2) System.out.println("  [ OK ] AVX2 available");
            else System.out.println("  [WARN] No AVX2 — SSE2 kernel used");
        } else if (arch.contains("aarch64")) {
            System.out.println("  [ OK ] AArch64 — NEON available");
            if (simd.getOrDefault("sve", false)) System.out.println("  [ OK ] SVE available");
        }

        System.out.println("\n  Overall: " +
            (status == 0 ? "READY" : status == 1 ? "READY (with warnings)" : "NOT READY") + "\n");
    }

    /* ── Recommendations ─────────────────────────────── */

    static void printRecommendations(int cores, long ramMb, Map<String, String> gpu,
                                     Map<String, Boolean> simd, String arch) {
        System.out.println("--- Recommendations ---\n");

        int workers = Math.min(8, Math.max(1, cores / 2));
        System.out.println("  workers:        " + workers);

        int batch;
        if (ramMb >= 32768) batch = 64;
        else if (ramMb >= 16384) batch = 32;
        else if (ramMb >= 8192) batch = 16;
        else batch = 8;
        System.out.println("  batch_size:     " + batch);

        int imgsz = ramMb < 4096 ? 416 : 640;
        System.out.println("  image_size:     " + imgsz);

        if (arch.contains("amd64") || arch.contains("x86_64")) {
            if (simd.getOrDefault("avx512f", false)) System.out.println("  simd_kernel:    avx512");
            else if (simd.getOrDefault("avx2", false)) System.out.println("  simd_kernel:    avx2");
            else System.out.println("  simd_kernel:    sse2");
        } else if (arch.contains("aarch64")) {
            if (simd.getOrDefault("sve", false)) System.out.println("  simd_kernel:    sve");
            else System.out.println("  simd_kernel:    neon");
        } else {
            System.out.println("  simd_kernel:    scalar");
        }

        int gpuCount = Integer.parseInt(gpu.getOrDefault("count", "0"));
        if (gpuCount > 0) {
            System.out.println("  device:         cuda:0 (" + gpu.get("name") + ")");
            long vram = Long.parseLong(gpu.getOrDefault("vram_mb", "0"));
            int gb = (int)(vram / 1024);
            int gbatch;
            if (gb >= 24) gbatch = 64;
            else if (gb >= 12) gbatch = 32;
            else if (gb >= 8) gbatch = 16;
            else if (gb >= 4) gbatch = 8;
            else gbatch = 4;
            System.out.println("  gpu_batch:      " + gbatch);
            float compute = Float.parseFloat(gpu.getOrDefault("compute", "0.0"));
            System.out.println("  amp_training:   " + (compute >= 7.0 ? "yes" : "no"));
        } else {
            System.out.println("  device:         cpu");
        }
        System.out.println();
    }

    /* ── JSON output ─────────────────────────────────── */

    static void printJson(String arch, String brand, String vendor, int cores,
                          long totalMb, long availMb, Map<String, Boolean> simd,
                          Map<String, Long> cacheInfo, Map<String, String> gpu, int numa) {
        System.out.println("{");
        System.out.println("  \"arch\": \"" + arch + "\",");
        System.out.println("  \"vendor\": \"" + vendor + "\",");
        System.out.println("  \"brand\": \"" + jsonEscape(brand) + "\",");
        System.out.println("  \"cores\": " + cores + ",");
        System.out.println("  \"numa_nodes\": " + numa + ",");
        System.out.println("  \"java_version\": \"" + System.getProperty("java.version") + "\",");
        System.out.println("  \"os\": \"" + System.getProperty("os.name") + " " + System.getProperty("os.version") + "\",");
        System.out.print("  \"simd\": {");
        boolean first = true;
        for (Map.Entry<String, Boolean> e : simd.entrySet()) {
            if (!first) System.out.print(",");
            System.out.print(" \"" + e.getKey() + "\": " + e.getValue());
            first = false;
        }
        System.out.println(" },");
        System.out.print("  \"cache\": {");
        first = true;
        for (Map.Entry<String, Long> e : cacheInfo.entrySet()) {
            if (!first) System.out.print(",");
            System.out.print(" \"" + e.getKey() + "\": " + e.getValue());
            first = false;
        }
        System.out.println(" },");
        System.out.println("  \"memory\": { \"total_mb\": " + totalMb + ", \"available_mb\": " + availMb + " },");
        System.out.println("  \"gpu\": { \"count\": " + gpu.get("count") +
                ", \"name\": \"" + jsonEscape(gpu.get("name")) +
                "\", \"vram_mb\": " + gpu.get("vram_mb") +
                ", \"compute\": \"" + gpu.get("compute") + "\" }");
        System.out.println("}");
    }

    static String jsonEscape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /* ── Main ────────────────────────────────────────── */

    public static void main(String[] args) {
        boolean jsonMode = false;
        for (String arg : args) {
            if (arg.equals("--json")) jsonMode = true;
            if (arg.equals("--help") || arg.equals("-h")) {
                System.out.println("Usage: java HardwareProfiler [--json]");
                System.out.println("  Cross-platform hardware profiler for dtflowcv pipeline.");
                System.out.println("  --json    Output in JSON format");
                return;
            }
        }

        String arch = getArch();
        String brand = getCpuBrand();
        String vendor = getCpuVendor();
        int cores = getAvailableProcessors();
        long totalMb = getTotalMemoryMB();
        long availMb = getAvailableMemoryMB();
        Map<String, Boolean> simd = getSimdFlags();
        Map<String, Long> cacheInfo = getCacheInfo();
        Map<String, String> gpu = getGpuInfo();
        int numa = getNumaNodes();

        if (jsonMode) {
            printJson(arch, brand, vendor, cores, totalMb, availMb, simd, cacheInfo, gpu, numa);
            return;
        }

        System.out.println("================================================================");
        System.out.println("  dtflowcv Hardware Profiler (Java)");
        System.out.println("================================================================\n");

        System.out.println("--- System ---");
        System.out.println("  OS:             " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        System.out.println("  Arch:           " + arch);
        System.out.println("  Java:           " + System.getProperty("java.version") + " (" + System.getProperty("java.vendor") + ")");
        try { System.out.println("  Hostname:       " + InetAddress.getLocalHost().getHostName()); }
        catch (Exception ignored) {}
        System.out.println();

        System.out.println("--- CPU ---");
        System.out.println("  Vendor:         " + vendor);
        System.out.println("  Brand:          " + brand);
        System.out.println("  Cores:          " + cores);
        if (numa > 0) System.out.println("  NUMA nodes:     " + numa);
        System.out.println();

        System.out.println("--- SIMD Features ---");
        for (Map.Entry<String, Boolean> e : simd.entrySet()) {
            System.out.printf("  %-14s  %s%n", e.getKey() + ":", e.getValue() ? "YES" : "NO");
        }
        System.out.println();

        if (!cacheInfo.isEmpty()) {
            System.out.println("--- Cache ---");
            for (Map.Entry<String, Long> e : cacheInfo.entrySet()) {
                String unit = e.getKey().endsWith("bytes") ? " bytes" : " KB";
                System.out.println("  " + e.getKey() + ": " + e.getValue() + unit);
            }
            System.out.println();
        }

        System.out.println("--- Memory ---");
        System.out.println("  Total:          " + totalMb + " MB");
        System.out.println("  Available:      " + availMb + " MB");
        System.out.println("  JVM max:        " + (Runtime.getRuntime().maxMemory() / 1024 / 1024) + " MB");
        System.out.println();

        System.out.println("--- GPU ---");
        int gpuCount = Integer.parseInt(gpu.getOrDefault("count", "0"));
        if (gpuCount > 0) {
            System.out.println("  CUDA devices:   " + gpuCount);
            System.out.println("  Device 0:       " + gpu.get("name") + " (" + gpu.get("vram_mb") + " MB, compute " + gpu.get("compute") + ")");
        } else {
            System.out.println("  CUDA devices:   none");
        }
        System.out.println();

        checkSuitability(cores, totalMb, simd, arch);
        printRecommendations(cores, totalMb, gpu, simd, arch);
    }
}
