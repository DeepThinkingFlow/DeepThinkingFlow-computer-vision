/*
 * dtflowcv_hwguide — Standalone hardware guide for any machine
 *
 * Compile:  gcc -O2 -o dtflowcv_hwguide hwguide_standalone.c
 *           (no external dependencies)
 *
 * This is a single-file, self-contained hardware detection and guidance
 * tool that works on ANY x86-64 or AArch64 Linux/macOS system without
 * needing Python, Rust, or any other runtime installed.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
  #define IS_WINDOWS 1
  #include <windows.h>
  #include <intrin.h>
#else
  #define IS_POSIX 1
  #include <unistd.h>
  #include <sys/utsname.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #define IS_X86 1
  #if !defined(IS_WINDOWS)
    #include <cpuid.h>
  #endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
  #define IS_AARCH64 1
  #if defined(__linux__)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
  #endif
#endif

/* ── Data structures ─────────────────────────────────────── */

typedef struct {
    char arch[16];
    char vendor[64];
    char brand[128];
    int logical_cores;
    int physical_cores;
    int has_sse2, has_sse3, has_ssse3, has_sse41, has_sse42;
    int has_avx, has_avx2, has_fma3;
    int has_avx512f, has_avx512bw, has_avx512vl;
    int has_neon, has_sve, has_sve2, has_fp16;
    size_t l1d_kb, l2_kb, l3_kb;
    size_t cache_line;
    int freq_mhz;
    int numa_nodes;
} cpu_t;

typedef struct {
    size_t total_mb;
    size_t avail_mb;
    int huge_pages;
    size_t huge_kb;
} mem_t;

typedef struct {
    int count;
    char name[128];
    size_t vram_mb;
    int compute_major, compute_minor;
} gpu_t;

/* ── CPUID wrapper ───────────────────────────────────────── */

#if defined(IS_X86)
static void do_cpuid(uint32_t leaf, uint32_t sub,
                     uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d) {
#if defined(IS_WINDOWS)
    int r[4];
    __cpuidex(r, (int)leaf, (int)sub);
    *a=(uint32_t)r[0]; *b=(uint32_t)r[1]; *c=(uint32_t)r[2]; *d=(uint32_t)r[3];
#else
    __cpuid_count(leaf, sub, *a, *b, *c, *d);
#endif
}
#endif

/* ── Detection functions ─────────────────────────────────── */

static void detect_cpu(cpu_t *c) {
    memset(c, 0, sizeof(*c));

#if defined(__x86_64__) || defined(_M_X64)
    strcpy(c->arch, "x86_64");
#elif defined(IS_AARCH64)
    strcpy(c->arch, "aarch64");
#elif defined(__arm__)
    strcpy(c->arch, "arm32");
#elif defined(__riscv)
    strcpy(c->arch, "riscv");
#else
    strcpy(c->arch, "unknown");
#endif

#if defined(IS_X86)
    {
        uint32_t a,b,cx,d;
        do_cpuid(0,0,&a,&b,&cx,&d);
        uint32_t mx=a;
        char v[13]; memcpy(v,&b,4); memcpy(v+4,&d,4); memcpy(v+8,&cx,4); v[12]=0;
        strncpy(c->vendor,v,63);

        if(mx>=1){do_cpuid(1,0,&a,&b,&cx,&d);
            c->has_sse2=(d>>26)&1; c->has_sse3=cx&1; c->has_ssse3=(cx>>9)&1;
            c->has_sse41=(cx>>19)&1; c->has_sse42=(cx>>20)&1;
            c->has_avx=(cx>>28)&1; c->has_fma3=(cx>>12)&1;
        }
        if(mx>=7){do_cpuid(7,0,&a,&b,&cx,&d);
            c->has_avx2=(b>>5)&1; c->has_avx512f=(b>>16)&1;
            c->has_avx512bw=(b>>30)&1; c->has_avx512vl=(b>>31)&1;
        }
        do_cpuid(0x80000000,0,&a,&b,&cx,&d);
        if(a>=0x80000004){
            uint32_t br[12];
            do_cpuid(0x80000002,0,&br[0],&br[1],&br[2],&br[3]);
            do_cpuid(0x80000003,0,&br[4],&br[5],&br[6],&br[7]);
            do_cpuid(0x80000004,0,&br[8],&br[9],&br[10],&br[11]);
            char bs[49]; memcpy(bs,br,48); bs[48]=0;
            const char *s=bs; while(*s==' ')s++;
            strncpy(c->brand,s,127);
        }

        /* Cache: leaf 4 (Intel) */
        for(int i=0;i<16;i++){
            do_cpuid(4,(uint32_t)i,&a,&b,&cx,&d);
            if((a&0x1F)==0) break;
            int lv=(a>>5)&7; int tp=a&0x1F;
            uint32_t ls=(b&0xFFF)+1, pt=((b>>12)&0x3FF)+1, wa=((b>>22)&0x3FF)+1, st=cx+1;
            size_t sz=(size_t)ls*pt*wa*st/1024;
            c->cache_line=ls;
            if(lv==1&&tp==1) c->l1d_kb=sz;
            else if(lv==2) c->l2_kb=sz;
            else if(lv==3) c->l3_kb=sz;
        }

        /* AMD fallback: leaf 0x8000001D */
        if(c->l1d_kb==0){
            do_cpuid(0x80000000,0,&a,&b,&cx,&d);
            if(a>=0x8000001D){
                for(int i=0;i<16;i++){
                    do_cpuid(0x8000001D,(uint32_t)i,&a,&b,&cx,&d);
                    if((a&0x1F)==0) break;
                    int lv=(a>>5)&7; int tp=a&0x1F;
                    uint32_t ls=(b&0xFFF)+1, pt=((b>>12)&0x3FF)+1, wa=((b>>22)&0x3FF)+1, st=cx+1;
                    size_t sz=(size_t)ls*pt*wa*st/1024;
                    c->cache_line=ls;
                    if(lv==1&&tp==1) c->l1d_kb=sz;
                    else if(lv==2) c->l2_kb=sz;
                    else if(lv==3) c->l3_kb=sz;
                }
            }
        }
    }
#endif

#if defined(IS_AARCH64)
    strcpy(c->vendor, "ARM");
    c->has_neon = 1;
  #if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
    c->has_fp16 = (hw & HWCAP_FPHP) ? 1 : 0;
    c->has_sve = (hw & HWCAP_SVE) ? 1 : 0;
    unsigned long hw2 = getauxval(AT_HWCAP2);
    c->has_sve2 = (hw2 & HWCAP2_SVE2) ? 1 : 0;

    /* Brand from /proc/cpuinfo */
    FILE *f=fopen("/proc/cpuinfo","r");
    if(f){ char l[256]; while(fgets(l,sizeof(l),f)){
        if(strncmp(l,"model name",10)==0||strncmp(l,"Hardware",8)==0){
            char *p=strchr(l,':'); if(p){p++; while(*p==' ')p++;
            char *nl=strchr(p,'\n'); if(nl)*nl=0;
            strncpy(c->brand,p,127); break;
    }}} fclose(f); }

    /* Cache from sysfs */
    for(int i=0;i<8;i++){
        char pt[256],ps[256],pl[256],pc[256];
        snprintf(pt,256,"/sys/devices/system/cpu/cpu0/cache/index%d/type",i);
        snprintf(ps,256,"/sys/devices/system/cpu/cpu0/cache/index%d/size",i);
        snprintf(pl,256,"/sys/devices/system/cpu/cpu0/cache/index%d/level",i);
        snprintf(pc,256,"/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size",i);
        FILE *ft=fopen(pt,"r"); if(!ft) break;
        char ts[32]={0}; if(fgets(ts,32,ft)){char *n=strchr(ts,'\n');if(n)*n=0;} fclose(ft);
        FILE *fs=fopen(ps,"r"); size_t kb=0;
        if(fs){char ss[32]={0}; if(fgets(ss,32,fs)) kb=(size_t)atol(ss); fclose(fs);}
        FILE *fll=fopen(pl,"r"); int lv=0;
        if(fll){char ls[8]={0}; if(fgets(ls,8,fll)) lv=atoi(ls); fclose(fll);}
        FILE *fc=fopen(pc,"r");
        if(fc){char cs[16]={0}; if(fgets(cs,16,fc)) c->cache_line=(size_t)atol(cs); fclose(fc);}
        if(lv==1&&strncmp(ts,"Data",4)==0) c->l1d_kb=kb;
        else if(lv==2) c->l2_kb=kb;
        else if(lv==3) c->l3_kb=kb;
    }
  #elif defined(__APPLE__)
    c->has_fp16 = 1;
  #endif
#endif

    /* Core count */
#if defined(IS_WINDOWS)
    SYSTEM_INFO si; GetSystemInfo(&si);
    c->logical_cores=(int)si.dwNumberOfProcessors;
    c->physical_cores=c->logical_cores;
#elif defined(IS_POSIX)
    long np=sysconf(_SC_NPROCESSORS_ONLN);
    c->logical_cores=(np>0)?(int)np:1;
    c->physical_cores=c->logical_cores;
  #if defined(__linux__)
    {FILE *f=fopen("/proc/cpuinfo","r"); if(f){
        int mc=-1; char l[256];
        while(fgets(l,sizeof(l),f)){
            if(strncmp(l,"core id",7)==0){char *p=strchr(l,':');
            if(p){int ci=atoi(p+1); if(ci>mc)mc=ci;}}}
        fclose(f); if(mc>=0) c->physical_cores=mc+1;
    }}
    {FILE *f=fopen("/sys/devices/system/node/online","r"); if(f){
        char b[64]={0}; if(fgets(b,64,f)){char *d=strchr(b,'-');
        c->numa_nodes=d?atoi(d+1)+1:1;} fclose(f);
    }}
  #endif
#endif

    /* Frequency */
#if defined(__linux__)
    {FILE *f=fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq","r");
    if(f){char b[32]={0}; if(fgets(b,32,f)) c->freq_mhz=atoi(b)/1000; fclose(f);}
    if(c->freq_mhz==0){
        FILE *f2=fopen("/proc/cpuinfo","r"); if(f2){char l[256];
        while(fgets(l,sizeof(l),f2)){if(strncmp(l,"cpu MHz",7)==0){
        char *p=strchr(l,':'); if(p) c->freq_mhz=(int)atof(p+1); break;}}
        fclose(f2);}}}
#endif
}

static void detect_mem(mem_t *m) {
    memset(m, 0, sizeof(*m));
#if defined(IS_WINDOWS)
    MEMORYSTATUSEX ms; ms.dwLength=sizeof(ms);
    if(GlobalMemoryStatusEx(&ms)){
        m->total_mb=(size_t)(ms.ullTotalPhys/1024/1024);
        m->avail_mb=(size_t)(ms.ullAvailPhys/1024/1024);
    }
#elif defined(__linux__)
    {long p=sysconf(_SC_PHYS_PAGES),s=sysconf(_SC_PAGESIZE);
    if(p>0&&s>0) m->total_mb=(size_t)((size_t)p*(size_t)s/1024/1024);
    long a=sysconf(_SC_AVPHYS_PAGES);
    if(a>0&&s>0) m->avail_mb=(size_t)((size_t)a*(size_t)s/1024/1024);}
    {FILE *f=fopen("/proc/meminfo","r"); if(f){char l[256];
    while(fgets(l,sizeof(l),f)){if(strncmp(l,"Hugepagesize:",13)==0){
    size_t kb=0; if(sscanf(l+13," %zu",&kb)==1){m->huge_kb=kb; m->huge_pages=1;}
    }} fclose(f);}}
#elif defined(__APPLE__)
    {FILE *p=popen("sysctl -n hw.memsize 2>/dev/null","r"); if(p){
    char b[64]={0}; if(fgets(b,64,p)) m->total_mb=(size_t)(atoll(b)/1024/1024);
    pclose(p);} m->avail_mb=m->total_mb/2;}
#endif
}

static void detect_gpu(gpu_t *g) {
    memset(g, 0, sizeof(*g));
#if defined(IS_POSIX)
    FILE *p=popen("nvidia-smi --query-gpu=name,memory.total,compute_cap "
                  "--format=csv,noheader,nounits 2>/dev/null","r");
    if(p){char b[256]={0}; if(fgets(b,256,p)){
        char *ne=strchr(b,','); if(ne){*ne=0;
        strncpy(g->name,b,127); char *ms=ne+1; while(*ms==' ')ms++;
        g->vram_mb=(size_t)atol(ms);
        char *cs=strchr(ms,','); if(cs){cs++; while(*cs==' ')cs++;
        float cap=(float)atof(cs); g->compute_major=(int)cap;
        g->compute_minor=(int)((cap-(int)cap)*10);}
        g->count=1;
    }} int st=pclose(p); if(st!=0) g->count=0;}
    if(g->count>0){
        FILE *p2=popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l","r");
        if(p2){char b2[16]={0}; if(fgets(b2,16,p2)) g->count=atoi(b2); pclose(p2);}
    }
#endif
}

/* ── Suitability ─────────────────────────────────────────── */

static void check_suitability(const cpu_t *c, const mem_t *m) {
    printf("--- Suitability for dtflowcv Pipeline ---\n\n");

    int status = 0; /* 0=OK, 1=WARN, 2=FAIL */

    if (c->logical_cores < 2) {
        printf("  [FAIL] Less than 2 CPU cores. Minimum: 2 cores.\n");
        status = 2;
    } else if (c->logical_cores < 4) {
        printf("  [WARN] Only %d cores. Recommend >= 4 for parallel data loading.\n", c->logical_cores);
        if (status < 1) status = 1;
    } else {
        printf("  [ OK ] CPU cores: %d\n", c->logical_cores);
    }

    if (m->total_mb < 2048) {
        printf("  [FAIL] RAM: %zu MB. Minimum: 2048 MB.\n", m->total_mb);
        status = 2;
    } else if (m->total_mb < 8192) {
        printf("  [WARN] RAM: %zu MB. Recommend >= 8192 MB for YOLO training.\n", m->total_mb);
        if (status < 1) status = 1;
    } else {
        printf("  [ OK ] RAM: %zu MB\n", m->total_mb);
    }

    if (strcmp(c->arch, "x86_64") == 0) {
        if (!c->has_sse2) {
            printf("  [FAIL] SSE2 not available. Required for x86_64 pipeline.\n");
            status = 2;
        } else {
            printf("  [ OK ] SSE2 available\n");
        }
        if (c->has_avx2)
            printf("  [ OK ] AVX2 available — fast kernel path\n");
        else
            printf("  [WARN] No AVX2 — SSE2 kernel used (slower, but functional)\n");

        if (c->has_avx512f)
            printf("  [ OK ] AVX-512 available — maximum throughput kernel possible\n");
    } else if (strcmp(c->arch, "aarch64") == 0) {
        printf("  [ OK ] NEON available (mandatory on AArch64)\n");
        if (c->has_sve)  printf("  [ OK ] SVE available — high-throughput kernel possible\n");
        if (c->has_sve2) printf("  [ OK ] SVE2 available\n");
    }

    printf("\n  Overall: %s\n\n",
           status == 0 ? "READY" : status == 1 ? "READY (with warnings)" : "NOT READY");
}

/* ── Recommendations ─────────────────────────────────────── */

static void print_recommendations(const cpu_t *c, const mem_t *m, const gpu_t *g) {
    printf("--- Pipeline Configuration Recommendations ---\n\n");

    int workers = c->logical_cores / 2;
    if (workers < 1) workers = 1;
    if (workers > 8) workers = 8;
    printf("  workers:            %d\n", workers);

    int batch = 16;
    if (m->total_mb >= 32768)      batch = 64;
    else if (m->total_mb >= 16384) batch = 32;
    else if (m->total_mb >= 8192)  batch = 16;
    else                           batch = 8;
    printf("  batch_size:         %d\n", batch);

    int imgsz = (m->total_mb < 4096) ? 416 : 640;
    printf("  image_size:         %d\n", imgsz);

    if (strcmp(c->arch, "x86_64") == 0) {
        if (c->has_avx512f)     printf("  simd_kernel:        avx512 (optimal)\n");
        else if (c->has_avx2)   printf("  simd_kernel:        avx2 (fast)\n");
        else                    printf("  simd_kernel:        sse2 (baseline)\n");
    } else if (strcmp(c->arch, "aarch64") == 0) {
        if (c->has_sve)         printf("  simd_kernel:        sve (optimal)\n");
        else                    printf("  simd_kernel:        neon (baseline)\n");
    } else {
        printf("  simd_kernel:        scalar (no simd)\n");
    }

    if (g->count > 0) {
        printf("  device:             cuda:0 (%s, %zu MB)\n", g->name, g->vram_mb);
        int gb = (int)(g->vram_mb / 1024);
        int gbatch = 8;
        if (gb >= 24)      gbatch = 64;
        else if (gb >= 12) gbatch = 32;
        else if (gb >= 8)  gbatch = 16;
        else if (gb >= 4)  gbatch = 8;
        else               gbatch = 4;
        printf("  gpu_batch:          %d\n", gbatch);
        printf("  amp_training:       %s\n", g->compute_major >= 7 ? "yes" : "no");
    } else {
        printf("  device:             cpu\n");
        printf("  gpu_batch:          N/A\n");
        printf("  amp_training:       no\n");
    }

    if (c->l1d_kb > 0)
        printf("  l1d_cache:          %zu KB\n", c->l1d_kb);
    if (c->l2_kb > 0)
        printf("  l2_cache:           %zu KB\n", c->l2_kb);
    if (c->l3_kb > 0)
        printf("  l3_cache:           %zu KB\n", c->l3_kb);
    if (c->cache_line > 0)
        printf("  cache_line:         %zu bytes\n", c->cache_line);
    if (c->numa_nodes > 1)
        printf("  numa_nodes:         %d (use: numactl --interleave=all)\n", c->numa_nodes);
    if (m->huge_pages)
        printf("  huge_pages:         %zu KB available\n", m->huge_kb);
    printf("\n");
}

/* ── JSON output ─────────────────────────────────────────── */

static void print_json(const cpu_t *c, const mem_t *m, const gpu_t *g) {
    printf("{\n");
    printf("  \"arch\": \"%s\",\n", c->arch);
    printf("  \"vendor\": \"%s\",\n", c->vendor);
    printf("  \"brand\": \"%s\",\n", c->brand);
    printf("  \"physical_cores\": %d,\n", c->physical_cores);
    printf("  \"logical_cores\": %d,\n", c->logical_cores);
    if (c->freq_mhz > 0) printf("  \"max_freq_mhz\": %d,\n", c->freq_mhz);
    if (c->numa_nodes > 0) printf("  \"numa_nodes\": %d,\n", c->numa_nodes);
    printf("  \"simd\": {\n");
    if (strcmp(c->arch, "x86_64") == 0) {
        printf("    \"sse2\": %s, \"sse3\": %s, \"ssse3\": %s,\n",
               c->has_sse2?"true":"false", c->has_sse3?"true":"false", c->has_ssse3?"true":"false");
        printf("    \"sse41\": %s, \"sse42\": %s,\n",
               c->has_sse41?"true":"false", c->has_sse42?"true":"false");
        printf("    \"avx\": %s, \"avx2\": %s, \"fma3\": %s,\n",
               c->has_avx?"true":"false", c->has_avx2?"true":"false", c->has_fma3?"true":"false");
        printf("    \"avx512f\": %s, \"avx512bw\": %s, \"avx512vl\": %s\n",
               c->has_avx512f?"true":"false", c->has_avx512bw?"true":"false", c->has_avx512vl?"true":"false");
    } else if (strcmp(c->arch, "aarch64") == 0) {
        printf("    \"neon\": %s, \"fp16\": %s, \"sve\": %s, \"sve2\": %s\n",
               c->has_neon?"true":"false", c->has_fp16?"true":"false",
               c->has_sve?"true":"false", c->has_sve2?"true":"false");
    }
    printf("  },\n");
    printf("  \"cache\": { \"l1d_kb\": %zu, \"l2_kb\": %zu, \"l3_kb\": %zu, \"line_bytes\": %zu },\n",
           c->l1d_kb, c->l2_kb, c->l3_kb, c->cache_line);
    printf("  \"memory\": { \"total_mb\": %zu, \"available_mb\": %zu, \"huge_pages\": %s },\n",
           m->total_mb, m->avail_mb, m->huge_pages?"true":"false");
    printf("  \"gpu\": { \"count\": %d, \"name\": \"%s\", \"vram_mb\": %zu, \"compute\": \"%d.%d\" }\n",
           g->count, g->name, g->vram_mb, g->compute_major, g->compute_minor);
    printf("}\n");
}

/* ── Main ────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int json_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json_mode = 1;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--json]\n", argv[0]);
            printf("  Detects hardware capabilities and prints pipeline recommendations.\n");
            printf("  --json    Output in JSON format\n");
            return 0;
        }
    }

    cpu_t cpu; detect_cpu(&cpu);
    mem_t mem; detect_mem(&mem);
    gpu_t gpu; detect_gpu(&gpu);

    if (json_mode) {
        print_json(&cpu, &mem, &gpu);
        return 0;
    }

    printf("================================================================\n");
    printf("  dtflowcv Hardware Guide\n");
    printf("================================================================\n\n");

    printf("--- System ---\n");
#if defined(IS_POSIX)
    {struct utsname u; if(uname(&u)==0)
        printf("  OS:                 %s %s\n", u.sysname, u.release);}
#else
    printf("  OS:                 Windows\n");
#endif
    printf("  Architecture:       %s\n", cpu.arch);
    printf("  Vendor:             %s\n", cpu.vendor);
    printf("  Brand:              %s\n", cpu.brand);
    printf("  Physical cores:     %d\n", cpu.physical_cores);
    printf("  Logical cores:      %d\n", cpu.logical_cores);
    if (cpu.freq_mhz > 0)
        printf("  Max frequency:      %d MHz\n", cpu.freq_mhz);
    printf("\n");

    if (strcmp(cpu.arch, "x86_64") == 0) {
        printf("--- x86-64 SIMD Features ---\n");
        printf("  SSE2:    %-4s  SSE3:    %-4s  SSSE3:   %-4s\n",
               cpu.has_sse2?"YES":"NO", cpu.has_sse3?"YES":"NO", cpu.has_ssse3?"YES":"NO");
        printf("  SSE4.1:  %-4s  SSE4.2:  %-4s  AVX:     %-4s\n",
               cpu.has_sse41?"YES":"NO", cpu.has_sse42?"YES":"NO", cpu.has_avx?"YES":"NO");
        printf("  AVX2:    %-4s  FMA3:    %-4s\n", cpu.has_avx2?"YES":"NO", cpu.has_fma3?"YES":"NO");
        printf("  AVX-512F:%-4s  AVX-512BW:%-3s  AVX-512VL:%-3s\n",
               cpu.has_avx512f?"YES":"NO", cpu.has_avx512bw?"YES":"NO", cpu.has_avx512vl?"YES":"NO");
        printf("\n");
    }
    if (strcmp(cpu.arch, "aarch64") == 0) {
        printf("--- AArch64 SIMD Features ---\n");
        printf("  NEON: %-4s  FP16: %-4s  SVE: %-4s  SVE2: %-4s\n",
               cpu.has_neon?"YES":"NO", cpu.has_fp16?"YES":"NO",
               cpu.has_sve?"YES":"NO", cpu.has_sve2?"YES":"NO");
        printf("\n");
    }

    printf("--- Cache ---\n");
    if (cpu.l1d_kb)    printf("  L1 Data:            %zu KB\n", cpu.l1d_kb);
    if (cpu.l2_kb)     printf("  L2:                 %zu KB\n", cpu.l2_kb);
    if (cpu.l3_kb)     printf("  L3:                 %zu KB\n", cpu.l3_kb);
    if (cpu.cache_line) printf("  Line size:          %zu bytes\n", cpu.cache_line);
    printf("\n");

    printf("--- Memory ---\n");
    printf("  Total:              %zu MB\n", mem.total_mb);
    printf("  Available:          %zu MB\n", mem.avail_mb);
    if (mem.huge_pages) printf("  Huge pages:         %zu KB\n", mem.huge_kb);
    printf("\n");

    printf("--- GPU ---\n");
    if (gpu.count > 0) {
        printf("  CUDA devices:       %d\n", gpu.count);
        printf("  Device 0:           %s (%zu MB, compute %d.%d)\n",
               gpu.name, gpu.vram_mb, gpu.compute_major, gpu.compute_minor);
    } else {
        printf("  CUDA devices:       none\n");
    }
    printf("\n");

    check_suitability(&cpu, &mem);
    print_recommendations(&cpu, &mem, &gpu);

    return 0;
}
