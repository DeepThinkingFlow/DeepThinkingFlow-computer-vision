# dtflowcv_native

PyO3 extension for native preprocessing kernels.

Current source:

- Rust PyO3 module exposing `capabilities()` and `normalize_hwc_u8_to_chw_f32(...)`.
- C scalar HWC uint8 to CHW float32 normalization kernel.
- Runtime AVX-512F capability check.

Current non-claims:

- No AVX-512 speedup is claimed until an AVX-512 kernel is implemented, built, and benchmarked on target hardware.
- No NASM path is built in this environment because `nasm` is not installed.
- CUDA/Triton kernels are not present yet; Phase 5 profiling must prove the bottleneck before adding them.
