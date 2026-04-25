#include "preprocess.h"

void dtflowcv_normalize_hwc_u8_to_chw_f32(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    size_t channels,
    const float *mean,
    const float *std) {
    const size_t plane = height * width;
    if (channels != 3) {
        return;
    }
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const size_t hwc = (y * width + x) * channels;
            const size_t pixel = y * width + x;
            output[pixel] = ((float)input[hwc] / 255.0f - mean[0]) / std[0];
            output[plane + pixel] = ((float)input[hwc + 1] / 255.0f - mean[1]) / std[1];
            output[2 * plane + pixel] = ((float)input[hwc + 2] / 255.0f - mean[2]) / std[2];
        }
    }
}

int dtflowcv_cpu_has_avx512f(void) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") ? 1 : 0;
#else
    return 0;
#endif
#else
    return 0;
#endif
}
