#pragma once

#include <stddef.h>
#include <stdint.h>

void dtflowcv_normalize_hwc_u8_to_chw_f32(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    size_t channels,
    const float *mean,
    const float *std);

int dtflowcv_cpu_has_avx512f(void);
