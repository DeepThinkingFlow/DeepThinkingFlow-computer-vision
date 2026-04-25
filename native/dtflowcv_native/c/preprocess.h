#pragma once

#include <stddef.h>
#include <stdint.h>

/* Scalar fallback: always available. */
void dtflowcv_normalize_hwc_u8_to_chw_f32(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    size_t channels,
    const float *mean,
    const float *std);

/* SSE2 path: available on all x86-64. */
void dtflowcv_normalize_hwc_u8_to_chw_f32_sse2(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    const float *mean,
    const float *std);

/* Runtime dispatcher: picks fastest available path. */
void dtflowcv_normalize_dispatch(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    size_t channels,
    const float *mean,
    const float *std);

/* Batch: normalize N images. */
void dtflowcv_normalize_batch(
    const uint8_t *const *inputs,
    float **outputs,
    size_t count,
    size_t height,
    size_t width,
    const float *mean,
    const float *std);

/* Box IoU for N×M pairs. out must have n_a * n_b elements. */
void dtflowcv_box_iou_matrix(
    const float *boxes_a,
    size_t n_a,
    const float *boxes_b,
    size_t n_b,
    float *out);

int dtflowcv_cpu_has_avx512f(void);
int dtflowcv_cpu_has_sse2(void);
