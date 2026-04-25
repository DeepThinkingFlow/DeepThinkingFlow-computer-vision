#include "preprocess.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>  /* SSE2 — baseline for all x86-64 */
#endif

/* =========================================================
 * Scalar fallback (unchanged ABI, 4× unrolled, FMA-style)
 * ========================================================= */
void dtflowcv_normalize_hwc_u8_to_chw_f32(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    size_t channels,
    const float *mean,
    const float *std)
{
    if (channels != 3) return;

    const size_t plane = height * width;

    /* Precompute: out = (pixel/255 - mean) / std
     *           = pixel * scale + bias
     * where scale = 1/(255*std), bias = -mean/std           */
    const float scale0 = 1.0f / (255.0f * std[0]);
    const float scale1 = 1.0f / (255.0f * std[1]);
    const float scale2 = 1.0f / (255.0f * std[2]);
    const float bias0  = -mean[0] / std[0];
    const float bias1  = -mean[1] / std[1];
    const float bias2  = -mean[2] / std[2];

    float *out0 = output;
    float *out1 = output + plane;
    float *out2 = output + 2 * plane;

    size_t total = height * width;
    size_t i = 0;

    /* 4× unrolled main loop */
    for (; i + 4 <= total; i += 4) {
        const uint8_t *p0 = input + (i    ) * 3;
        const uint8_t *p1 = input + (i + 1) * 3;
        const uint8_t *p2 = input + (i + 2) * 3;
        const uint8_t *p3 = input + (i + 3) * 3;

        out0[i    ] = (float)p0[0] * scale0 + bias0;
        out0[i + 1] = (float)p1[0] * scale0 + bias0;
        out0[i + 2] = (float)p2[0] * scale0 + bias0;
        out0[i + 3] = (float)p3[0] * scale0 + bias0;

        out1[i    ] = (float)p0[1] * scale1 + bias1;
        out1[i + 1] = (float)p1[1] * scale1 + bias1;
        out1[i + 2] = (float)p2[1] * scale1 + bias1;
        out1[i + 3] = (float)p3[1] * scale1 + bias1;

        out2[i    ] = (float)p0[2] * scale2 + bias2;
        out2[i + 1] = (float)p1[2] * scale2 + bias2;
        out2[i + 2] = (float)p2[2] * scale2 + bias2;
        out2[i + 3] = (float)p3[2] * scale2 + bias2;
    }
    /* Remainder */
    for (; i < total; ++i) {
        const uint8_t *p = input + i * 3;
        out0[i] = (float)p[0] * scale0 + bias0;
        out1[i] = (float)p[1] * scale1 + bias1;
        out2[i] = (float)p[2] * scale2 + bias2;
    }
}

/* =========================================================
 * SSE2 SIMD path — processes 4 pixels per iteration
 * x86-64 guarantees SSE2 availability.
 *
 * Register usage per iteration:
 *   xmm0-xmm2  : loaded R/G/B uint8 → int32 → float32
 *   xmm3       : scale broadcast
 *   xmm4       : bias broadcast
 *   xmm5       : scratch
 *
 * Layout: input is HWC interleaved [R0 G0 B0 R1 G1 B1 ...]
 *         output is planar          [R0 R1 R2 ... | G0 G1 ... | B0 B1 ...]
 * ========================================================= */
#if defined(__x86_64__) || defined(_M_X64)
void dtflowcv_normalize_hwc_u8_to_chw_f32_sse2(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    const float *mean,
    const float *std)
{
    const size_t plane = height * width;
    const size_t total = plane;

    const float scale0 = 1.0f / (255.0f * std[0]);
    const float scale1 = 1.0f / (255.0f * std[1]);
    const float scale2 = 1.0f / (255.0f * std[2]);
    const float bias0  = -mean[0] / std[0];
    const float bias1  = -mean[1] / std[1];
    const float bias2  = -mean[2] / std[2];

    /* Broadcast scale and bias into SSE registers */
    __m128 v_scale0 = _mm_set1_ps(scale0);
    __m128 v_scale1 = _mm_set1_ps(scale1);
    __m128 v_scale2 = _mm_set1_ps(scale2);
    __m128 v_bias0  = _mm_set1_ps(bias0);
    __m128 v_bias1  = _mm_set1_ps(bias1);
    __m128 v_bias2  = _mm_set1_ps(bias2);

    float *out0 = output;
    float *out1 = output + plane;
    float *out2 = output + 2 * plane;

    size_t i = 0;

    /* Process 4 pixels per SSE2 iteration.
     * Each pixel is 3 bytes (RGB), so we read 12 bytes = 4 pixels.
     * We de-interleave into R, G, B channels manually. */
    for (; i + 4 <= total; i += 4) {
        const uint8_t *p = input + i * 3;

        /* De-interleave: extract R, G, B for 4 pixels.
         * p[0]=R0, p[1]=G0, p[2]=B0, p[3]=R1, p[4]=G1, p[5]=B1, ...
         *
         * Use SSE2 integer operations to convert uint8 → float.
         * Step 1: zero-extend uint8 → int32
         * Step 2: cvt int32 → float32
         * Step 3: fma: result = pixel * scale + bias
         */

        /* Load 4 R values as int32, then convert to float */
        __m128i r_i32 = _mm_set_epi32(
            (int)p[9],   /* pixel 3: R */
            (int)p[6],   /* pixel 2: R */
            (int)p[3],   /* pixel 1: R */
            (int)p[0]    /* pixel 0: R */
        );
        __m128 r_f32 = _mm_cvtepi32_ps(r_i32);
        /* result_r = r_f32 * scale0 + bias0 */
        __m128 res_r = _mm_add_ps(_mm_mul_ps(r_f32, v_scale0), v_bias0);
        _mm_storeu_ps(out0 + i, res_r);

        /* Load 4 G values */
        __m128i g_i32 = _mm_set_epi32(
            (int)p[10],  /* pixel 3: G */
            (int)p[7],   /* pixel 2: G */
            (int)p[4],   /* pixel 1: G */
            (int)p[1]    /* pixel 0: G */
        );
        __m128 g_f32 = _mm_cvtepi32_ps(g_i32);
        __m128 res_g = _mm_add_ps(_mm_mul_ps(g_f32, v_scale1), v_bias1);
        _mm_storeu_ps(out1 + i, res_g);

        /* Load 4 B values */
        __m128i b_i32 = _mm_set_epi32(
            (int)p[11],  /* pixel 3: B */
            (int)p[8],   /* pixel 2: B */
            (int)p[5],   /* pixel 1: B */
            (int)p[2]    /* pixel 0: B */
        );
        __m128 b_f32 = _mm_cvtepi32_ps(b_i32);
        __m128 res_b = _mm_add_ps(_mm_mul_ps(b_f32, v_scale2), v_bias2);
        _mm_storeu_ps(out2 + i, res_b);
    }

    /* Scalar remainder for pixels not divisible by 4 */
    for (; i < total; ++i) {
        const uint8_t *p = input + i * 3;
        out0[i] = (float)p[0] * scale0 + bias0;
        out1[i] = (float)p[1] * scale1 + bias1;
        out2[i] = (float)p[2] * scale2 + bias2;
    }
}
#else
/* Non-x86: SSE2 path falls back to scalar */
void dtflowcv_normalize_hwc_u8_to_chw_f32_sse2(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    const float *mean,
    const float *std)
{
    dtflowcv_normalize_hwc_u8_to_chw_f32(input, output, height, width, 3, mean, std);
}
#endif

/* =========================================================
 * Runtime dispatcher
 * ========================================================= */
void dtflowcv_normalize_dispatch(
    const uint8_t *input,
    float *output,
    size_t height,
    size_t width,
    size_t channels,
    const float *mean,
    const float *std)
{
    if (channels != 3) return;
#if defined(__x86_64__) || defined(_M_X64)
    dtflowcv_normalize_hwc_u8_to_chw_f32_sse2(input, output, height, width, mean, std);
#else
    dtflowcv_normalize_hwc_u8_to_chw_f32(input, output, height, width, channels, mean, std);
#endif
}

/* =========================================================
 * Batch normalize: process N images sequentially
 * ========================================================= */
void dtflowcv_normalize_batch(
    const uint8_t *const *inputs,
    float **outputs,
    size_t count,
    size_t height,
    size_t width,
    const float *mean,
    const float *std)
{
    for (size_t n = 0; n < count; ++n) {
        dtflowcv_normalize_dispatch(inputs[n], outputs[n], height, width, 3, mean, std);
    }
}

/* =========================================================
 * Box IoU matrix: compute IoU for all pairs (a_i, b_j)
 * boxes layout: [x1, y1, x2, y2] per box, contiguous
 * out layout: row-major n_a × n_b
 * ========================================================= */
void dtflowcv_box_iou_matrix(
    const float *boxes_a,
    size_t n_a,
    const float *boxes_b,
    size_t n_b,
    float *out)
{
    for (size_t i = 0; i < n_a; ++i) {
        float ax1 = boxes_a[i * 4 + 0];
        float ay1 = boxes_a[i * 4 + 1];
        float ax2 = boxes_a[i * 4 + 2];
        float ay2 = boxes_a[i * 4 + 3];
        float area_a = (ax2 - ax1) * (ay2 - ay1);
        if (area_a < 0.0f) area_a = 0.0f;

        for (size_t j = 0; j < n_b; ++j) {
            float bx1 = boxes_b[j * 4 + 0];
            float by1 = boxes_b[j * 4 + 1];
            float bx2 = boxes_b[j * 4 + 2];
            float by2 = boxes_b[j * 4 + 3];

            float ix1 = ax1 > bx1 ? ax1 : bx1;
            float iy1 = ay1 > by1 ? ay1 : by1;
            float ix2 = ax2 < bx2 ? ax2 : bx2;
            float iy2 = ay2 < by2 ? ay2 : by2;

            float iw = ix2 - ix1;
            float ih = iy2 - iy1;
            if (iw < 0.0f) iw = 0.0f;
            if (ih < 0.0f) ih = 0.0f;

            float inter = iw * ih;
            float area_b = (bx2 - bx1) * (by2 - by1);
            if (area_b < 0.0f) area_b = 0.0f;
            float uni = area_a + area_b - inter;

            out[i * n_b + j] = (uni > 0.0f) ? (inter / uni) : 0.0f;
        }
    }
}

/* =========================================================
 * CPU feature detection
 * ========================================================= */
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

int dtflowcv_cpu_has_sse2(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return 1;  /* SSE2 is mandatory on all x86-64 CPUs */
#else
    return 0;
#endif
}
