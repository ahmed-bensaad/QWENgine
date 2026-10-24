/// @file quant_ref.cpp
/// Scalar reference dequantization / quantization, ported 1:1 from
/// ggml/src/ggml-quants.c (MIT License, The ggml authors). These are the ground
/// truth for every optimized CPU/GPU kernel in this project and are unit-tested
/// bit-exact against libggml (tests/unit/test_quant_ref.cpp).

#include <cmath>
#include <cstring>
#include <algorithm>
#include "quant/quant_formats.h"
#include "quant/iq_tables.h"

namespace q38rt {

namespace {

inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void dq_q2_K(const block_q2_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

void dq_q4_K(const block_q4_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dq_q6_K(const block_q6_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dq_iq2_xxs(const block_iq2_xxs* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

void dq_iq2_xs(const block_iq2_xs* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    float db[2];
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (x[i].qs[4 * ib32 + l] & 511));
                const uint8_t signs = ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

void dq_iq2_s(const block_iq2_s* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    float db[2];
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

void dq_iq3_xxs(const block_iq3_xxs* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    uint32_t aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

void dq_iq3_s(const block_iq3_s* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

void dq_iq1_s(const block_iq1_s* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint16_t* qh = x[i].qh;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) y[j] = dl * (grid[j] + delta);
                y += 8;
            }
            qs += 4;
        }
    }
}

void dq_iq1_m(const block_iq1_m* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    float delta[4];
    uint16_t idx[4];
    for (int64_t i = 0; i < nb; i++) {
        uint16_t sc[4];
        memcpy(sc, x[i].scales, 8);
        const uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = fp16_to_fp32(scale_u16);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) y[j] = dl1 * (grid[j] + delta[l]);
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) y[j] = dl2 * (grid[j] + delta[l]);
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

void dq_iq4_xs(const block_iq4_xs* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

void dq_q8_0(const block_q8_0* x, float* y, int64_t k) {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = x[i].qs[j] * d;
    }
}

void dq_q4_0(const block_q4_0* x, float* y, int64_t k) {
    const int64_t nb = k / QK4_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            y[i * QK4_0 + j] = ((x[i].qs[j] & 0x0F) - 8) * d;
            y[i * QK4_0 + j + QK4_0 / 2] = ((x[i].qs[j] >> 4) - 8) * d;
        }
    }
}

} // namespace

bool dequantize_row_ref(QuantFormat t, const void* src, float* dst, int64_t n) {
    switch (t) {
        case QuantFormat::FP32: memcpy(dst, src, n * sizeof(float)); return true;
        case QuantFormat::FP16: {
            const fp16_t* s = (const fp16_t*)src;
            for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_fp32(s[i]);
            return true;
        }
        case QuantFormat::BF16: {
            const bf16_t* s = (const bf16_t*)src;
            for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_fp32(s[i]);
            return true;
        }
        case QuantFormat::Q8_0: dq_q8_0((const block_q8_0*)src, dst, n); return true;
        case QuantFormat::Q4_0: dq_q4_0((const block_q4_0*)src, dst, n); return true;
        case QuantFormat::Q2_K: dq_q2_K((const block_q2_K*)src, dst, n); return true;
        case QuantFormat::Q4_K: dq_q4_K((const block_q4_K*)src, dst, n); return true;
        case QuantFormat::Q6_K: dq_q6_K((const block_q6_K*)src, dst, n); return true;
        case QuantFormat::IQ2_XXS: dq_iq2_xxs((const block_iq2_xxs*)src, dst, n); return true;
        case QuantFormat::IQ2_XS: dq_iq2_xs((const block_iq2_xs*)src, dst, n); return true;
        case QuantFormat::IQ2_S: dq_iq2_s((const block_iq2_s*)src, dst, n); return true;
        case QuantFormat::IQ3_XXS: dq_iq3_xxs((const block_iq3_xxs*)src, dst, n); return true;
        case QuantFormat::IQ3_S: dq_iq3_s((const block_iq3_s*)src, dst, n); return true;
        case QuantFormat::IQ1_S: dq_iq1_s((const block_iq1_s*)src, dst, n); return true;
        case QuantFormat::IQ1_M: dq_iq1_m((const block_iq1_m*)src, dst, n); return true;
        case QuantFormat::IQ4_XS: dq_iq4_xs((const block_iq4_xs*)src, dst, n); return true;
        default: return false;
    }
}

void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        float max = 0, amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            const float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            const int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = (int8_t)std::min(127, v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1 / iscale;
        x += QK_K;
    }
}

// ggml make_qkx2_quants (MIT, the ggml authors): weighted min/scale search for asymmetric quants
static float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, uint8_t* L, float* the_min, uint8_t* Laux,
                              float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        const int l = nearest_int(iscale * (x[i] - min));
        L[i] = (uint8_t)std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = (uint8_t)l;
            const float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

void quantize_row_q4_K_ref(const float* x, block_q4_K* y, int64_t k) {
    const int64_t nb = k / QK_K;
    uint8_t L[QK_K], Laux[32];
    float weights[32], mins[QK_K / 32], scales[QK_K / 32];
    for (int64_t i = 0; i < nb; i++) {
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
            const float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        memset(y[i].scales, 0, sizeof y[i].scales);
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = (uint8_t)nearest_int(inv_scale * scales[j]);
            uint8_t lm = (uint8_t)nearest_int(inv_min * mins[j]);
            ls = std::min<uint8_t>(63, ls);
            lm = std::min<uint8_t>(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
                y[i].scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
                y[i].scales[j - 0] |= (uint8_t)((lm >> 4) << 6);
            }
        }
        y[i].d = fp32_to_fp16(max_scale / 63.f);
        y[i].dmin = fp32_to_fp16(max_min / 63.f);
        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            const float dm = fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                L[32 * j + ii] = (uint8_t)std::max(0, std::min(15, l));
            }
        }
        uint8_t* q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            q += 32;
        }
        x += QK_K;
    }
}

void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int64_t k) {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) amax = std::max(amax, fabsf(x[i * QK8_0 + j]));
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) y[i].qs[j] = (int8_t)roundf(x[i * QK8_0 + j] * id);
    }
}

void quantize_row_q4_0_ref(const float* x, block_q4_0* y, int64_t k) {
    const int64_t nb = k / QK4_0;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            const float v = x[i * QK4_0 + j];
            if (amax < fabsf(v)) { amax = fabsf(v); max = v; }
        }
        const float d = max / -8;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const float x0 = x[i * QK4_0 + j] * id;
            const float x1 = x[i * QK4_0 + QK4_0 / 2 + j] * id;
            const uint8_t xi0 = (uint8_t)std::min<int>(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = (uint8_t)std::min<int>(15, (int8_t)(x1 + 8.5f));
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

} // namespace q38rt
