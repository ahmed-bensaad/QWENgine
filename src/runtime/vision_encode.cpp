/// @file vision_encode.cpp
/// Engine::encode_image: the GPU forward of the vision encoder (runtime/vision.h).
///
/// Weights: each layer's four matrices (30.5 MB FP16) are copied from pinned host memory into one half of the decode
/// streaming ring (2 x 32 MB, idle between decode passes) while the GPU computes the previous layer from the other
/// half; the projector matrices go through in row pieces. Activations (FP32 residual stream, FP16 GEMM inputs) live in
/// scratch VRAM; the attention runs per head and per chunk of query rows (scores, softmax, P V as cuBLAS GEMMs), so
/// memory grows linearly with the image (about 110 MB for a 1024-token image).

#include <algorithm>
#include <cmath>
#include "cuda/prefill.cuh"
#include "cuda/vision.cuh"
#include "runtime/engine.h"
#include "runtime/vision.h"

namespace q38rt {

bool Engine::encode_image(const VisionModel& vm, const ImagePatches& ip, std::vector<float>& out, std::string& err) {
    using V = VisionModel;
    const int N = ip.n_patches(), T = ip.n_tokens(), D = V::D, FF = V::FF, KP = 3 * V::PATCH * V::PATCH;
    if (N <= 0 || (size_t)ip.patch.size() != (size_t)N * KP) { err = "empty image"; return false; }
    const size_t slot_bytes = (size_t)NSLOT * SLOT_BYTES / 2;
    if (!d_slots_ || V::LAYER_ELEMS * 2 > slot_bytes) { err = "vision encoder needs the decode streaming ring"; return false; }
    if (spec_open_) { err = "encode_image during a speculative pass"; return false; }
    const int C = std::min(N, 1024);   // token chunk: QKV projection, attention queries, FFN

    // ---- scratch: buffers live in different phases share space ----
    const size_t small_n = V::LAYER_SMALL * V::LAYERS + D + 2 * D + V::MERGED + V::OUT;
    const size_t r_attn = (size_t)C * 3 * D * 4 + (size_t)C * N * 4 + (size_t)C * N * 2;   // QKV32 | S32 | P16
    const size_t r_ffn = (size_t)C * FF * 4 + (size_t)C * FF * 2;                          // U32 | U16
    const size_t r_misc = std::max({r_attn, r_ffn, (size_t)T * V::OUT * 4, (size_t)N * KP * 2});   // + OUT32, IN16
    enum { B_SMALL, B_PW, B_X, B_XN, B_QK, B_VT, B_R, B_O32, B_O16, B_WS, NB };
    std::vector<size_t> sizes(NB);
    sizes[B_SMALL] = small_n * 4;
    sizes[B_PW] = (size_t)D * KP * 2;
    sizes[B_X] = (size_t)N * D * 4;          // residual stream; merger: H32 [T][4608] lives in B_QK
    sizes[B_XN] = (size_t)N * D * 2;         // LayerNorm output; merger input M16 [T][4608]
    sizes[B_QK] = (size_t)2 * N * D * 2;     // Q16, K16 [16][N][72]
    sizes[B_VT] = (size_t)N * D * 2;         // Vt16 [16][72][N]; merger: H16
    sizes[B_R] = r_misc;
    sizes[B_O32] = (size_t)C * D * 4;
    sizes[B_O16] = (size_t)C * D * 2;
    sizes[B_WS] = 4u << 20;                  // cuBLAS workspace
    std::vector<void*> b;
    CHECK_CUDA(cudaStreamSynchronize(cst_));   // decode-ring prefetches land in d_slots_
    if (!scratch_acquire(sizes, b)) {
        err = "not enough free VRAM for an image of " + std::to_string(T) + " tokens (lower --image-max-tokens)";
        return false;
    }
    float* small = (float*)b[B_SMALL];
    __half* pw = (__half*)b[B_PW];
    float* X = (float*)b[B_X];
    __half* XN = (__half*)b[B_XN];
    __half* Q = (__half*)b[B_QK];
    __half* K = Q + (size_t)N * D;
    __half* Vt = (__half*)b[B_VT];
    float* QKV = (float*)b[B_R];
    float* S = QKV + (size_t)C * 3 * D;
    __half* P = (__half*)(S + (size_t)C * N);
    float* U32 = (float*)b[B_R];
    __half* U16 = (__half*)(U32 + (size_t)C * FF);
    float* O32 = (float*)b[B_O32];
    __half* O16 = (__half*)b[B_O16];
    if (!cuda::cublas_bind(cublas_, st_, b[B_WS], sizes[B_WS])) { scratch_release(); err = "cuBLAS setup failed"; return false; }

    // small parameters: per-layer norms and biases | patch bias | post norm | projector biases
    float* sm_patch_b = small + V::LAYER_SMALL * V::LAYERS;
    float* sm_post = sm_patch_b + D;
    float* sm_mm = sm_post + 2 * D;
    CHECK_CUDA(cudaMemcpyAsync(small, vm.small().data(), V::LAYER_SMALL * V::LAYERS * 4, cudaMemcpyHostToDevice, st_));
    CHECK_CUDA(cudaMemcpyAsync(sm_patch_b, vm.patch_bias().data(), D * 4, cudaMemcpyHostToDevice, st_));
    CHECK_CUDA(cudaMemcpyAsync(sm_post, vm.post_ln().data(), 2 * D * 4, cudaMemcpyHostToDevice, st_));
    CHECK_CUDA(cudaMemcpyAsync(sm_mm, vm.mm_bias().data(), (V::MERGED + V::OUT) * 4, cudaMemcpyHostToDevice, st_));
    CHECK_CUDA(cudaMemcpyAsync(pw, vm.patch_weight(), sizes[B_PW], cudaMemcpyHostToDevice, st_));

    // ---- weight stream: jobs in consumption order through the two halves of the ring ----
    struct Job { const uint16_t* src; size_t elems; };
    std::vector<Job> jobs;
    for (int il = 0; il < V::LAYERS; il++) jobs.push_back({vm.layer_weights(il), V::LAYER_ELEMS});
    const int piece_rows = (int)(slot_bytes / ((size_t)V::MERGED * 2));
    struct Piece { int r0, n; };
    std::vector<Piece> mm0_pieces, mm2_pieces;
    for (int r0 = 0; r0 < V::MERGED; r0 += piece_rows) {
        mm0_pieces.push_back({r0, std::min(piece_rows, V::MERGED - r0)});
        jobs.push_back({vm.mm0() + (size_t)r0 * V::MERGED, (size_t)mm0_pieces.back().n * V::MERGED});
    }
    for (int r0 = 0; r0 < V::OUT; r0 += piece_rows) {
        mm2_pieces.push_back({r0, std::min(piece_rows, V::OUT - r0)});
        jobs.push_back({vm.mm2() + (size_t)r0 * V::MERGED, (size_t)mm2_pieces.back().n * V::MERGED});
    }
    cudaEvent_t ready[2], freed[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&freed[i], cudaEventDisableTiming));
    }
    bool used[2] = {false, false};
    auto issue = [&](size_t j) {
        if (j >= jobs.size()) return;
        const int s = (int)(j & 1);
        if (used[s]) CHECK_CUDA(cudaStreamWaitEvent(cst_, freed[s], 0));
        CHECK_CUDA(cudaMemcpyAsync(d_slots_ + s * slot_bytes, jobs[j].src, jobs[j].elems * 2, cudaMemcpyHostToDevice, cst_));
        CHECK_CUDA(cudaEventRecord(ready[s], cst_));
    };
    size_t job = 0;
    issue(0);
    issue(1);
    auto take = [&]() -> const __half* {   // weights of the next job (stream-ordered after its copy)
        const int s = (int)(job & 1);
        CHECK_CUDA(cudaStreamWaitEvent(st_, ready[s], 0));
        return (const __half*)(d_slots_ + s * slot_bytes);
    };
    auto done = [&]() {   // the GPU is done with the current job's slot: refill it with the job two ahead
        const int s = (int)(job & 1);
        CHECK_CUDA(cudaEventRecord(freed[s], st_));
        used[s] = true;
        issue(job + 2);
        job++;
    };

    // ---- patch embedding + bias + resized position embeddings ----
    {
        std::vector<__half> in16((size_t)N * KP);
        for (size_t i = 0; i < in16.size(); i++) in16[i] = __float2half_rn(ip.patch[i]);
        const std::vector<float> pos = vm.position_embeddings(ip.nx, ip.ny);
        __half* IN16 = (__half*)b[B_R];
        CHECK_CUDA(cudaMemcpyAsync(IN16, in16.data(), in16.size() * 2, cudaMemcpyHostToDevice, st_));
        CHECK_CUDA(cudaMemcpyAsync(X, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, st_));
        cuda::gemm_f16(cublas_, pw, D, KP, IN16, KP, N, X, D, true, st_);   // X = pos + patches W^T
        cuda::vis_bias(X, sm_patch_b, N, D, st_);
        CHECK_CUDA(cudaStreamSynchronize(st_));   // host buffers in16 / pos go out of scope
    }

    // ---- 27 blocks ----
    const float kq_scale = 1.0f / std::sqrt((float)V::HEAD_DIM), eps = vm.eps();
    for (int il = 0; il < V::LAYERS; il++) {
        const __half* W = take();
        const __half *Wqkv = W, *Wout = W + V::QKV_ELEMS, *Wup = Wout + V::OUT_ELEMS, *Wdown = Wup + V::UP_ELEMS;
        const float* sp = small + (size_t)il * V::LAYER_SMALL;
        const float *ln1w = sp, *ln1b = sp + D, *qkvb = sp + 2 * D, *outb = qkvb + 3 * D, *ln2w = outb + D,
                    *ln2b = ln2w + D, *upb = ln2b + D, *downb = upb + FF;
        // attention: Q, K (rotated), V for all patches, then per chunk of query rows and head
        cuda::vis_layernorm(X, ln1w, ln1b, N, D, eps, XN, st_);
        for (int t0 = 0; t0 < N; t0 += C) {
            const int c = std::min(C, N - t0);
            cuda::gemm_f16(cublas_, Wqkv, 3 * D, D, XN + (size_t)t0 * D, D, c, QKV, 3 * D, false, st_);
            cuda::vis_qkv_rope(QKV, qkvb, t0, c, N, ip.nx, Q, K, Vt, st_);
        }
        for (int t0 = 0; t0 < N; t0 += C) {
            const int c = std::min(C, N - t0);
            for (int h = 0; h < V::HEADS; h++) {
                const __half *Qh = Q + (size_t)h * N * V::HEAD_DIM, *Kh = K + (size_t)h * N * V::HEAD_DIM,
                             *Vh = Vt + (size_t)h * V::HEAD_DIM * N;
                cuda::gemm_f16(cublas_, Kh, N, V::HEAD_DIM, Qh + (size_t)t0 * V::HEAD_DIM, V::HEAD_DIM, c, S, N, false, st_);
                cuda::vis_softmax(S, c, N, kq_scale, P, st_);
                cuda::gemm_f16(cublas_, Vh, V::HEAD_DIM, N, P, N, c, O32 + h * V::HEAD_DIM, D, false, st_);
            }
            cuda::vis_to_f16(O32, (size_t)c * D, O16, st_);
            cuda::gemm_f16(cublas_, Wout, D, D, O16, D, c, O32, D, false, st_);
            cuda::vis_residual(X + (size_t)t0 * D, O32, outb, c, D, st_);
        }
        // MLP
        cuda::vis_layernorm(X, ln2w, ln2b, N, D, eps, XN, st_);
        for (int t0 = 0; t0 < N; t0 += C) {
            const int c = std::min(C, N - t0);
            cuda::gemm_f16(cublas_, Wup, FF, D, XN + (size_t)t0 * D, D, c, U32, FF, false, st_);
            cuda::vis_bias_gelu(U32, upb, c, FF, U16, st_);
            cuda::gemm_f16(cublas_, Wdown, D, FF, U16, FF, c, O32, D, false, st_);
            cuda::vis_residual(X + (size_t)t0 * D, O32, downb, c, D, st_);
        }
        done();
    }

    // ---- post norm, 2x2 merge (4 consecutive patches = one row of 4608), projector ----
    cuda::vis_layernorm(X, sm_post, sm_post + D, N, D, eps, XN, st_);
    float* H32 = (float*)b[B_QK];
    __half* H16 = (__half*)b[B_VT];
    float* OUT = (float*)b[B_R];
    for (const Piece& pc : mm0_pieces) {
        const __half* W = take();
        cuda::gemm_f16(cublas_, W, pc.n, V::MERGED, XN, V::MERGED, T, H32 + pc.r0, V::MERGED, false, st_);
        done();
    }
    cuda::vis_bias_gelu(H32, sm_mm, T, V::MERGED, H16, st_);
    for (const Piece& pc : mm2_pieces) {
        const __half* W = take();
        cuda::gemm_f16(cublas_, W, pc.n, V::MERGED, H16, V::MERGED, T, OUT + pc.r0, V::OUT, false, st_);
        done();
    }
    cuda::vis_bias(OUT, sm_mm + V::MERGED, T, V::OUT, st_);
    out.resize((size_t)T * V::OUT);
    CHECK_CUDA(cudaMemcpyAsync(out.data(), OUT, out.size() * 4, cudaMemcpyDeviceToHost, st_));
    CHECK_CUDA(cudaStreamSynchronize(st_));
    for (int i = 0; i < 2; i++) { cudaEventDestroy(ready[i]); cudaEventDestroy(freed[i]); }
    sseq_ = sissued_ = 0;   // the decode ring held encoder weights: the next pass re-issues its first pieces
    scratch_release();
    return true;
}

} // namespace q38rt
