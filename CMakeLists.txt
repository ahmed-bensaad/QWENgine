cmake_minimum_required(VERSION 3.24)

# nvcc 12.4 does not accept GCC 15 as host compiler: use GCC 12 when it is installed (else nvcc's default).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(Q38_GXX12 g++-12)
    if(Q38_GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${Q38_GXX12} CACHE STRING "nvcc host compiler")
    endif()
endif()
# Tuned on the Quadro RTX 4000 (TU104, sm_75). Other GPUs: -DCMAKE_CUDA_ARCHITECTURES=native, or e.g. 86 / 89; the
# kernels need compute capability 7.5 or newer (FP16 m16n8k8 tensor-core instructions).
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures (75 = Turing, the tuned target)")

project(q38rt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Host code targets the build machine (developed on an i9-9880H: AVX2/FMA/F16C, no AVX-512). The CPU kernels are
# written with AVX2 intrinsics: stop here with a clear message on CPUs without them.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS -march=native)
check_cxx_source_compiles("#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error missing
#endif
int main() { return 0; }" Q38_HAVE_AVX2)
unset(CMAKE_REQUIRED_FLAGS)
if(NOT Q38_HAVE_AVX2)
    message(FATAL_ERROR "QWENgine's CPU kernels need an x86-64 CPU with AVX2, FMA and F16C (-march=native lacks them)")
endif()
set(Q38_CXX_FLAGS -O3 -march=native -mtune=native -Wall -Wno-unused-function)
set(Q38_CUDA_FLAGS --expt-relaxed-constexpr -lineinfo -Xptxas=-v,-warn-spills
    -Xcompiler=-O3,-march=native)

# Frokenizer (the Qwen pre-tokenizer lexer) is a git submodule
set(Q38_FROKENIZER ${CMAKE_SOURCE_DIR}/third_party/frokenizer/include)
if(NOT EXISTS ${Q38_FROKENIZER}/frokenizer_generated/lexer.hpp)
    message(FATAL_ERROR "third_party/frokenizer is missing: run 'git submodule update --init' (or clone with --recursive)")
endif()
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${Q38_FROKENIZER})

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(q38rt_core STATIC
    src/runtime/gguf_parser.cpp
    src/quant/quant_ref.cpp
    src/reference/ref_model.cpp
    src/cpu/cpu_kernels.cpp
    src/runtime/tokenizer.cpp
)
target_compile_options(q38rt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${Q38_CXX_FLAGS}>
                                          $<$<COMPILE_LANGUAGE:CUDA>:${Q38_CUDA_FLAGS}>)
find_package(OpenMP REQUIRED)
target_link_libraries(q38rt_core PUBLIC CUDA::cudart Threads::Threads OpenMP::OpenMP_CXX)
# The reference (de)quantizers must follow ISO C rounding exactly (no FMA contraction),
# otherwise rare rounding differences appear vs. ggml.
set_source_files_properties(src/quant/quant_ref.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)

# CUDA kernels
add_library(q38rt_cuda STATIC
    src/cuda/gemv.cu
    src/cuda/ops.cu
    src/cuda/prefill.cu
    src/cuda/vision.cu
)
target_compile_options(q38rt_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q38_CUDA_FLAGS}>)
target_link_libraries(q38rt_cuda PUBLIC q38rt_core CUDA::cublas)

# Engine (hybrid CPU+GPU decode)
add_library(q38rt_engine STATIC
    src/runtime/planner.cpp
    src/runtime/engine.cpp
    src/runtime/vision.cpp
    src/runtime/vision_encode.cpp
)
target_compile_options(q38rt_engine PRIVATE ${Q38_CXX_FLAGS})
target_compile_definitions(q38rt_engine PUBLIC Q38_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
target_link_libraries(q38rt_engine PUBLIC q38rt_cuda)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
add_executable(q38_eval tools/q38_eval.cpp)
target_compile_options(q38_eval PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(q38_eval PRIVATE q38rt_engine)

# OpenAI-compatible server: JSON, HTTP, chat template + output parsing, generation (speculative decoding,
# sampling, prompt reuse), API endpoints
add_library(q38rt_server STATIC
    src/server/json.cpp
    src/server/http.cpp
    src/server/chat.cpp
    src/server/generator.cpp
    src/server/api.cpp
    src/server/metrics.cpp
)
target_compile_options(q38rt_server PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(q38rt_server PUBLIC q38rt_engine)
add_executable(q38_server tools/q38_server.cpp)
target_compile_options(q38_server PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(q38_server PRIVATE q38rt_server)

add_executable(ref_compare tools/ref_compare.cpp)
target_link_libraries(ref_compare PRIVATE q38rt_core)

add_executable(inspect_gguf tools/inspect_gguf.cpp)
target_link_libraries(inspect_gguf PRIVATE q38rt_core)

add_library(hw_probe_cpu OBJECT tools/hw_probe_cpu.cpp)
target_compile_options(hw_probe_cpu PRIVATE -O3 -mavx2 -fopenmp)
add_executable(hw_probe tools/hw_probe.cu $<TARGET_OBJECTS:hw_probe_cpu>)
target_link_options(hw_probe PRIVATE -fopenmp)
target_link_libraries(hw_probe PRIVATE CUDA::cudart)

# ---------------------------------------------------------------------------
# Tests (ctest). Model-dependent tests read Q38_MODEL / Q38_MTP_MODEL and are disabled when the files (or the
# generated golden data) are missing.
# ---------------------------------------------------------------------------
enable_testing()
set(Q38_MODEL_DIR "${CMAKE_SOURCE_DIR}/models" CACHE PATH "Directory holding the two GGUF files")
set(Q38_MODEL "${Q38_MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
    CACHE FILEPATH "Target GGUF used by model-level tests")
set(Q38_MTP_MODEL "${Q38_MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
    CACHE FILEPATH "Same model with the MTP (nextn) block, Q6_K")
set(Q38_MMPROJ "${Q38_MODEL_DIR}/mmproj-Qwen3.8-27B-BF16.gguf"
    CACHE FILEPATH "Vision encoder (optional: image tests run when it exists)")
foreach(F ${Q38_MODEL} ${Q38_MTP_MODEL})
    if(NOT EXISTS ${F})
        message(STATUS "model file not found: ${F} (tests that need it will not run; see README)")
    endif()
endforeach()

add_executable(test_json tests/unit/test_json.cpp src/server/json.cpp)
target_compile_options(test_json PRIVATE ${Q38_CXX_FLAGS})
add_test(NAME unit.json COMMAND test_json)

add_executable(test_chat tests/unit/test_chat.cpp)
target_compile_options(test_chat PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(test_chat PRIVATE q38rt_server)
if(EXISTS ${Q38_MODEL})   # with the model: also checks the tokenization of the split prompt
    add_test(NAME unit.chat COMMAND test_chat ${CMAKE_SOURCE_DIR}/tests/unit/data/chat_cases.json ${Q38_MODEL})
else()
    add_test(NAME unit.chat COMMAND test_chat ${CMAKE_SOURCE_DIR}/tests/unit/data/chat_cases.json)
endif()

add_executable(test_metrics tests/unit/test_metrics.cpp)
target_compile_options(test_metrics PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(test_metrics PRIVATE q38rt_server)
add_test(NAME unit.metrics COMMAND test_metrics)

add_executable(test_image tests/unit/test_image.cpp)
target_compile_options(test_image PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(test_image PRIVATE q38rt_engine)
add_test(NAME unit.image_resize COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/unit/test_image_resize.py $<TARGET_FILE:test_image>)

add_executable(test_gguf_parser tests/unit/test_gguf_parser.cpp)
target_link_libraries(test_gguf_parser PRIVATE q38rt_core)
add_test(NAME unit.gguf_parser COMMAND test_gguf_parser ${Q38_MODEL})

add_executable(test_cpu_gemv tests/kernels/test_cpu_gemv.cpp)
target_compile_options(test_cpu_gemv PRIVATE ${Q38_CXX_FLAGS})
target_link_libraries(test_cpu_gemv PRIVATE q38rt_core)
add_test(NAME kernels.cpu_gemv COMMAND test_cpu_gemv ${Q38_MODEL})
add_test(NAME kernels.cpu_gemv.mtp COMMAND test_cpu_gemv ${Q38_MTP_MODEL})

add_executable(test_ops tests/kernels/test_ops.cu)
target_compile_options(test_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q38_CUDA_FLAGS}>)
target_link_libraries(test_ops PRIVATE q38rt_cuda)
add_test(NAME kernels.ops COMMAND test_ops ${Q38_MODEL})

add_executable(test_attn_prefill tests/kernels/test_attn_prefill.cu)
target_compile_options(test_attn_prefill PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q38_CUDA_FLAGS}>)
target_link_libraries(test_attn_prefill PRIVATE q38rt_cuda)
add_test(NAME kernels.attn_prefill COMMAND test_attn_prefill)

add_executable(test_gemv tests/kernels/test_gemv.cu)
target_compile_options(test_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q38_CUDA_FLAGS}>)
target_link_libraries(test_gemv PRIVATE q38rt_cuda)
add_test(NAME kernels.gemv COMMAND test_gemv ${Q38_MODEL})
add_test(NAME kernels.gemv.mtp COMMAND test_gemv ${Q38_MTP_MODEL})

# Trusted reference implementation (a llama.cpp build with shared libraries) used only by tests and tools.
set(Q38_GGML_DIR "" CACHE PATH "Directory with libggml-base.so / libggml-cpu.so / libllama.so of a llama.cpp build")
if(Q38_GGML_DIR)
    find_library(GGML_BASE_LIB ggml-base PATHS ${Q38_GGML_DIR} NO_DEFAULT_PATH)
endif()
if(GGML_BASE_LIB)
    add_executable(test_quant_ref tests/unit/test_quant_ref.cpp)
    target_link_libraries(test_quant_ref PRIVATE q38rt_core ${GGML_BASE_LIB})
    set_target_properties(test_quant_ref PROPERTIES BUILD_RPATH ${Q38_GGML_DIR})
    add_test(NAME unit.quant_ref COMMAND test_quant_ref ${Q38_MODEL})
    add_test(NAME unit.quant_ref.mtp COMMAND test_quant_ref ${Q38_MTP_MODEL})
    find_library(LLAMA_LIB llama PATHS ${Q38_GGML_DIR} NO_DEFAULT_PATH)
    get_filename_component(Q38_LLAMA_DEFAULT_INC "${Q38_GGML_DIR}/../.." ABSOLUTE)   # <llama.cpp>/build/bin
    set(Q38_LLAMA_INC "${Q38_LLAMA_DEFAULT_INC}" CACHE PATH "llama.cpp source dir (headers)")
    add_executable(test_tokenizer tests/unit/test_tokenizer.cpp)
    target_include_directories(test_tokenizer PRIVATE ${Q38_LLAMA_INC}/include ${Q38_LLAMA_INC}/ggml/include)
    target_link_libraries(test_tokenizer PRIVATE q38rt_core ${LLAMA_LIB} ${GGML_BASE_LIB})
    set_target_properties(test_tokenizer PROPERTIES BUILD_RPATH ${Q38_GGML_DIR})
    add_test(NAME unit.tokenizer COMMAND test_tokenizer ${Q38_MODEL}
             ${CMAKE_SOURCE_DIR}/tests/golden/short.txt ${CMAKE_SOURCE_DIR}/tests/golden/chat.txt
             ${CMAKE_SOURCE_DIR}/tests/golden/code.txt ${CMAKE_SOURCE_DIR}/tests/golden/multilingual.txt)
    # golden logits generator (tests/golden/make_golden.sh)
    add_executable(ref_llama_logits tools/ref_llama_logits.cpp)
    target_include_directories(ref_llama_logits PRIVATE ${Q38_LLAMA_INC}/include ${Q38_LLAMA_INC}/ggml/include)
    target_link_libraries(ref_llama_logits PRIVATE ${LLAMA_LIB} ${GGML_BASE_LIB})
    set_target_properties(ref_llama_logits PROPERTIES BUILD_RPATH ${Q38_GGML_DIR})
    # ggml's own CPU dot products on the same tensors (baseline for the AVX2 kernels in src/cpu)
    find_library(GGML_CPU_LIB ggml-cpu PATHS ${Q38_GGML_DIR} NO_DEFAULT_PATH)
    add_executable(bench_cpu_ggml tools/bench_cpu_ggml.cpp)
    target_include_directories(bench_cpu_ggml PRIVATE ${Q38_LLAMA_INC}/ggml/include)
    target_compile_options(bench_cpu_ggml PRIVATE ${Q38_CXX_FLAGS})
    target_link_libraries(bench_cpu_ggml PRIVATE q38rt_core ${GGML_CPU_LIB} ${GGML_BASE_LIB})
    set_target_properties(bench_cpu_ggml PROPERTIES BUILD_RPATH ${Q38_GGML_DIR})
    # reference image embeddings from llama.cpp's multimodal library (vision encoder test)
    find_library(MTMD_LIB mtmd PATHS ${Q38_GGML_DIR} NO_DEFAULT_PATH)
    if(MTMD_LIB)
        add_executable(ref_mtmd_embd tools/ref_mtmd_embd.cpp)
        target_include_directories(ref_mtmd_embd PRIVATE ${Q38_LLAMA_INC}/include ${Q38_LLAMA_INC}/ggml/include
                                   ${Q38_LLAMA_INC}/tools/mtmd)
        target_link_libraries(ref_mtmd_embd PRIVATE ${MTMD_LIB} ${LLAMA_LIB} ${GGML_BASE_LIB})
        set_target_properties(ref_mtmd_embd PROPERTIES BUILD_RPATH ${Q38_GGML_DIR})
    endif()
else()
    message(STATUS "no llama.cpp libraries (Q38_GGML_DIR='${Q38_GGML_DIR}'): reference tests and tools disabled")
endif()

# Golden regression: engine vs FP32 reference logits (data from tests/golden/make_golden.sh + make_ref.sh).
set(Q38_GOLDEN ${CMAKE_SOURCE_DIR}/tests/golden/data)
foreach(P short chat code multilingual)
    add_test(NAME golden.${P}
             COMMAND q38_eval ${Q38_MODEL} golden ${CMAKE_SOURCE_DIR}/tests/golden/data/${P}.llama_gpu.bin
                     --ref ${CMAKE_SOURCE_DIR}/tests/golden/data/${P}.ref.bin --kv f16 --ctx 4096
                     --min-top1 30 --max-kl 0.002)
    set_tests_properties(golden.${P} PROPERTIES REQUIRED_FILES "${Q38_GOLDEN}/${P}.llama_gpu.bin;${Q38_GOLDEN}/${P}.ref.bin")
endforeach()

# Batched prefill correctness: prompt rows from prefill_batched, generated rows from decode.
foreach(P short code)
    add_test(NAME golden.prefill.${P}
             COMMAND q38_eval ${Q38_MODEL} golden ${CMAKE_SOURCE_DIR}/tests/golden/data/${P}.llama_gpu.bin
                     --ref ${CMAKE_SOURCE_DIR}/tests/golden/data/${P}.ref.bin --kv f16 --ctx 4096 --batched-prefill
                     --chunk 64 --min-top1 30 --max-kl 0.002)
    set_tests_properties(golden.prefill.${P} PROPERTIES REQUIRED_FILES "${Q38_GOLDEN}/${P}.llama_gpu.bin;${Q38_GOLDEN}/${P}.ref.bin")
endforeach()

# Vision encoder vs llama.cpp's mtmd on the CPU (data from tests/golden/make_vision_ref.sh): per-token cosine of the
# image embeddings, mean over the image (measured >= 0.99996; llama.cpp's own CUDA path: 0.9995 against its CPU path)
# and minimum (measured 0.9995 and 0.9999; on the 1024x1024 image 0.982 at one token whose embedding is ill-conditioned:
# llama.cpp's CUDA and CPU paths agree there only to 0.927)
foreach(I 768x512:0.995 320x640:0.995 1024x1024:0.97)
    string(REPLACE ":" ";" IT ${I})
    list(GET IT 0 I)
    list(GET IT 1 MINCOS)
    add_test(NAME golden.vision.${I}
             COMMAND q38_eval ${Q38_MODEL} vision --mmproj ${Q38_MMPROJ} --image ${Q38_GOLDEN}/vision_${I}.ppm
                     --ref ${Q38_GOLDEN}/vision_${I}.ref.bin --ctx 4096 --min-cos ${MINCOS} --min-mean-cos 0.9999)
    set_tests_properties(golden.vision.${I} PROPERTIES
                         REQUIRED_FILES "${Q38_MMPROJ};${Q38_GOLDEN}/vision_${I}.ppm;${Q38_GOLDEN}/vision_${I}.ref.bin")
endforeach()

# Multi-token decode (MTP verification path): decode_batch must be bit-identical to sequential decode (top-k and
# full logits), for every KV type.
foreach(KV q8_0 q4_0)
    add_test(NAME integration.decode_batch.${KV} COMMAND q38_eval ${Q38_MODEL} batchtest --ctx 4096 --kv ${KV} --depth 32)
endforeach()

# MTP speculative decoding: greedy output with drafts (1 and 3 per step) must equal plain greedy decoding exactly.
foreach(K 1 3 auto)
    add_test(NAME integration.spec.k${K} COMMAND q38_eval ${Q38_MTP_MODEL} spec --ctx 4096 --kv q8_0 --prompt chat --gen 40 --draft ${K})
endforeach()
# prompt through batched prefill (fills the MTP KV cache across chunk boundaries)
add_test(NAME integration.spec.prefill COMMAND q38_eval ${Q38_MTP_MODEL} spec --ctx 4096 --kv q8_0 --prompt code --gen 40
         --draft auto --batched-prefill --chunk 16)
# image prompt (M-RoPE positions, image embeddings in the trunk and the MTP block) across prefill chunk boundaries
add_test(NAME integration.spec.image COMMAND q38_eval ${Q38_MTP_MODEL} spec --ctx 4096 --kv q8_0
         --prompt "Describe this image." --gen 40 --draft auto --chunk 64 --mmproj ${Q38_MMPROJ}
         --image ${Q38_GOLDEN}/vision_320x640.ppm)
set_tests_properties(integration.spec.image PROPERTIES REQUIRED_FILES "${Q38_MMPROJ};${Q38_GOLDEN}/vision_320x640.ppm")
# Prefill buffers placed in the borrowed LM-head rows (normally only when free VRAM is short), restored afterwards:
# prefill logits, decode after it and the MTP draft head must be unaffected
add_test(NAME golden.prefill.borrow
         COMMAND q38_eval ${Q38_MODEL} golden ${CMAKE_SOURCE_DIR}/tests/golden/data/code.llama_gpu.bin
                 --ref ${CMAKE_SOURCE_DIR}/tests/golden/data/code.ref.bin --kv f16 --ctx 4096 --batched-prefill
                 --chunk 64 --min-top1 30 --max-kl 0.002)
add_test(NAME integration.spec.prefill.borrow COMMAND q38_eval ${Q38_MTP_MODEL} spec --ctx 4096 --kv q8_0 --prompt code
         --gen 40 --draft auto --batched-prefill --chunk 16)
set_tests_properties(golden.prefill.borrow integration.spec.prefill.borrow PROPERTIES ENVIRONMENT Q38_PREFILL_BORROW=1)
set_tests_properties(golden.prefill.borrow PROPERTIES REQUIRED_FILES "${Q38_GOLDEN}/code.llama_gpu.bin;${Q38_GOLDEN}/code.ref.bin")

# Snapshot save / restore (conversation reuse in the server) is exact
add_test(NAME integration.snapshot COMMAND q38_eval ${Q38_MTP_MODEL} snaptest --ctx 4096 --kv q8_0)

# Context growth: resize mid-sequence (same weight plan via the budget cap) is bit-identical to no resize
add_test(NAME integration.resize COMMAND q38_eval ${Q38_MTP_MODEL} resizetest --ctx 2048 --kv q8_0 --cap-mib 4000 --depth 40)

# OpenAI-compatible server end to end (starts q38_server on a free port)
add_test(NAME integration.server COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/integration/test_server.py
         $<TARGET_FILE:q38_server> ${Q38_MTP_MODEL} ${Q38_MMPROJ})

# Model files each test needs (golden tests above also need their generated data). Tests whose files are missing at
# configure time are disabled (listed by ctest, not counted as failures): re-run cmake after adding them.
get_property(Q38_TESTS DIRECTORY PROPERTY TESTS)
foreach(T IN LISTS Q38_TESTS)
    if(T MATCHES "^unit\\.(json|metrics|chat|image_resize)$" OR T STREQUAL "kernels.attn_prefill")
        continue()
    endif()
    if(T MATCHES "\\.mtp$" OR T MATCHES "^integration\\.(spec|snapshot|resize|server)")
        set(M ${Q38_MTP_MODEL})
    else()
        set(M ${Q38_MODEL})
    endif()
    get_test_property(${T} REQUIRED_FILES R)
    if(R)
        list(APPEND R ${M})
    else()
        set(R ${M})
    endif()
    set_tests_properties(${T} PROPERTIES REQUIRED_FILES "${R}")
    foreach(F IN LISTS R)
        if(NOT EXISTS ${F})
            set_tests_properties(${T} PROPERTIES DISABLED TRUE)
        endif()
    endforeach()
endforeach()
