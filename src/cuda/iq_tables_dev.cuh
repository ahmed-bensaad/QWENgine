#pragma once
/// @file iq_tables_dev.cuh
/// Device copies of the ggml i-quant lookup tables (one copy per translation unit, as in ggml-cuda).
#include <cstdint>

#ifndef IQ1S_DELTA
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
#endif

namespace q38rt {
namespace cuda {
#define Q38_TABLE_QUALIFIER static __device__ const
#include "quant/iq_tables.inc"
#undef Q38_TABLE_QUALIFIER
} // namespace cuda
} // namespace q38rt
