#pragma once
/// @file iq_tables.h
/// Host copies of the ggml i-quant lookup tables (see iq_tables.inc).
#include <cstdint>

#ifndef IQ1S_DELTA
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
#endif

namespace q38rt {
#define Q38_TABLE_QUALIFIER static const
#include "quant/iq_tables.inc"
#undef Q38_TABLE_QUALIFIER
} // namespace q38rt
