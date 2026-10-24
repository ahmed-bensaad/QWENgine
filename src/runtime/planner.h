#pragma once
/// @file planner.h
/// Residency planner: decides, for every large weight matrix, how many rows (or FFN columns)
/// live in VRAM (computed by the GPU) versus host DRAM (computed by the CPU), given the VRAM
/// budget and measured per-format CPU/GPU GEMV throughput.
///
/// Model: decode executes a sequence of sync groups; within a group the GPU part and the CPU part
/// run concurrently, so group time ~= max(T_gpu, T_cpu). Every byte must be read once per token,
/// so the planner moves to the GPU, under the VRAM budget, the bytes that save the most CPU time,
/// never pushing a group past its CPU/GPU balance point.

#include <cstdint>
#include <string>
#include <vector>
#include "q38rt/q38rt_types.h"

namespace q38rt {

/// Measured decode GEMV throughput (GB/s) of a weight format on each device.
struct DeviceRates {
    double cpu_gbs[32] = {};
    double gpu_gbs[32] = {};
    /// Values measured on the target machine (see docs). re: host rows of re-encodable formats run the re-encoded
    /// kernels (cpu::re_*: R4 for IQ3, R2 for IQ2) -- 1 all, 0 none, -1 per the process setting (set_cpu_reencode).
    static DeviceRates measured_defaults(int re = -1);
    /// Time of one weight pass computing n (1..4) token columns relative to n = 1 (measured, see docs).
    static double cpu_cols_factor(QuantFormat t, int n, int re = -1);
    static void set_cpu_reencode(bool r4, bool r2);
    static bool reencoded(QuantFormat t);   // per the process setting
    static double gpu_cols_factor(int n);
};

/// One splittable unit: a matrix (or the FFN triple sharing one split) inside a sync group.
struct PlanItem {
    std::string name;
    int group = -1;            // sync group index
    int units = 0;             // split granularity count (rows, or 256-column blocks for the FFN)
    int unit_rows = 1;         // rows (or columns) per unit
    double bytes_per_unit = 0; // weight bytes per unit
    double cpu_s_per_unit = 0; // CPU seconds per unit
    double gpu_s_per_unit = 0; // GPU seconds per unit
    int gpu_units = 0;         // result: units placed on the GPU (prefix)
    int min_gpu_units = 0;     // forced GPU prefix (placed before the greedy, e.g. the MTP draft-head rows)
    std::string reason;        // why it ended where it is
};

struct PlanGroup {
    double gpu_fixed_s = 0;    // GPU-only work in this group (recurrence, attention, ...)
    double weight = 1;         // executions per planned step (e.g. MTP groups run once per draft token)
};

struct Plan {
    std::vector<PlanItem> items;
    std::vector<PlanGroup> groups;
    double budget_bytes = 0;
    double gpu_bytes = 0, cpu_bytes = 0;
    double est_cpu_s = 0, est_gpu_s = 0, est_total_s = 0;   // per token (sum over groups of max)
};

/// Greedy planner: fills `gpu_units` of every item. Returns estimated per-token time.
void plan_residency(Plan& plan);

/// Human-readable summary (per type, per group kind).
std::string plan_summary(const Plan& plan);

} // namespace q38rt
