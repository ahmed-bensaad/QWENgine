/// @file planner.cpp
/// Greedy residency planning (see planner.h).

#include "runtime/planner.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <queue>
#include <sstream>

namespace q38rt {

namespace {
bool g_cpu_r4 = true, g_cpu_r2 = true;
bool is_iq3(QuantFormat t) { return t == QuantFormat::IQ3_S || t == QuantFormat::IQ3_XXS; }
bool is_iq2(QuantFormat t) { return t == QuantFormat::IQ2_XXS || t == QuantFormat::IQ2_XS || t == QuantFormat::IQ2_S; }
bool use_re(QuantFormat t, int re) { return re < 0 ? DeviceRates::reencoded(t) : re != 0 && (is_iq3(t) || is_iq2(t)); }
} // namespace
void DeviceRates::set_cpu_reencode(bool r4, bool r2) { g_cpu_r4 = r4; g_cpu_r2 = r2; }
bool DeviceRates::reencoded(QuantFormat t) { return (g_cpu_r4 && is_iq3(t)) || (g_cpu_r2 && is_iq2(t)); }

DeviceRates DeviceRates::measured_defaults(int re) {
    // Measured on the target machine: CPU 2026-10-15 (tests/kernels/test_cpu_gemv --bench, 8 threads, 512 MB
    // working set); GPU 2026-10-16 (tests/kernels/test_gemv --bench, conservative average over K = 5120 / 6144 /
    // 17408). GB/s of weight bytes.
    DeviceRates r;
    auto set = [&](QuantFormat t, double c, double g) { r.cpu_gbs[(int)t] = c; r.gpu_gbs[(int)t] = g; };
    // (Q2_K, Q4_K, IQ2_*: scaled to the current kernels by speed ratios from interleaved runs, 2026-10-17)
    set(QuantFormat::Q2_K, 31.4, 175);
    set(QuantFormat::Q4_K, 32.1, 220);
    // (R2: scaled by the re-encoded / gather single-column ratios, GB/s of source bytes)
    set(QuantFormat::IQ2_XXS, use_re(QuantFormat::IQ2_XXS, re) ? 23.8 : 19.0, 140);
    set(QuantFormat::IQ2_XS, use_re(QuantFormat::IQ2_XS, re) ? 21.2 : 19.5, 155);
    set(QuantFormat::IQ2_S, use_re(QuantFormat::IQ2_S, re) ? 25.1 : 23.4, 165);
    set(QuantFormat::IQ3_XXS, use_re(QuantFormat::IQ3_XXS, re) ? 21.3 : 21.7, 180);   // R4: GB/s of source bytes (R4/gather 0.98)
    set(QuantFormat::IQ3_S, 25.9, 182);
    set(QuantFormat::IQ1_S, 14.9, 85);
    set(QuantFormat::IQ1_M, 8.7, 122);
    set(QuantFormat::IQ4_XS, 31.6, 320);
    set(QuantFormat::Q6_K, 33.0, 210);    // MTP block only (measured on the -mtp.gguf tensors)
    // sustained decode runs the CPU thermally throttled (~2.6 GHz at 100 C vs the cool micro-benchmarks); planning
    // with rates scaled by 0.8 measured best (plain decode A/B: 1.0 / 0.8 / 0.65 -> 7.49 / 7.66 / 7.51 tok/s)
    double scale = 0.8;
    if (const char* e = getenv("Q38_CPU_RATE_SCALE")) scale = atof(e);
    for (double& c : r.cpu_gbs) c *= scale;
    return r;
}

double DeviceRates::cpu_cols_factor(QuantFormat t, int n, int re) {
    // tests/kernels/test_cpu_gemv --bench, 8 threads: pass time for n = 2..7 columns vs 1. n = 2-4: average of 2
    // interleaved runs (2026-10-17); n = 5-7: one run (2026-10-22)
    if (n <= 1) return 1.0;
    n = std::min(n, 7);
    struct F { QuantFormat t; double f[6]; };
    static const F tab[] = {
        {QuantFormat::IQ4_XS, {1.02, 1.05, 1.18, 1.23, 1.44, 1.60}}, {QuantFormat::IQ3_S, {1.22, 1.53, 1.64, 1.66, 1.76, 2.07}},
        {QuantFormat::IQ3_XXS, {1.12, 1.25, 1.46, 1.71, 1.92, 2.04}}, {QuantFormat::IQ2_S, {1.25, 1.53, 1.86, 1.86, 2.13, 2.41}},
        {QuantFormat::IQ2_XS, {1.16, 1.50, 1.70, 2.06, 2.06, 2.26}}, {QuantFormat::IQ2_XXS, {1.18, 1.39, 1.61, 1.78, 2.04, 2.32}},
        {QuantFormat::Q2_K, {1.11, 1.43, 1.65, 1.74, 1.95, 2.34}}, {QuantFormat::Q4_K, {1.04, 1.12, 1.26, 1.26, 1.30, 1.61}},
        {QuantFormat::IQ1_S, {1.21, 1.60, 2.00, 2.23, 2.60, 2.94}}, {QuantFormat::IQ1_M, {1.08, 1.22, 1.65, 1.65, 1.65, 1.66}},
        {QuantFormat::Q6_K, {1.03, 1.03, 1.08, 1.15, 1.25, 1.35}}};
    // re-encoded rows: R4 and R2, averages of 3 runs; n = 5-7 one run
    static const F tab_re[] = {{QuantFormat::IQ3_S, {1.05, 1.08, 1.24, 1.24, 1.41, 1.60}},
                               {QuantFormat::IQ3_XXS, {1.03, 1.08, 1.20, 1.27, 1.47, 1.61}},
                               {QuantFormat::IQ2_XXS, {1.01, 1.21, 1.52, 2.12, 2.20, 2.25}},
                               {QuantFormat::IQ2_XS, {1.00, 1.34, 1.42, 1.51, 1.88, 1.97}},
                               {QuantFormat::IQ2_S, {1.06, 1.22, 1.44, 1.49, 1.79, 1.94}}};
    if (use_re(t, re)) for (const F& e : tab_re) if (e.t == t) return e.f[n - 2];
    for (const F& e : tab) if (e.t == t) return e.f[n - 2];
    return 1.0 + 0.25 * (n - 1);
}

double DeviceRates::gpu_cols_factor(int n) {
    // tests/kernels/test_gemv --bench (2026-10-16), average over formats
    static const double f[5] = {1.0, 1.0, 1.30, 1.65, 2.00};
    return f[std::max(1, std::min(n, 4))];
}

namespace {

struct GroupState {
    double cpu = 0, gpu = 0;
};

double group_time(const GroupState& g) { return std::max(g.cpu, g.gpu); }

} // namespace

void plan_residency(Plan& plan) {
    const int ng = (int)plan.groups.size();
    std::vector<GroupState> gs(ng);
    for (int g = 0; g < ng; g++) gs[g].gpu = plan.groups[g].gpu_fixed_s;
    double used = 0;
    for (PlanItem& it : plan.items) {
        it.gpu_units = std::min(it.min_gpu_units, it.units);
        gs[it.group].cpu += (it.units - it.gpu_units) * it.cpu_s_per_unit;
        gs[it.group].gpu += it.gpu_units * it.gpu_s_per_unit;
        used += it.gpu_units * it.bytes_per_unit;
    }
    // Move units in chunks; benefit per byte = reduction of the group's max(cpu, gpu).
    // Priority: largest benefit per byte. Recompute lazily (group state changes).
    struct Cand { double score; int item; };
    auto chunk_of = [](const PlanItem& it) { return std::max(1, it.units / 64); };
    auto benefit = [&](int i, int* n_out) -> double {
        const PlanItem& it = plan.items[i];
        const int left = it.units - it.gpu_units;
        if (left <= 0) return -1;
        const int n = std::min(left, chunk_of(it));
        *n_out = n;
        const GroupState& g = gs[it.group];
        GroupState g2 = g;
        g2.cpu -= n * it.cpu_s_per_unit;
        g2.gpu += n * it.gpu_s_per_unit;
        const double gain = plan.groups[it.group].weight * (group_time(g) - group_time(g2));
        return gain / (n * it.bytes_per_unit);
    };
    std::priority_queue<std::pair<double, int>> pq;
    for (int i = 0; i < (int)plan.items.size(); i++) {
        int n;
        const double b = benefit(i, &n);
        if (b > 0) pq.push({b, i});
    }
    while (!pq.empty()) {
        auto [score, i] = pq.top();
        pq.pop();
        int n;
        const double b = benefit(i, &n);
        if (b <= 0) continue;
        if (b < score * 0.999) { pq.push({b, i}); continue; }   // stale: re-queue with fresh score
        PlanItem& it = plan.items[i];
        const double bytes = n * it.bytes_per_unit;
        if (used + bytes > plan.budget_bytes) {
            // fit as many units as remain in the budget
            const int fit = (int)std::floor((plan.budget_bytes - used) / it.bytes_per_unit);
            if (fit <= 0) continue;
            n = std::min(n, fit);
        }
        it.gpu_units += n;
        used += n * it.bytes_per_unit;
        gs[it.group].cpu -= n * it.cpu_s_per_unit;
        gs[it.group].gpu += n * it.gpu_s_per_unit;
        int n2;
        const double b2 = benefit(i, &n2);
        if (b2 > 0) pq.push({b2, i});
    }
    plan.gpu_bytes = used;
    plan.cpu_bytes = 0;
    plan.est_cpu_s = plan.est_gpu_s = plan.est_total_s = 0;
    for (PlanItem& it : plan.items) {
        plan.cpu_bytes += (it.units - it.gpu_units) * it.bytes_per_unit;
        if (it.gpu_units == it.units) it.reason = "all on GPU: budget allowed and GPU time stays hidden";
        else if (it.gpu_units == 0) it.reason = "all on CPU: VRAM spent where it saves more CPU time per byte";
        else it.reason = "split: group balanced or VRAM budget reached";
    }
    for (int g = 0; g < ng; g++) {
        const double w = plan.groups[g].weight;
        plan.est_cpu_s += w * gs[g].cpu;
        plan.est_gpu_s += w * gs[g].gpu;
        plan.est_total_s += w * group_time(gs[g]);
    }
}

std::string plan_summary(const Plan& p) {
    std::ostringstream o;
    char buf[256];
    snprintf(buf, sizeof buf, "budget %.0f MiB | GPU weights %.0f MiB | CPU weights %.0f MiB\n", p.budget_bytes / 1048576.0,
             p.gpu_bytes / 1048576.0, p.cpu_bytes / 1048576.0);
    o << buf;
    snprintf(buf, sizeof buf, "estimated per token: CPU %.1f ms, GPU %.1f ms, sum of group maxima %.1f ms (%.2f tok/s)\n",
             p.est_cpu_s * 1e3, p.est_gpu_s * 1e3, p.est_total_s * 1e3, 1.0 / p.est_total_s);
    o << buf;
    std::map<std::string, std::pair<double, double>> by_kind;   // gpu bytes, cpu bytes
    for (const PlanItem& it : p.items) {
        std::string kind = it.name.substr(it.name.find('.', 4) + 1);
        auto& e = by_kind[kind];
        e.first += it.gpu_units * it.bytes_per_unit;
        e.second += (it.units - it.gpu_units) * it.bytes_per_unit;
    }
    for (auto& kv : by_kind) {
        snprintf(buf, sizeof buf, "  %-22s GPU %8.1f MiB  CPU %8.1f MiB\n", kv.first.c_str(), kv.second.first / 1048576.0,
                 kv.second.second / 1048576.0);
        o << buf;
    }
    return o.str();
}

} // namespace q38rt
