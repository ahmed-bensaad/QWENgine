#!/usr/bin/env python3
"""Run a decode benchmark and append an immutable JSON record to bench/results/.

Usage: bench/run_bench.py --label "what changed" [--ctx 8192] [--kv q8_0] [--depth 0]
                          [--synthetic-depth N] [--gen 48] [--extra "args for q38_eval"]
Samples GPU utilization / VRAM (nvidia-smi) and CPU utilization (/proc/stat) during the run.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("Q38_MODEL", os.path.join(ROOT, "models", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"))

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def cpu_times():
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    idle = v[3] + v[4]
    return sum(v), idle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--synthetic-depth", type=int, default=0)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cmd = (f"{ROOT}/build/q38_eval {MODEL} bench --ctx {a.ctx} --kv {a.kv} --depth {a.depth} "
           f"--synthetic-depth {a.synthetic_depth} --gen {a.gen} {a.extra}")
    samples = {"gpu_util": [], "vram_mib": [], "cpu_util": []}
    stop = threading.Event()
    def sampler():
        t0, i0 = cpu_times()
        while not stop.is_set():
            time.sleep(0.5)
            out = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
            if out:
                g, m = out.split(",")
                samples["gpu_util"].append(float(g)); samples["vram_mib"].append(float(m))
            t1, i1 = cpu_times()
            if t1 > t0:
                samples["cpu_util"].append(100.0 * (1 - (i1 - i0) / (t1 - t0)))
            t0, i0 = t1, i1
    th = threading.Thread(target=sampler); th.start()
    t_start = time.time()
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    stop.set(); th.join()
    out = proc.stdout + proc.stderr
    m = re.search(r"RESULT (.*)", out)
    if not m:
        print(out); raise SystemExit("benchmark failed")
    res = dict(kv.split("=") for kv in m.group(1).split())
    res = {k: float(v) for k, v in res.items()}
    # samples during the decode phase are a subset; report the busy-phase maxima / means
    def stats(xs):
        return {"mean": round(sum(xs) / len(xs), 1), "max": round(max(xs), 1)} if xs else None
    head = open(MODEL, "rb").read(64 << 20)
    rec = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": a.label,
        "commit": sh(f"git -C {ROOT} rev-parse --short HEAD") + ("-dirty" if sh(f"git -C {ROOT} status --porcelain -uno") else ""),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "cuda_toolkit": sh("nvcc --version | tail -1"),
        "model": os.path.basename(MODEL),
        "model_sha256_first64MiB": hashlib.sha256(head).hexdigest()[:16],
        "context_capacity": a.ctx, "kv": a.kv, "depth_real": a.depth, "depth_synthetic": a.synthetic_depth,
        "gen_tokens": a.gen,
        "decode_tok_s": res["decode_tok_s"], "median_ms_per_token": res["median_ms"],
        "prefill_tok_s": res["prefill_tok_s"] or None,
        "vram_engine_mib": res["vram_mib"], "host_weight_mib": res["host_weight_mib"],
        "planner_estimate_ms": res["est_ms"], "cpu_work_ms_per_token": res["cpu_ms"],
        "gpu_wait_ms_per_token": res["wait_ms"], "lm_head_ms_per_token": res["lm_ms"],
        "pcie_note": "per token H2D/D2H ~0.5 MB of activations; weights do not cross PCIe in decode",
        "gpu_util_pct": stats(samples["gpu_util"]), "vram_total_used_mib": stats(samples["vram_mib"]),
        "cpu_util_pct": stats(samples["cpu_util"]),
        "wall_s": round(time.time() - t_start, 1), "cmd": cmd,
    }
    os.makedirs(f"{ROOT}/bench/results", exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", a.label).strip("-")[:40]
    path = f"{ROOT}/bench/results/{rec['timestamp'].replace(':', '-')}_{slug}_ctx{a.synthetic_depth + a.depth}_{a.kv}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: rec[k] for k in ("label", "commit", "kv", "depth_synthetic", "decode_tok_s", "median_ms_per_token",
                                         "cpu_work_ms_per_token", "gpu_wait_ms_per_token", "lm_head_ms_per_token",
                                         "vram_engine_mib", "host_weight_mib", "gpu_util_pct", "cpu_util_pct")}))
    print("saved", path)

if __name__ == "__main__":
    main()
