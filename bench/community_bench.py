#!/usr/bin/env python3
"""The standard benchmark for sharing results from any machine (BENCHMARKS.md says how to submit them).

Describes the machine (GPU, CPU, memory, driver, build), measures it with hw_probe, runs the long-context coding
benchmark (q38_eval longctx: a code base fills the context, then coding tasks are answered from it, as a coding agent
works) at 64K and 128K tokens of context, times the image encoder, and writes bench/community/<date>_<gpu>_<cpu>.json
plus the Markdown row for the table in BENCHMARKS.md.

  python3 bench/community_bench.py [--quick] [--tests] [--name NAME] [--ram "DDR4-2666, 2 channels"] [--notes "..."]

  --quick   64K only, 2 tasks (~6 minutes instead of ~20)
  --tests   also run ctest and record the outcome (recommended on a new GPU or CPU: it checks the kernels there)
  --kv T    KV-cache type (default q4_0, the faster one at long context on 8 GB; say so in the notes if you change it)

Models: Q38_MTP_MODEL (default models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf); Q38_MMPROJ (default
models/mmproj-Qwen3.8-27B-BF16.gguf, optional) for the image encoder timing. Close other GPU programs first; on a
laptop, plug it in. Standard library only.
"""
import argparse, datetime, glob, json, os, platform, re, shlex, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from ab import cool, pkg_temp   # noqa: E402  (CPU package temperature, wait until cooled)

BUILD = os.path.join(ROOT, "build")
MTP = os.environ.get("Q38_MTP_MODEL", os.path.join(ROOT, "models", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"))
MMPROJ = os.environ.get("Q38_MMPROJ", os.path.join(ROOT, "models", "mmproj-Qwen3.8-27B-BF16.gguf"))


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def machine():
    m = {}
    q = sh("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,pcie.link.gen.max,pcie.link.width.max,"
           "clocks.max.sm,compute_cap,power.limit --format=csv,noheader,nounits").split("\n")[0]
    f = [x.strip() for x in q.split(",")] if q else []
    keys = ["gpu", "vram_mib", "vram_used_mib_at_start", "driver", "pcie_gen", "pcie_width", "gpu_max_sm_mhz",
            "compute_cap", "gpu_power_limit_w"]
    for k, v in zip(keys, f):
        m[k] = v
    lscpu = sh("lscpu")
    def field(name):
        r = re.search(r"^" + re.escape(name) + r":\s*(.+)$", lscpu, re.M)
        return r.group(1).strip() if r else ""
    m["cpu"] = field("Model name")
    try:
        m["cpu_cores"] = int(field("Core(s) per socket")) * int(field("Socket(s)") or 1)
        m["cpu_threads"] = m["cpu_cores"] * int(field("Thread(s) per core") or 1)
    except ValueError:
        pass
    flags = read("/proc/cpuinfo")
    m["avx512"] = " avx512f" in flags
    # AVX2 gathers (the CPU IQ kernels) slow down under the Gather Data Sampling mitigation
    m["gather_data_sampling"] = read("/sys/devices/system/cpu/vulnerabilities/gather_data_sampling") or "n/a"
    mem = re.search(r"MemTotal:\s*(\d+)", read("/proc/meminfo"))
    m["ram_gib"] = round(int(mem.group(1)) / 1048576, 1) if mem else None
    m["laptop"] = bool(glob.glob("/sys/class/power_supply/BAT*"))   # power-capped GPU and CPU
    osr = re.search(r'PRETTY_NAME="([^"]+)"', read("/etc/os-release"))
    m["os"] = (osr.group(1) if osr else platform.system()) + ", kernel " + platform.release()
    return m


def build_info(version=""):
    cache = read(os.path.join(BUILD, "CMakeCache.txt"))
    arch = re.search(r"^CMAKE_CUDA_ARCHITECTURES:\w+=(.*)$", cache, re.M)
    cuda = ""
    for path in sorted(os.listdir(os.path.join(BUILD, "CMakeFiles"))) if os.path.isdir(os.path.join(BUILD, "CMakeFiles")) else []:
        r = re.search(r'CMAKE_CUDA_COMPILER_VERSION "([^"]+)"', read(os.path.join(BUILD, "CMakeFiles", path, "CMakeCUDACompiler.cmake")))
        if r:
            cuda = r.group(1)
    if version:   # a version without a public commit (a release package)
        return {"commit": version, "cuda": cuda, "cuda_arch": arch.group(1) if arch else ""}
    commit = sh(f"git -C {shlex.quote(ROOT)} rev-parse --short HEAD")
    dirty = sh(f"git -C {shlex.quote(ROOT)} status --porcelain --untracked-files=no") != ""
    return {"commit": commit + ("+modified" if dirty else ""), "cuda": cuda, "cuda_arch": arch.group(1) if arch else ""}


def run(label, args, cool_to, max_wait, runs, parse=None):
    """Run build/<args>, return the values of its RESULT line, or of parse(output), or None; keep a record."""
    t_cpu, waited = cool(cool_to, max_wait)
    gpu_t = sh("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits")
    cmd = [os.path.join(BUILD, args[0])] + args[1:]
    print(f"[{datetime.datetime.now():%H:%M:%S}] {label}: {' '.join(shlex.quote(a) for a in cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    vals = parse(p.stdout) if parse else {}
    for line in p.stdout.splitlines() if not parse else []:
        if line.startswith("RESULT "):
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
    rec = {"label": label, "cmd": cmd, "exit": p.returncode, "seconds": round(time.time() - t0, 1),
           "cpu_temp_start": t_cpu, "gpu_temp_start": gpu_t, "result": vals}
    if p.returncode != 0 or not vals:
        rec["output_tail"] = p.stdout[-4000:]
        print(f"    failed (exit {p.returncode}); see the JSON record", flush=True)
    else:
        print("    " + " ".join(f"{k}={v:g}" for k, v in vals.items()), flush=True)
    runs.append(rec)
    return vals or None, p.stdout


def hw_probe(runs):
    exe = os.path.join(BUILD, "hw_probe")
    if not os.path.exists(exe):
        return {}, ""
    print("hw_probe (GPU, PCIe and DRAM bandwidth)", flush=True)
    out = subprocess.run([exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    r = {}
    v = [float(x) for x in re.findall(r"read kernel \d+ blocks x256: ([\d.]+) GB/s", out)]
    if v:
        r["vram_read_gbs"] = max(v)
    m = re.search(r"64 MiB: H2D pinned ([\d.]+)", out)
    if m:
        r["pcie_h2d_gbs"] = float(m.group(1))
    v = [float(x) for x in re.findall(r"read\s+\d+ threads: ([\d.]+) GB/s", out)]
    if v:
        r["cpu_dram_read_gbs"] = max(v)
    m = re.search(r"contention: CPU read 8thr ([\d.]+) GB/s", out)
    if m:
        r["cpu_dram_read_during_dma_gbs"] = float(m.group(1))
    runs.append({"label": "hw_probe", "result": r})
    return r, out


def test_image(path):
    """A 640x480 PPM (300 image tokens): gradients and shapes, deterministic."""
    w, h = 640, 480
    px = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            px[i], px[i + 1], px[i + 2] = (x * 255) // w, (y * 255) // h, 255 if (x // 80 + y // 80) % 2 else 40
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h) + bytes(px))


def ctest(runs):
    print("ctest (correctness on this machine)", flush=True)
    t0 = time.time()
    p = subprocess.run(["ctest", "--test-dir", BUILD], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", p.stdout)
    failed = re.findall(r"^\s*\d+ - (\S+) \(Failed\)", p.stdout, re.M)
    disabled = len(re.findall(r"Not Run \(Disabled\)|\*\*\*Not Run \(Disabled\)", p.stdout))
    r = {"exit": p.returncode, "summary": m.group(0) if m else p.stdout[-300:], "failed": failed, "disabled": disabled,
         "seconds": round(time.time() - t0)}
    print("    " + r["summary"] + (f", {disabled} disabled" if disabled else ""), flush=True)
    runs.append({"label": "ctest", "result": r})
    return r


def slug(s):
    s = re.sub(r"\(r\)|\(tm\)|cpu @.*|nvidia|geforce|intel|amd|core|processor", "", s.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:32]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--kv", default="q4_0", choices=["q4_0", "q8_0", "f16"])
    ap.add_argument("--name", default="", help="your name or GitHub handle (optional)")
    ap.add_argument("--ram", default="", help='memory type and channels, e.g. "DDR4-2666, 2 channels" (optional)')
    ap.add_argument("--notes", default="", help="anything unusual: laptop power mode, desktop VRAM use, build changes")
    ap.add_argument("--version", default="", help="label for the code version instead of the git commit")
    ap.add_argument("--cool-to", type=float, default=70.0, help="CPU package temperature to wait for before each run")
    ap.add_argument("--max-wait", type=float, default=120.0)
    a = ap.parse_args()
    if not os.path.exists(os.path.join(BUILD, "q38_eval")):
        raise SystemExit("build/q38_eval not found: build first (README, Quick start)")
    if not os.path.exists(MTP):
        raise SystemExit(f"model not found: {MTP} (set Q38_MTP_MODEL)")

    rec = {"schema": 1, "date": datetime.date.today().isoformat(), "submitter": a.name, "notes": a.notes,
           "machine": machine(), "build": build_info(a.version), "quick": a.quick}
    if a.ram:
        rec["machine"]["ram_type"] = a.ram
    print("machine: " + json.dumps(rec["machine"]) + "\nbuild: " + json.dumps(rec["build"]), flush=True)
    if pkg_temp() is None:
        print("(no CPU package temperature sensor: runs are not spaced by temperature)")
    runs = []
    probe, probe_text = hw_probe(runs)
    res = {"kv": a.kv}
    # (name, context depth in tokens, KV capacity): 64K of code; 128K = a full 131072-position context (the depth
    # leaves room for a task and 256 generated tokens)
    for name, depth, ctx in [("64k", 65536, 66560), ("128k", 130048, 131072)][:1 if a.quick else 2]:
        v, _ = run(f"long context {name}", ["q38_eval", MTP, "longctx", "--depth", str(depth), "--ctx", str(ctx), "--kv",
                   a.kv, "--gen", "256", "--nprompts", "2" if a.quick else "4"], a.cool_to, a.max_wait, runs)
        if v:
            res[f"spec_{name}_tok_s"] = v.get("spec_tok_s")
            res[f"acceptance_{name}"] = v.get("acceptance")
            res[f"plain_{name}_tok_s"] = v.get("plain_tok_s")
            res[f"prefill_{name}_tok_s"] = v.get("prefill_tok_s")
    if os.path.exists(MMPROJ):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "image.ppm")
            test_image(img)
            def encode_ms(out):
                m = re.search(r"encode (\d+) ms", out)
                return {"encode_ms": float(m.group(1))} if m else {}
            v, _ = run("image encode (300 tokens)", ["q38_eval", MTP, "vision", "--no-mtp", "--mmproj", MMPROJ,
                       "--image", img, "--ctx", "4096"], a.cool_to, a.max_wait, runs, encode_ms)
            if v:
                res["image_encode_300_ms"] = v["encode_ms"]
    rec["results"] = res
    rec["hw_probe"] = probe
    rec["tests"] = ctest(runs) if a.tests else None
    rec["runs"] = runs
    rec["hw_probe_output"] = probe_text

    mc = rec["machine"]
    os.makedirs(os.path.join(ROOT, "bench", "community"), exist_ok=True)
    name = f"{rec['date']}_{slug(mc.get('gpu', 'gpu'))}_{slug(mc.get('cpu', 'cpu'))}.json"
    path = os.path.join(ROOT, "bench", "community", name)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")

    def num(k, fmt="{:.1f}"):
        return fmt.format(res[k]) if res.get(k) is not None else "–"
    t = rec["tests"]
    tests = "not run" if t is None else ("pass" if t["exit"] == 0 else f"{len(t['failed'])} failed")
    gpu = (f"{mc.get('gpu', '?')}{' (laptop)' if mc.get('laptop') else ''} {round(int(mc.get('vram_mib', 0) or 0) / 1024)} GB, "
           f"PCIe {mc.get('pcie_gen', '?')}.0 x{mc.get('pcie_width', '?')}")
    cpu_name = re.sub(r"\((R|TM)\)|\bCPU\b|@.*|\d+-Core Processor|Processor", "", mc.get("cpu", "?"))
    cpu = f"{' '.join(cpu_name.split())} ({mc.get('cpu_cores', '?')} cores)"
    ram = f"{mc.get('ram_gib', '?')} GiB" + (f" {a.ram}" if a.ram else "")
    row = (f"| {rec['date']} | {gpu} | {cpu} | {ram} | {num('spec_64k_tok_s')} | {num('spec_128k_tok_s')} | "
           f"{num('plain_64k_tok_s')} | {num('prefill_64k_tok_s', '{:.0f}')} | {num('prefill_128k_tok_s', '{:.0f}')} | "
           f"{num('image_encode_300_ms', '{:.0f}')} | {tests} | `{rec['build']['commit']}` | [json](bench/community/{name}) | "
           f"{'; '.join(x for x in (a.name, a.notes, '' if a.kv == 'q4_0' else 'KV ' + a.kv) if x)} |")
    print(f"\nwrote {os.path.relpath(path, ROOT)}\nrow for BENCHMARKS.md:\n{row}")


if __name__ == "__main__":
    main()
