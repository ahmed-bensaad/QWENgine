#!/usr/bin/env python3
"""Thermally controlled A/B benchmark.

The laptop CPU runs at its thermal limit during decode, and consecutive runs drift by +-5-10%. This script waits
until the CPU package has cooled to --cool-to degrees C before every run (bounded by --max-wait), alternates the
order ABBA BAAB ... so drift cancels, and reports each pair's B/A ratio and their geometric mean.

  bench/ab.py --a "build/q38_eval M specset ..." --b "/path/other/q38_eval M specset ..." --metric specset_tok_s \
              --rounds 2 --label my_change

Each command must print a line "RESULT key=value ...". Results (all runs) are written to bench/results/ as JSON
when --label is given.
"""
import argparse
import datetime
import glob
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time


def pkg_temp():
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(z + "/type").read().strip() == "x86_pkg_temp":
                return int(open(z + "/temp").read()) / 1000.0
        except OSError:
            pass
    return None


def cool(target, max_wait):
    t0 = time.time()
    t = pkg_temp()
    while t is not None and t > target and time.time() - t0 < max_wait:
        time.sleep(2)
        t = pkg_temp()
    return t, time.time() - t0


def run(cmd, metric):
    out = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    vals = {}
    for line in out.splitlines():
        if line.startswith("RESULT "):
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
    if metric not in vals:
        sys.stderr.write(out[-3000:])
        raise SystemExit(f"metric {metric} not found in output of: {cmd}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--rounds", type=int, default=2, help="ABBA blocks (2 runs of each per block)")
    ap.add_argument("--cool-to", type=float, default=70.0)
    ap.add_argument("--max-wait", type=float, default=300.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    runs = []
    for r in range(args.rounds):
        order = "ABBA" if r % 2 == 0 else "BAAB"
        for which in order:
            t_start, waited = cool(args.cool_to, args.max_wait)
            vals = run(args.a if which == "A" else args.b, args.metric)
            runs.append({"which": which, "round": r, "temp_start": t_start, "waited_s": round(waited, 1), **vals})
            print(f"round {r} {which}: {args.metric}={vals[args.metric]:.4f} (start {t_start} C, waited {waited:.0f} s)",
                  flush=True)
    # pair consecutive A/B inside each block: ABBA -> (A1,B1),(B2,A2)
    ratios = []
    for r in range(args.rounds):
        blk = [x for x in runs if x["round"] == r]
        a = [x[args.metric] for x in blk if x["which"] == "A"]
        b = [x[args.metric] for x in blk if x["which"] == "B"]
        ratios += [bb / aa for aa, bb in zip(a, b)]
    gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    mean_a = sum(x[args.metric] for x in runs if x["which"] == "A") / sum(1 for x in runs if x["which"] == "A")
    mean_b = sum(x[args.metric] for x in runs if x["which"] == "B") / sum(1 for x in runs if x["which"] == "B")
    print(f"A mean {mean_a:.4f}  B mean {mean_b:.4f}  B/A per pair: " + " ".join(f"{x:.3f}" for x in ratios) +
          f"  geomean {gm:.4f} ({(gm - 1) * 100:+.1f}%)")
    if args.label:
        os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = os.path.join(os.path.dirname(__file__), "results", f"{ts}_ab_{re.sub(r'[^A-Za-z0-9_-]', '_', args.label)}.json")
        with open(path, "w") as f:
            json.dump({"label": args.label, "a": args.a, "b": args.b, "metric": args.metric, "cool_to": args.cool_to,
                       "runs": runs, "ratios": ratios, "geomean": gm}, f, indent=1)
        print("written", path)


if __name__ == "__main__":
    main()
