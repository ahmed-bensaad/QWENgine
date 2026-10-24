## What and why

<!-- What the change does and the reasoning or measurement behind it. -->

## Checks

<!-- Benchmark submission (BENCHMARKS.md): only the first two boxes apply. -->
- [ ] Benchmark submission: `bench/community/<record>.json` added, and its row appended to the table in BENCHMARKS.md
- [ ] Benchmark submission: unmodified code, or the changes are described in the notes column

- [ ] `cd build && ctest; echo $?`: exit status 0 (say on which machine, and which tests were disabled)
- [ ] Performance change: `bench/ab.py` record added to `bench/results/`, entry added to `docs/optimization_log.md`
- [ ] Change for other hardware: no slowdown on the reference machine (`bench/ab.py`, or ask the maintainer to run it)
- [ ] Change meant to keep results identical: bit-identity checked (`golden --save` + `cmp` under `--cap-mib`)
- [ ] Documentation updated (ARCHITECTURE / SERVER / DEVELOPMENT) where behavior changed

## Measurements

<!-- ab.py output (per-pair ratios and geometric mean), community_bench.py records, or "n/a". -->
