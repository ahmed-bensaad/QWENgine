# QWENgine optimization log

Every change made to speed the engine up is recorded here, including the ones that failed: hypothesis, change,
measurements before and after, correctness checks and the decision (KEPT or REVERTED). Entries are in
chronological order and numbered in the order they were run. Raw records live in `bench/results/` and are never
modified.

Measurement method. End-to-end decode (or prefill) throughput decides. Builds are compared with `bench/ab.py` (ABBA
order, CPU cooled to 70 °C before every run) on `q38_eval specset` (speculative throughput over 16 corpus prompts),
because laptop thermals move single runs by ±5-10% and plan changes alter the text and acceptance of any single
prompt ([DEVELOPMENT.md](DEVELOPMENT.md#comparing-two-versions)). For prefill, `pbench --depth N` with
`--metric prefill_tok_s`.

Starting point: the first public release. Its numbers on the reference machine are in the [README](../README.md#performance)
and [BENCHMARKS.md](../BENCHMARKS.md).

## Entry format

```markdown
## Experiment N — short title (KEPT | REVERTED)
- Hypothesis: what limits the speed now (a profile, a measurement), and why the change should help.
- Change: what the code does differently.
- Correctness: ctest result; bit-identity checks for changes meant to keep results identical.
- Result: bench/ab.py per-pair ratios and their geometric mean, before -> after. Record: bench/results/<file>.json.
```

## Experiments

None yet.
