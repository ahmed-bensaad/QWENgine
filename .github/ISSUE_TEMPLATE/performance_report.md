---
name: Performance or hardware report
about: A performance regression, a test failure on your hardware, or a question about results on your machine
labels: performance
---

<!-- To share benchmark numbers from your machine, please open a pull request instead: run
     `python3 bench/community_bench.py --tests` and follow BENCHMARKS.md ("Add your machine"). -->

**Machine**
- GPU (model, VRAM, PCIe link: `nvidia-smi -q | grep -A3 "Link Width"`):
- CPU (model, cores, `cat /sys/devices/system/cpu/vulnerabilities/gather_data_sampling`):
- RAM (size, type, channels):
- Build (`git rev-parse HEAD`, `-DCMAKE_CUDA_ARCHITECTURES` value, CUDA version):

**What happened**
<!-- The command, the output (RESULT lines of q38_eval, `ctest --output-on-failure -R <test>`), and what you expected.
     A bench/community_bench.py JSON record says most of the above at once: attach it if you have one. -->

```
```

**Notes**
<!-- Temperatures, other GPU users (desktop VRAM), power mode. For a regression: the last good commit. -->
