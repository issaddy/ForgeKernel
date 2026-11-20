# Online warp normalization and weighted accumulation

## Purpose

Combine successive tiles without materializing an entire probability row. Keep the running maximum, sum, and weighted sum in FP32 registers. This is the reduction and rescaling primitive used by tiled Attention.

## Minimal runnable example

[The weighted reduction probe](probes/online_weighted_softmax.py) streams three segments of 32 values. One warp owns a row. Each tile reduces its maximum, then its exponential sum and weighted sum through warp shuffles.

```sh
bash gpu_entry.sh cmd python3 /path/to/online_weighted_softmax.py
```

Let `m`, `l`, and `o` denote the running max, denominator and weighted numerator. For a nonempty new tile, pick `m_new = max(m, tile_max)`. Multiply **both** old sums by `exp(m - m_new)`, then add the new tile contributions weighted by `exp(score - m_new)`. The final outputs are `o / l` and `m + log(l)`.

## Gotchas

- A fully masked tile leaves the running state unchanged. In particular, skip the initial all-masked tile before subtracting two negative infinities.
- Preserve warp convergence through the shuffles. Mask loads and neutralize unused lanes. The probe fills masked input with NaNs to check that it has no effect.
- Compare rows where a later tile raises the maximum and rows where it leaves the maximum unchanged. A split that misses the former can hide a missing numerator rescale.
- The probe uses finite logits and one value component. A complete Attention implementation also needs QK/PV fragment mappings, causal and batch offsets, multiple value components, and the expected empty-row convention.

## Sources

- [FlashAttention, online softmax and tiling](https://arxiv.org/abs/2205.14135) and [NVIDIA CuTe DSL architecture API](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_arch.html).
- Porting basis: SM90/SM100 `softmax_segment_ilp` and [SM120 masked warp softmax](masked_warp_softmax.md).
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: `PROBE PASS max_diff=9.5367431640625e-07` across output and finite log-sum-exp compared with CPU PyTorch. The empty row returned zero and negative-infinity log-sum-exp.
