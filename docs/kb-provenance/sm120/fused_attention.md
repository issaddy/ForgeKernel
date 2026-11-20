# Fused Attention correctness baseline

## Purpose

Exercise the online reduction in a single kernel that consumes real Q, K and V tensors. The probe is a correctness baseline for optimized SM120 Attention implementations.

## Minimal runnable example

[The Attention probe](probes/fused_attention.py) assigns one warp to each query. It computes QK in FP32 from BF16 inputs, updates the online maximum and normalizer over three K tiles, and accumulates eight PV components in registers. It writes both the normalized output and log-sum-exp.

```sh
bash gpu_entry.sh cmd python3 /path/to/fused_attention.py
```

The test has four batches, five queries per batch, head dimension 16, value dimension 8 and up to 96 keys. It checks causal and noncausal masks, a per-batch query offset, empty rows and partial key tiles. The CPU calculation materializes the score matrix separately for comparison.

Each mask mode runs three variants. Stage count 0 loads K and V directly. Stage counts 1 and 2 load K and V via two TMA copies per tile, with one transaction barrier accounting for both. Both variants drain the last active slot and synchronize readers before reusing a slot. The two-slot variant can prefetch the next tile. A masked tail can still be fetched physically, so the consumer masks both score and value reads.

## Measured comparison

RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1, NCU 2025.3.1. Five profiled launches per variant, BF16 Q/K/V, FP32 output and accumulation, four batches with valid key counts 0/33/65/96.

| Path | Noncausal GPU active-time median (range) | Causal GPU active-time median (range) | Registers/thread, noncausal / causal | Dynamic SMEM/CTA |
| --- | ---: | ---: | ---: | ---: |
| Direct | 5.440 µs (5.408–5.984) | 6.144 µs (6.048–6.304) | 70 / 70 | 0 B |
| One TMA slot | 6.720 µs (6.624–6.784) | 6.752 µs (6.624–6.816) | 47 / 47 | 1,664 B |
| Two TMA slots | 5.760 µs (5.664–5.824) | 5.792 µs (5.792–6.016) | 44 / 46 | 3,200 B |

Local load and store sectors were zero for all variants. Two slots improved on one slot in both mask modes. The direct path remained faster by the median in the noncausal case. This is a small SIMT baseline, with one CTA per query and repeated K/V traffic between CTAs. Recheck the ordering on target shapes and a Tensor Core implementation.

- `[PRICED: +14.29%, noncausal one-slot to two-slot GPU active-time reduction on the measured shape]`
- `[PRICED: +14.22%, causal one-slot to two-slot GPU active-time reduction on the measured shape]`

```sh
bash gpu_entry.sh cmd 'ncu --csv --kernel-name "regex:.*cutlass_attention.*" --launch-count 30 --metrics gpu__time_active.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum python3 /path/to/fused_attention.py'
```

## Gotchas

- Apply the causal offset and valid-key limit before loading K and V. Masked padding contains NaNs in this probe.
- If a new tile raises the maximum, rescale every PV accumulator component and the denominator with the same factor.
- This variant uses SIMT dot products to isolate numerical behavior. It provides a comparison point for a Tensor Core, staged Attention kernel.
- Time an optimized candidate at actual workload shapes, with the same masking, value dimension and output contract.

## Sources

- [FlashAttention tiling and online softmax](https://arxiv.org/abs/2205.14135) and [online weighted softmax probe](../../../hw_kb/cutedsl/sm120/online_weighted_softmax.md).
- GPU certification on RTX PRO 5000 72GB Blackwell, CuTe DSL 4.8.1: all six cases returned `PROBE PASS max_diff=4.76837158203125e-07` across output and finite log-sum-exp.
